# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/28 8:05 PM
# @File: llama_pt_tuning
# @Email: mlshenkai@163.com
import os
import torch
from pyrootutils import pyrootutils

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
import wandb
from itertools import chain
from typing import List, Dict, Any, Mapping
import numpy as np
from datasets import load_dataset, interleave_datasets
from transformers import AutoConfig, LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from src.LMBuilder.LLM.data_build.llama_data_args import LlamaPTTrainDataArgs
from src.LMBuilder.LLM.models.llama.model_args import LlamaModelArgs, LoraModelArgs
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_utils import get_last_checkpoint

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "pt_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)




def main():

    model_args = LlamaModelArgs()
    peft_args = LoraModelArgs()
    data_args = LlamaPTTrainDataArgs()
    model_name_or_path = model_args.model_name_or_path
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, config=model_config
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path
    )
    model.resize_token_embeddings(len(tokenizer))
    target_modules = peft_args.lora_target_modules
    lora_r = peft_args.lora_r
    lora_alpha = peft_args.lora_alpha
    lora_dropout = peft_args.lora_dropout
    model_to_save = peft_args.module_to_save
    peft_config = LoraConfig(
        peft_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        modules_to_save=model_to_save,
    )
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda x, *_, **__: get_peft_model_state_dict(x, old_state_dict())
    ).__get__(model, type(model))

    block_size = min(data_args.block_size, tokenizer.model_max_length)
    def group_texts(examples):
    # Concatenate all texts.

        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    def tokenize_func(examples):
        output = tokenizer(examples["text"])
        return output

    def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.

        try:
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
        except ValueError: # quick fix by simply take the first example
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([features[0][k]] * len(features))
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                    else:
                        batch[k] = torch.tensor([features[0][k]] * len(features))

        return batch

    zh_data_files = {
        "train": "/llm/data_resources/base/data/zh/*.jsonl"
    }
    en_data_files = {
        "train": "/llm/data_resources/base/data/en/*.jsonl"
    }
    translate_data_files = {
        "train": "/llm/data_resources/base/data/translate/*.jsonl"
    }


    output_dir = "./sk_pt_train"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    last_checkpoint = get_last_checkpoint(output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=model_args.learning_rate,
        num_train_epochs=model_args.num_train_epochs,
        logging_dir=f"{output_dir}/logs",
        logging_steps=model_args.logging_steps,
        max_steps=model_args.max_steps,
        warmup_steps=model_args.warmup_steps,
        per_device_train_batch_size=model_args.per_device_train_batch_size,
        per_device_eval_batch_size=model_args.per_device_train_batch_size,
        gradient_accumulation_steps=model_args.gradient_accumulation_steps,
        # warmup_steps=model_args.warmup_steps,
        save_steps=model_args.save_steps,
        # save_steps=2,
        optim=model_args.optimizer,
        deepspeed=f"{project_path}/configs/ds_zero2_no_offload.json",
        save_strategy=model_args.save_strategy,
        save_total_limit=model_args.save_total_limit,
        fp16=model_args.fp16,
        remove_unused_columns=model_args.remove_unused_columns,
        report_to=model_args.report_to,
        # overwrite_output_dir=model_args.overwrite_output_dir,
    )
    with training_args.main_process_first():
        dataset_zh = load_dataset("json", data_files=zh_data_files, split="train", streaming=True)
        dataset_en = load_dataset("json", data_files=en_data_files, split="train", streaming=True)
        dataset_translate = load_dataset("json", data_files=translate_data_files, split="train", streaming=True)
        dataset_zh = dataset_zh.remove_columns("meta")
        dataset_en = dataset_en.remove_columns("meta")
        dataset_translate = dataset_translate.remove_columns("meta")
        dataset = interleave_datasets([dataset_zh, dataset_en, dataset_translate],probabilities=[0.4, 0.4, 0.2], seed=6886)
        tokenized_dataset = dataset.map(
            tokenize_func,
            batched=True,
            # keep_in_memory=False,
            remove_columns="text",
        )
        grouped_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            # keep_in_memory=False
        )


        processed_dataset = grouped_dataset

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,

    )
    print(f"last_checkpoint: {last_checkpoint}")
    trainer.add_callback(SavePeftModelCallback)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_state(trainer)


if __name__ == "__main__":
    wandb.init(project="llama-7b-pt", group="llama-7b-pt1")
    main()
    wandb.finish()
