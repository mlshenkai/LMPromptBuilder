# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/7/3 10:22 AM
# @File: llama_sft_tuning
# @Email: mlshenkai@163.com
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
from pyrootutils import pyrootutils

from src.LMBuilder.LLM.data_build.build_dataset import (
    build_sft_dataset,
    DataCollatorForSupervisedDataset,
)

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
import os
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    TrainingArguments,
    AutoConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from src.LMBuilder.LLM.utils.llama_utils import smart_tokenizer_and_embedding_resize
import torch
from peft import (
    PeftModel,
    get_peft_model,
    LoraConfig,
    TaskType,
    get_peft_model_state_dict,
)
from src.LMBuilder.LLM.callback.model_checkpoint_callback import SavePeftModelCallback

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="/code-online/shenkai/LLMTuning/sk_finetuning_7b",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    train_file: Optional[str] = field(
        default="/llm/data_resources/alpaca/zh/*.json", metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The datasets processed stored"}
    )

    max_seq_length: Optional[int] = field(default=512)





@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable: Optional[str] = field(default="q_proj,v_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[int] = field(default=32)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    force_resize_embeddings: bool = field(default=False)
    max_steps: int = field(default=10000)
    seed: int = field(default=8999)
    do_train: bool = field(default=True)
    fp16: bool = field(default=True)
    per_device_train_batch_size: int = field(default=2)
    lr_scheduler_type: str = field(default="cosine")
    learning_rate: float = field(default=1e-4)
    warmup_ratio: float = field(default=0.03)
    weight_decay: int = field(default=0)
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="steps")
    save_total_limit: int = field(default=3)
    save_steps: int = field(default=500)
    gradient_accumulation_steps: int = field(default=1)
    deepspeed: str = field(default=f"{project_path}/configs/ds_zero2_no_offload.json")
    report_to: str = field(default="wandb")
    output_dir: str = field(default=f"{project_path}/sk_sft_7b")
    overwrite_output_dir: bool = field(default=True)




def main():

    # parser = HfArgumentParser(
    #     (ModelArguments, DataTrainingArguments, MyTrainingArguments)
    # )
    #
    # parse_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments = ModelArguments()
    data_args: DataTrainingArguments = DataTrainingArguments()
    training_args: MyTrainingArguments = MyTrainingArguments(output_dir=f"{project_path}/sk_sft_7b")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    last_checkpoints = None
    if (
        os.path.exists(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoints = get_last_checkpoint(training_args.output_dir)
        if last_checkpoints is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoints is not None
            and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoints}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.tokenizer_name:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    else:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=None,
        )

    torch_type = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        # torch_dtype=torch_type,

        low_cpu_mem_usage=True,
    )
    logger.info(f"len(tokenizer): {len(tokenizer)}")

    embeddings_size = model.get_input_embeddings().weight.shape[0]
    if embeddings_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if training_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, training_args.peft_path)

    logger.info("init nwe peft model")
    target_modules = training_args.trainable.split(",")
    modules_to_save = training_args.modules_to_save
    if modules_to_save is not None:
        modules_to_save = modules_to_save.split(",")
    lora_rank = training_args.lora_rank
    lora_dropout = training_args.lora_dropout
    lora_alpha = training_args.lora_alpha
    logger.info(f"target_modules: {target_modules}")
    logger.info(f"lora_rank: {lora_rank}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    logger.info(f"mode.modules_to_save: {model.modules_to_save}")
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # load dataset
    with training_args.main_process_first():
        sft_datasets = build_sft_dataset(
            data_path=[data_args.train_file],
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            data_cache_dir=data_args.data_cache_dir,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
        )
        train_dataset = sft_datasets["train"]
        valid_dataset = sft_datasets["test"]
        logger.info(f"load dataset : {len(train_dataset)}")
        logger.info("training example: ")
        logger.info(tokenizer.decode(train_dataset[0]["input_ids"]))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # training_args.report_to
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.add_callback(SavePeftModelCallback)
    resume_checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        resume_checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoints is not None:
        resume_checkpoint = last_checkpoints
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics(
        "train",
        metrics,
    )
    trainer.save_metrics( "train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    # wandb.init(project="llama-7b-sft", group="llama-7b-sft")
    main()
    # wandb.finish()
