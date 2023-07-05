# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/12 3:10 PM
# @File: llama_finetune_lora_pl
# @Email: mlshenkai@163.com
import argparse
import pyrootutils

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
import pandas as pd
from loguru import logger
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from examples.finetuning_examples.peft.llama.llama_finetune_lora import load_data
from src.LMBuilder.LLM.models import LlamaDataset
from src.LMBuilder.LLM.models import LlamaModelPL
from src.LMBuilder.LLM.models import LlamaArgs, LoraArgs
from src.common.save_peft_model import PeftModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import lightning.pytorch as pl
from functools import partial

# project_path = ""


def load_and_cache_examples(
    tokenizer, data, evaluate=False, data_args=None, no_cache: bool = None
):
    """
    create a LlamaDataset
    Args:
        data:
        evaluate:
        no_cache:

    Returns:
        LlamaDataset
    """
    mode = "dev" if evaluate else "train"
    return LlamaDataset(tokenizer, data_args, data, mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        default=f"{project_path}/resources/data/中英品名0512.txt",
        type=str,
        help="Training data file",
    )
    parser.add_argument(
        "--test_file",
        default=f"{project_path}/resources/data/中英品名0512.txt",
        type=str,
        help="Test data file",
    )
    parser.add_argument(
        "--model_type", default="llama", type=str, help="Transformers model type"
    )
    parser.add_argument(
        "--model_name",
        default="shibing624/chinese-alpaca-plus-7b-hf",
        type=str,
        help="Transformers model or path",
    )
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, help="Whether to run predict.")
    parser.add_argument(
        "--is_train_on_prompt", default=False, help="Whether to run language model task"
    )
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="Model output directory"
    )
    parser.add_argument(
        "--max_seq_length", default=128, type=int, help="Input max sequence length"
    )
    parser.add_argument(
        "--max_length", default=128, type=int, help="Output max sequence length"
    )
    parser.add_argument(
        "--num_epochs", default=20, type=float, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--eval_steps", default=50, type=int, help="Eval every X steps")
    parser.add_argument(
        "--save_steps", default=50, type=int, help="Save checkpoint every X steps"
    )
    args = parser.parse_args()
    parse_args = {
        "use_lora": True,
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": args.max_seq_length,
        "max_length": args.max_length,
        "per_device_train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "num_train_epochs": args.num_epochs,
        "is_train_on_prompt": args.is_train_on_prompt,
        "output_dir": args.output_dir,
        "resume_from_checkpoint": args.output_dir,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "model_name": args.model_name,
        "no_cache": True,
    }
    model_args = LlamaArgs()
    model_args.update_from_dict(parse_args)
    peft_config = LoraArgs()

    model = LlamaModelPL(model_args, peft_config)
    train_data = load_data(args.train_file)
    logger.debug("train_data: {}".format(train_data[:10]))
    train_df = pd.DataFrame(train_data, columns=["instruction", "input", "output"])
    eval_df = train_df[:10]
    train_df = train_df[10:]
    train_dataset = load_and_cache_examples(
        model.tokenizer, train_df, data_args=model_args
    )
    eval_dataset = load_and_cache_examples(
        model.tokenizer, eval_df, data_args=model_args
    )
    data_collator = DataCollatorForSeq2Seq(
        model.tokenizer,
        return_tensors="pt",
        padding="max_length",
        max_length=model_args.max_seq_length + model_args.max_length,
    )
    train_dataset_loader = DataLoader(
        dataset=train_dataset,
        batch_size=model_args.train_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    eval_dataset_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=model_args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
    )
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy, activation_checkpointing=LlamaDecoderLayer
    )
    save_checkpoint_callback = PeftModelCheckpoint(model_args.output_dir)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[1, 2, 3, 4, 5, 6],
        strategy=DeepSpeedStrategy(stage=2),
        callbacks=[save_checkpoint_callback],
        max_epochs=1
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataset_loader,
        val_dataloaders=eval_dataset_loader,
    )


if __name__ == "__main__":
    main()
