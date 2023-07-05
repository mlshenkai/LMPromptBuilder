# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/13 1:39 PM
# @File: llama_predict_lora_pl
# @Email: mlshenkai@163.com
import argparse

import pandas as pd
import pyrootutils
from loguru import logger
from tqdm import tqdm
from transformers import GenerationConfig

from examples.finetuning_examples.peft.llama.llama_finetune_lora import load_data

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
from src.LMBuilder.LLM.models import LlamaArgs
import lightning as L
from src.LMBuilder.LLM.models import LlamaModelPL


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
        "use_peft": False,
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
        "peft_name": "/code-online/shenkai/LMPromptBuilder/examples/finetuning_examples/peft/llama/outputs",
    }
    model_args = LlamaArgs()
    model_args.update_from_dict(parse_args)


    train_data = load_data(args.train_file)
    logger.debug("train_data: {}".format(train_data[:10]))
    train_df = pd.DataFrame(train_data, columns=["instruction", "input", "output"])
    test_df = train_df[:10]
    train_df = train_df[10:]
    def get_prompt(arr):
        if arr["input"].strip():
            return f"""Below is an instruction that describes a task. Write a response that appropriately 
                completes the request.\n\n### Instruction:\n{arr['instruction']}\n### 
                Input:\n{arr['input']}\n\n### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that 
                appropriately completes the request.\n\n### Instruction:\n{arr['instruction']}\n\n### Response:"""

    test_df["prompt"] = test_df.apply(get_prompt, axis=1)



    model_pl = LlamaModelPL(model_config=model_args)
    model = model_pl.model
    tokenizer = model_pl.tokenizer
    fabric = L.Fabric(
        accelerator="gpu",
        devices=1,
        # precision="16",
        # strategy="deepspeed"
    )
    fabric.launch()
    model.eval()

    model = fabric.setup(model)
    generation_config = GenerationConfig(
        max_new_tokens=model_args.max_length,
        temperature=model_args.temperature,
        top_p=model_args.top_p,
        top_k=model_args.top_k,
        num_beams=model_args.num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=model_args.num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
    )
    all_output = []
    sentences = test_df["prompt"].tolist()
    for batch in tqdm(
            [
                sentences[i : i + 1]
                for i in range(0, len(sentences), 1)
            ],
            desc="generator output",
            disable=model_args.silent,
    ):

        inputs = tokenizer(batch,  padding=True, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, generation_config=generation_config)
        for idx, (prompt_text, generated_sequence) in enumerate(
                zip(batch, outputs.sequences)
        ):
            # decode text
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            prompt_len = len(prompt_text)
            gen_text = text[prompt_len:]
            total_sequence = prompt_text + gen_text
            all_output.append(total_sequence)

    test_df["predict_after"] = all_output
    test_df.to_csv()
    print(test_df)

if __name__ == "__main__":
    main()
