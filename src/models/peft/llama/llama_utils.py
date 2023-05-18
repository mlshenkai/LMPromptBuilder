# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/16 4:32 PM
# @File: llama_utils
# @Email: mlshenkai@163.com


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n{input_text}\n\n### Response:\n\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n\n"
    ),
}


def preprocess_data(data):
    instruction, input_text, target_text, tokenizer, args = data

    if input_text:
        prompt = PROMPT_DICT["prompt_input"].format(
            instruction=instruction, input_text=input_text
        )
    else:
        prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)

    full_prompt = prompt + target_text + tokenizer.eos_token
    full_max_length = args.max_seq_length + args.max_length
    example = tokenizer(
        full_prompt,
        truncation=True,
        max_length=full_max_length,
        padding=False,
        add_special_tokens=False,
    )
    example["labels"] = example["input_ids"].copy()
    if not args.is_train_on_prompt:
        user_example = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        user_prompt_len = len(user_example["input_ids"])
        # set labels to full max length to adjust for DataCollatorForSeq2Seq padding
        example["labels"] = [-100] * (
            full_max_length - len(example["labels"]) + user_prompt_len
        ) + example["labels"][user_prompt_len:]
    else:
        example["labels"] = [-100] * (full_max_length - len(example["labels"])) + example["labels"]
    return example
