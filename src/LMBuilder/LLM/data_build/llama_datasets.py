# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/29 4:05 PM
# @File: llama_datasets
# @Email: mlshenkai@163.com
from itertools import chain
from typing import List, Dict, Any, Mapping
import numpy as np
from datasets import load_dataset, interleave_datasets
from transformers import LlamaTokenizer
from transformers.training_args import TrainingArguments
import torch

#
tokenizer = LlamaTokenizer.from_pretrained(
    "/llm/base_model_resources/chinese_llama/merge_chinese_llama_lora_13b"
)

block_size = 512


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
dataset_zh = load_dataset("json", data_files="/llm/data_resources/base/data/zh/*.jsonl", split="train",streaming=True)
dataset_en = load_dataset("json", data_files="/llm/data_resources/base/data/en/*.jsonl",  split="train",streaming=True)
dataset_translate = load_dataset("json", data_files="/llm/data_resources/base/data/translate/*.jsonl", split="train", streaming=True)
dataset_zh = dataset_zh.remove_columns("meta")
dataset_en = dataset_en.remove_columns("meta")
dataset_translate = dataset_translate.remove_columns("meta")

dataset = interleave_datasets([dataset_zh, dataset_en, dataset_translate],probabilities=[0.4, 0.4, 0.2], seed=6886)
train_args = TrainingArguments("")
with train_args.main_process_first():
    tokenized_dataset = dataset.map(
        tokenize_func,
        batched=True,
        # keep_in_memory=False,
        remove_columns="text",
    )
with train_args.main_process_first():
    grouped_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        # keep_in_memory=False
    )
processed_dataset = grouped_dataset
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset=processed_dataset, batch_size=2, collate_fn=fault_tolerance_data_collator
)
for data in dataloader:
    print(data)
