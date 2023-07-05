# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/7/3 11:37 AM
# @File: build_dataset
# @Email: mlshenkai@163.com
from pyrootutils import pyrootutils

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
import transformers
import datasets
import torch
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from loguru import logger
from typing import Union, List, Sequence, Dict
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
)


def build_sft_dataset(
    data_path: Union[List[str], str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    data_cache_dir=None,
    preprocessing_num_workers=None,
    stream: bool = False,
):
    def tokenize_func(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for instruction, input_str, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            if input_str is not None and input_str != "":
                instruction = instruction + "\n" + input_str
            source = prompt.format_map({"instruction": instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)
        tokenized_sources = tokenizer(sources, return_attention_mask=True)
        tokenized_targets = tokenizer(
            targets, return_attention_mask=True, add_special_tokens=False
        )
        all_input_ids = []
        all_labels = []
        for s, t in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {"input_ids": all_input_ids, "labels": all_labels}
        return results

    logger.info("start build dataset...")
    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]

    for file_path in data_path:
        dataset = datasets.load_dataset(
            "json",
            data_files=file_path,
            split="train",
            cache_dir=data_cache_dir,
            streaming=stream,
        )
        tokenized_dataset = dataset.map(
            tokenize_func,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=["instruction", "input", "output"],
            keep_in_memory=False,
            desc="preprocess on dataset",
        )
        processed_dataset = tokenized_dataset
        processed_dataset.set_format("torch")
        all_datasets.append(processed_dataset)

    all_datasets = interleave_datasets(all_datasets)
    all_datasets = all_datasets.train_test_split(test_size=0.05)

    return all_datasets


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
