# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/26 3:33 PM
# @File: llama_convert_feature
# @Email: mlshenkai@163.com
import os
import pyrootutils
import lightning.pytorch as pl
import torch
from typing import List, Dict, Any, Optional, Mapping
from itertools import chain


from commones.pool_executor import PoolExecutor

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
from src.data_build.llama_data_args import LlamaPTTrainDataArgs
from loguru import logger
from pathlib import Path
import datasets
import numpy as np


class LlamaConvertFeatures:
    def __init__(self, data_args, tokenizer):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.block_size = min(
            self.data_args.block_size, self.tokenizer.model_max_length
        )
        self.datasets = None
        self.dataset_names = []

    def load_dataset(self, dataset_names: list):
        lm_datasets = []
        for idx, dataset_name in enumerate(dataset_names):
            dataset_path = Path(self.data_args.dataset_dir).joinpath(dataset_name)
            file_name_list = dataset_path.glob("*.jsonl")
            for file_path in file_name_list:
                cache_dir = os.path.join(
                    self.data_args.data_cache_dir, dataset_name, file_path.stem
                )
                os.makedirs(cache_dir, exist_ok=True)
                try:
                    processed_dataset = datasets.load_from_disk(
                        cache_dir, keep_in_memory=False
                    )
                    logger.info(
                        f"training dataset-{dataset_name}-{file_path.stem} has been loaded from disk"
                    )
                except:
                    raw_dataset = datasets.load_dataset(
                        "json",
                        data_files=file_path.as_posix(),
                        cache_dir=cache_dir,
                        keep_in_memory=False,
                        streaming=True,
                    )
                    logger.info(f"{file_path.stem} has been loaded")
                    tokenized_dataset = raw_dataset.map(
                        self.tokenize_func,
                        batched=True,
                        # num_proc=10,
                        remove_columns="text",
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names={
                            k: os.path.join(cache_dir, f"tokenized.arrow")
                            for k in raw_dataset
                        },
                        desc="Running tokenizer on dataset",
                    )
                    grouped_datasets = tokenized_dataset.map(
                        self.group_texts,
                        batched=True,
                        num_proc=self.data_args.preprocessing_num_workers,
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names={
                            k: os.path.join(cache_dir, f"grouped.arrow")
                            for k in tokenized_dataset
                        },
                        desc=f"Grouping texts in chunks of {self.block_size}",
                    )
                    processed_dataset = grouped_datasets
                    # processed_dataset.save_to_disk(cache_dir)

    def convert_features(self):
        # self.load_dataset(["translate"])

        func_list = [
            (
                self.load_dataset,
                ["translate"]
            ),
            (
                self.load_dataset,
                ["en"]
            ),
            (
                self.load_dataset,
                ["zh"]
            )
        ]

        pool_executor = PoolExecutor(func_list, pool="process")
        pool_executor.execute()
        for result in pool_executor.get_results():
            continue


    def tokenize_func(self, examples):
        output = self.tokenizer(examples["text"])
        return output

    def group_texts(self, examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    @staticmethod
    def _data_collator(features: List) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(feature) for feature in features]

        first = features[0]

        batch = {}
        if "label" in first and first["label"] is not None:
            label = (
                first["label"].item()
                if isinstance(first["label"], torch.Tensor)
                else first["label"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = (
                    torch.long if type(first["label_ids"][0]) is int else torch.float
                )
                batch["labels"] = torch.tensor(
                    [f["label_ids"] for f in features], dtype=dtype
                )

        try:
            for k, v in first.items():
                if (
                    k not in ("label", "label_ids")
                    and v is not None
                    and not isinstance(v, str)
                ):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
        except ValueError:  # quick fix by simply take the first example
            for k, v in first.items():
                if (
                    k not in ("label", "label_ids")
                    and v is not None
                    and not isinstance(v, str)
                ):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([features[0][k]] * len(features))
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(
                            np.stack([features[0][k]] * len(features))
                        )
                    else:
                        batch[k] = torch.tensor([features[0][k]] * len(features))

        return batch

    @staticmethod
    def _get_dataset_names(data_dir: str):
        data_dir = Path(data_dir)
        dataset_names = []
        for item in data_dir.iterdir():
            if item.is_dir():
                dataset_names.append(item.name)
        return dataset_names


if __name__ == "__main__":
    pass
    # data_dir = Path("/code-online/shenkai/LLMTuning/resources/data")
    # dataset_names = []
    # for item in data_dir.iterdir():
    #     if item.is_dir():
    #         dataset_names.append(item.name)
    # print(dataset_names)

    from transformers import LlamaTokenizer

    #
    tokenizer = LlamaTokenizer.from_pretrained(
        "/llm/base_model_resources/chinese_llama/merge_chinese_llama_lora_13b"
    )
    data_args = LlamaPTTrainDataArgs()
    dataset_pl = LlamaConvertFeatures(data_args=data_args, tokenizer=tokenizer)
    dataset_pl.convert_features()
