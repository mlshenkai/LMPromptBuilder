# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/14 11:14 AM
# @File: llama_dataset_pt
# @Email: mlshenkai@163.com
import os
import random
import logging
import pyrootutils
import lightning.pytorch as pl
import torch
from typing import List, Dict, Any, Optional, Mapping
from itertools import chain

from datasets.formatting.formatting import LazyRow
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from datasets import Dataset

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
from src.LMBuilder.LLM.data_build import LlamaPTTrainDataArgs
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import datasets
import numpy as np
from queue import Queue
from threading import Thread
from functools import partial
logging.getLogger("datasets").setLevel(logging.ERROR)


class DataGeneratorThread(Thread):
    def __init__(self, data_args, tokenizer, max_samples=60000):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.dataset_path_infos = {"zh": [], "en": [], "translate": []}
        self.get_data_dir()
        self.max_samples: int = max_samples
        self.data_composition_weighs = self.data_args.data_composition_weighs
        self.data_queue = Queue(10000)
        self.block_size = min(data_args.block_size, tokenizer.model_max_length)
        self.daemon = True

    def generator(self):
        for i in range(self.max_samples):
            lm_dataset = []
            for dataset_name in list(self.dataset_path_infos.keys()):
                path_infos = self.dataset_path_infos[dataset_name]
                dataset_weight = self.data_composition_weighs[dataset_name]
                sample_path_infos = self.get_sample_data(path_infos, dataset_weight)
                for path_info in sample_path_infos:
                    cache_dir = os.path.join(
                        self.data_args.data_cache_dir,
                        dataset_name,
                        Path(path_info).stem,
                    )
                    os.makedirs(cache_dir, exist_ok=True)
                    raw_datasets = datasets.load_dataset(
                        "json",
                        data_files=path_info,
                        keep_in_memory=False,
                        cache_dir=cache_dir
                    )
                    tokenized_dataset = raw_datasets.map(
                        partial(self.tokenize_func, self.tokenizer),
                        batched=True,
                        num_proc=self.data_args.preprocessing_num_workers,
                        load_from_cache_file=True,
                        cache_file_names = {k: os.path.join(cache_dir, f'tokenized.arrow') for k in raw_datasets},
                        keep_in_memory=False,
                        remove_columns=["text", "meta"],
                    )
                    grouped_dataset = tokenized_dataset.map(
                        partial(self.group_texts, self.block_size),
                        batch_size=True,
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names = {k: os.path.join(cache_dir, f'grouped.arrow') for k in tokenized_dataset},
                        num_proc=self.data_args.preprocessing_num_workers,
                    )

                    processed_dataset = grouped_dataset
                    if len(lm_dataset) == 0:
                        lm_dataset = processed_dataset["train"]
                    else:
                        lm_dataset = datasets.concatenate_datasets(
                            [lm_dataset, processed_dataset["train"]]
                        )
                    logger.info(f"-------------dataset-i:{i} process: {os.getpid()}------------")

            for feature in lm_dataset:
                input_ids = feature["input_ids"]
                # pos = random.randint(
                #     int(0.2 * len(input_ids)) + 1, int(0.7 * len(input_ids))
                # )
                # loss_mask = [0] * pos + [1] * (len(input_ids) - pos)
                self.data_queue.put({"input_ids": input_ids})
    def run(self) -> None:
        self.generator()

    @staticmethod
    def tokenize_func(tokenizer, examples):
        output = tokenizer(examples["text"])
        return output

    def get_data_dir(self):
        dataset_names = self._get_dataset_names(self.data_args.dataset_dir)
        for dataset_name in dataset_names:
            dataset_path = Path(self.data_args.dataset_dir).joinpath(dataset_name)
            file_name_list = dataset_path.glob("*.jsonl")
            for file_path in file_name_list:
                self.dataset_path_infos[dataset_name].append(file_path.as_posix())

    @staticmethod
    def get_sample_data(path_infos: list, num_element: int):
        return random.sample(path_infos, num_element)

    @staticmethod
    def _get_dataset_names(data_dir: str):
        data_dir = Path(data_dir)
        dataset_names = []
        for item in data_dir.iterdir():
            if item.is_dir():
                dataset_names.append(item.name)
        return dataset_names

    @staticmethod
    def group_texts(block_size, examples):
        # Concatenate all texts.
        if isinstance(examples, LazyRow):
            examples = examples.data
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


def _data_collator(features: list, have_loss_mask=False):
    if not isinstance(features[0], Mapping):
        features = [vars(feature) for feature in features]
    max_length_in_example = max([len(feature["input_ids"]) for feature in features])
    batch = {"input_ids": [], "loss_mask": []}
    for idx in range(len(features)):
        feature = features[idx]
        input_ids = feature["input_ids"]
        loss_mask = feature["loss_mask"]
        seq_len = len(input_ids)
        input_ids = input_ids + [0] * (max_length_in_example - seq_len)
        loss_mask = loss_mask + [0] * (max_length_in_example - seq_len)
        assert len(input_ids) == len(loss_mask), f"input_ids: {len(input_ids)} != loss_mask: {len(loss_mask)}"
        batch["input_ids"].append(input_ids)
        batch["loss_mask"].append(loss_mask)
    batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
    batch["loss_mask"] = torch.tensor(batch["loss_mask"], dtype=torch.long)
    batch["labels"] = batch["input_ids"].clone()
    if not have_loss_mask:
        batch.pop("loss_mask")
    return batch

def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    import torch

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

class DataGenerator:
    def __init__(self, data_args, tokenizer, num_samples=60000, num_ratio=800):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.num_ratio = num_ratio

    def generator(self):
        data_generator_thread = DataGeneratorThread(self.data_args, self.tokenizer, self.num_samples)
        data_generator_thread.start()
        for i in range(data_generator_thread.max_samples*self.num_ratio):
            yield data_generator_thread.data_queue.get(True, 100)

if __name__ == "__main__":

    from transformers import LlamaTokenizer

    #
    tokenizer = LlamaTokenizer.from_pretrained(
        "/llm/base_model_resources/chinese_llama/merge_chinese_llama_lora_13b"
    )
    data_args = LlamaPTTrainDataArgs()
    data_gen = DataGenerator(data_args, tokenizer,60000,800)
    llm_dataset = Dataset.from_generator(
        data_gen.generator,
        keep_in_memory=False,
        streaming=True,
        # num_proc=data_args.preprocessing_num_workers,
    )
    dataloader = DataLoader(
        dataset=llm_dataset, batch_size=2, collate_fn=fault_tolerance_data_collator
    )
    for data in dataloader:
        logger.info("\n====================data============================\n")
        print(data)
