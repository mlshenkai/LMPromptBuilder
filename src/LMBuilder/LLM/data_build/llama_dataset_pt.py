# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/14 11:14 AM
# @File: llama_dataset_pt
# @Email: mlshenkai@163.com
import pyrootutils
import lightning.pytorch as pl
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

from src.LMBuilder.LLM.data_build import DataGenerator, _data_collator


project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
import datasets


class LlamaPTDatasetPL(pl.LightningDataModule):
    def __init__(self, data_args, tokenizer):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.block_size = min(
            self.data_args.block_size, self.tokenizer.model_max_length
        )
        self.datasets = None

        self.dataset_gen = DataGenerator(self.data_args, self.tokenizer)

    def setup(self, stage: str) -> None:
        self.datasets = datasets.Dataset.from_generator(
            self.dataset_gen.generator,
            keep_in_memory=False,
            streaming=True,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = self.datasets
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.data_args.train_batch_size,
            collate_fn=self._data_collator,
        )

    @staticmethod
    def _data_collator(features: List) -> Dict[str, Any]:
        return _data_collator(features)


# if __name__ == "__main__":
#     pass
#     # data_dir = Path("/code-online/shenkai/LLMTuning/resources/data")
#     # dataset_names = []
#     # for item in data_dir.iterdir():
#     #     if item.is_dir():
#     #         dataset_names.append(item.name)
#     # print(dataset_names)
#
#     from transformers import LlamaTokenizer
#
#     #
#     tokenizer = LlamaTokenizer.from_pretrained(
#         "/llm/base_model_resources/chinese_llama/merge_chinese_llama_lora_13b"
#     )
#     data_args = LlamaPTTrainDataArgs()
#     dataset_pl = LlamaPTDatasetPL(data_args=data_args, tokenizer=tokenizer)
#     dataset_pl.setup("fit")
#     train_data_loader = dataset_pl.train_dataloader()
#     print(train_data_loader)
