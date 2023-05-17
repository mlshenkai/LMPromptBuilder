# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/16 2:37 PM
# @File: llama_dataset
# @Email: mlshenkai@163.com
import os.path
from loguru import logger
import pickle
from torch.utils.data import Dataset
from multiprocessing import Pool
from tqdm.auto import tqdm
from src.models.peft.llama.llama_utils import preprocess_data
import datasets as hf_dataset



class LlamaDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        """
        build
        Args:
            tokenizer:
            args:
            data:
            mode:
        """
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cache_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s" % cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s" % args.cache_dir)

            data = [
                (instruction, input_text, target_text, tokenizer, args)
                for instruction, input_text, target_text in zip(
                    data["instruction"], data["input"], data["output"]
                )
            ]

            if (mode == "train" and args.use_multiprocessing) or (
                    mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_data, data, chunksize=chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def preprocess_batch_for_hf_dataset(example, tokenizer, args):
    data = (
        example["instruction"], example["input_text"], example["output"], tokenizer, args
    )
    example = preprocess_data(data)
    return example

def load_hf_dataset(data, tokenizer, args, mode):
    if isinstance(data, str):
        if data.endswith(".json") or data.endswith(".jsonl"):
            dataset = hf_dataset.load_dataset("json", data_files=data)
        elif os.path.isdir(data):
            dataset = hf_dataset.load_from_disk(data)
        else:
            dataset = hf_dataset.load_dataset(
                data, download_mode="force_redownload" if args.reprocess_input_data else "reuse_data_if_exists"
            )

        dataset = dataset["train"]
        if mode == "dev" and args.max_eval_samples is not None:
            max_eval_samples = min(len(dataset), args.max_eval_samples)
            dataset = dataset.select(range(max_eval_samples))
    else:
        dataset = hf_dataset.Dataset.from_pandas(data)

    dataset = dataset.shuffle().map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, args=args),
        batched=False, remove_columns=dataset.column_names
    ).filter(lambda x: len(x["input_ids"]) > 0)

    return dataset


