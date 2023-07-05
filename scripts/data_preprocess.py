# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/21 11:06 AM
# @File: data_preprocess
# @Email: mlshenkai@163.com
import random
from pathlib import Path
import json
from tqdm import tqdm
import os
import pyrootutils

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)


class DataPreprocess:
    def __init__(self, data_dir: Path, save_dir: Path):
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True)
        self.data_dir = data_dir
        self.save_dir = save_dir

    def save_result(self, epoch: int, json_list):
        with open(
            self.save_dir.joinpath(f"{epoch}.jsonl").as_posix(), "w", encoding="utf-8"
        ) as f:
            for json_info in json_list:
                line = json.dumps(json_info, ensure_ascii=False)
                f.write(f"{line}\n")


class WuDaoDataProcess(DataPreprocess):
    def __init__(self, data_dir: Path, save_dir: Path):
        super().__init__(data_dir, save_dir)
        self.data_file_list = self.data_dir.glob("*.json")

    def process(self):
        epoch = 10000
        json_list = []
        for data_file_path in tqdm(self.data_file_list):
            with open(data_file_path.as_posix(), "r", encoding="utf-8") as f:
                json_datas = json.load(f)
                for json_data in json_datas:
                    json_list.append(self.format_json(json_data, data_file_path.stem))
                    if len(json_list) == 1000:
                        epoch += 1
                        self.save_result(epoch, json_list=json_list)
                        json_list = []
        if len(json_list) > 0:
            epoch += 1
            self.save_result(epoch, json_list=json_list)
            json_list = []

    @staticmethod
    def format_json(json_data: dict, file_name: str):
        title = json_data["title"]
        content = json_data["content"]
        unique_key = json_data["uniqueKey"]
        text = f"{title}\n\n{content}"
        return {
            "text": text,
            "meta": {"source": "wudao", "file": file_name, "uniqueKey": unique_key},
        }


class PClueDataProcess(DataPreprocess):
    def __init__(self, data_dir: Path, save_dir: Path):
        super().__init__(data_dir, save_dir)
        self.data_file_list = self.data_dir.glob("pCLUE_train.*")

    def process(self):
        epoch = 69133
        json_list = []
        for data_file_path in tqdm(self.data_file_list):
            with open(data_file_path.as_posix(), "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    json_data = json.loads(line)
                    json_list.append(self.format_json(json_data, data_file_path.stem))
                    if len(json_list) == 1000:
                        epoch += 1
                        self.save_result(epoch, json_list=json_list)
                        json_list = []
        if len(json_list) > 0:
            epoch += 1
            self.save_result(epoch, json_list=json_list)
            json_list = []

    @staticmethod
    def format_json(json_data: dict, file_name: str):
        input = json_data["input"]
        target = json_data["target"]
        text = f"{input} {target}"
        return {
            "text": text,
            "meta": {"source": "pClue", "file": file_name},
        }

class WMTDataProcess(DataPreprocess):
    def __init__(self, data_dir: Path, save_dir: Path):
        super().__init__(data_dir, save_dir)
        self.train_zh_path = data_dir.joinpath("train.zho")
        self.train_en_path = data_dir.joinpath("train.eng")
        self.translate_prompts = {"en-zh":[], "zh-en":[]}
        for k in self.translate_prompts:
            with open(os.path.join(data_dir.as_posix(), f'{k}.prompt'), 'r', encoding='utf-8') as f:
                self.translate_prompts[k] = [line.strip() for line in f]

    def process(self):
        epoch = 10000
        json_list = []

        train_zh_f = open(self.train_zh_path.as_posix(), "r", encoding="utf-8")
        train_en_f = open(self.train_en_path.as_posix(), "r", encoding="utf-8")
        for train_zh, train_en in zip(train_zh_f, train_en_f):
            zh_text = train_zh.rstrip()
            en_text = train_en.rstrip()
            patten = random.choice(["en-zh", "zh-en"])
            prompt = random.choice(self.translate_prompts[patten])
            if patten == "en-zh":
                json_data = {
                    "text": en_text + "\n" + prompt + "\n" + zh_text,
                    "meta": {"source": "wmt", "file": f"{patten}\n{prompt}"},
                }
            else:
                json_data = {
                    "text": zh_text + "\n" + prompt + "\n" + en_text,
                    "meta": {"source": "wmt", "file": f"{patten}\n{prompt}"},
                }
            json_list.append(self.format_json(json_data))
            if len(json_list) == 1000:
                epoch += 1
                self.save_result(epoch, json_list=json_list)
                json_list = []
        if len(json_list) > 0:
            epoch += 1
            self.save_result(epoch, json_list=json_list)
            json_list = []

    @staticmethod
    def format_json(json_data: dict):
        return json_data

class ThePileDataProcess(DataPreprocess):
    def __init__(self, data_dir: Path, save_dir: Path):
        super().__init__(data_dir, save_dir)
        self.data_file_list = self.data_dir.glob("*.jsonl")

    def process(self):
        epoch = self.get_max_epoch(self.save_dir)
        print(f"start epoch: {epoch}")
        json_list = []
        for data_file_path in self.data_file_list:
            with open(data_file_path.as_posix(), "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    json_data = json.loads(line)
                    meta = json_data["meta"]
                    pile_set_name = meta["pile_set_name"]
                    if pile_set_name not in ["Pile-CC", "Github"]:
                        continue
                    json_list.append(self.format_json(json_data, data_file_path.stem))
                    if len(json_list) == 1000:
                        epoch += 1
                        self.save_result(epoch, json_list=json_list)
                        json_list = []
        if len(json_list) > 0:
            epoch += 1
            self.save_result(epoch, json_list=json_list)
            json_list = []
        epoch = self.get_max_epoch(self.save_dir)
        print(f"end epoch: {epoch}")

    @staticmethod
    def format_json(json_data: dict, file_name: str):
        text = json_data["text"]
        pile_set_name = json_data["meta"]["pile_set_name"]
        return {
            "text": text,
            "meta": {"source": f"the_pile_{pile_set_name}", "file": file_name},
        }

    @staticmethod
    def get_max_epoch(save_path: Path):
        file_name_list = [int(file_name.stem) for file_name in save_path.glob("*.jsonl")]
        max_epoch = max(file_name_list) if len(file_name_list) > 0 else 0
        return max_epoch






if __name__ == "__main__":
    ThePileDataProcess(Path("/llm/data_resources/base/pile/train/pile_cc"),
                   Path("/llm/data_resources/base/data/en")).process()


