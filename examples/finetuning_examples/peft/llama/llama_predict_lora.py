# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/17 10:09 AM
# @File: llama_predict_lora
# @Email: mlshenkai@163.com
import torch

device_indices = [1, 2, 3, 4]
for index in device_indices:
    torch.cuda.set_device(index)
from loguru import logger
import pyrootutils
import pandas as pd

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
from src.models import LlamaModelPeft


def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.replace("\\n", "\n")
            line = line.strip("\n")
            terms = line.split("￥")
            instruction = "从下面的品名文本中提取英文品名:"
            if len(terms) < 2:
                continue

            origin_goods_name = terms[0]
            eng_goods_name = terms[-1]
            if eng_goods_name and origin_goods_name:

                data.append([instruction, terms[0], terms[-1]])
            # if len(terms) == 2:
            #     data.append([instruction, terms[0], terms[1]])
            # else:
            #     logger.warning(f'line error: {line}')
    return data


def get_prompt(arr):
    if arr["input"].strip():
        return f"""Below is an instruction that describes a task. Write a response that appropriately 
                completes the request.\n\n### Instruction:\n{arr['instruction']}\n### 
                Input:\n{arr['input']}\n\n### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that 
                appropriately completes the request.\n\n### Instruction:\n{arr['instruction']}\n\n### Response:"""


model = LlamaModelPeft(
    "llama",
    "shibing624/chinese-alpaca-plus-7b-hf",
    peft_name=f"{project_path}/examples/peft/llama/outputs_p_tuning",
)
test_file = f"{project_path}/resources/data/中英品名0512.txt"
test_data = load_data(test_file)[1:10]
test_df = pd.DataFrame(test_data, columns=["instruction", "input", "output"])
logger.debug("test_df: {}".format(test_df))

test_df["prompt"] = test_df.apply(get_prompt, axis=1)
# model.model.stream_chat()
# test_df["predict_after"] = model.predict(test_df["prompt"].tolist())
# logger.debug("test_df result: {}".format(test_df[["output", "predict_after"]]))
# out_df = test_df[["instruction", "input", "output", "predict_after"]]
# out_df.to_json(
#     f"{project_path}/resources/data/test_result.json",
#     force_ascii=False,
#     orient="records",
#     lines=True,
# )
