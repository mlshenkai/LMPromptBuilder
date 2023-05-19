# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/19 2:42 PM
# @File: torch_utils
# @Email: mlshenkai@163.com
import torch
import torch.backends.mps
from loguru import logger


def torch_gc():
    if torch.cuda.is_available():
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache

            empty_cache()
        except Exception as e:
            logger.info(e)
            logger.info(
                "如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。"
            )
