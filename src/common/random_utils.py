# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/16 11:39 AM
# @File: random_utils
# @Email: mlshenkai@163.com
import random
import torch
import numpy as np


def manual_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)
