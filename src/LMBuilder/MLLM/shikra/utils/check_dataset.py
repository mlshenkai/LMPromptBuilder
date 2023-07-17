# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/7/8 9:47 PM
# @File: check_dataset
# @Email: mlshenkai@163.com

from torch.utils.data import DataLoader

def check_data(dataset, data_collator):
    dataloader = DataLoader(dataset=dataset, collate_fn=data_collator, batch_size=8,shuffle=False)
    for batch in dataloader:
        print(batch)