# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/7/4 2:09 PM
# @File: deepspeed_convert_fp32
# @Email: mlshenkai@163.com

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict


if __name__ == "__main__":
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir="/code-online/shenkai/LMPromptBuilder/exp/shikra_pretrain_final19_stage2/checkpoint-2000",
        output_file="/code-online/shenkai/LMPromptBuilder/exp/shikra_pretrain_final19_stage2/checkpoint-200_bin/pytorch_model.bin",
        tag="global_step2000",
    )
