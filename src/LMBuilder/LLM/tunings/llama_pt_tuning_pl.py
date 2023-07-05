# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/14 2:09 PM
# @File: llama_pt_tuning_pl
# @Email: mlshenkai@163.com
import os

from lightning.pytorch.loggers import WandbLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
import pyrootutils
import lightning.pytorch as pl
from lightning.pytorch.strategies import DeepSpeedStrategy


project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichModelSummary,
)
from src.LMBuilder.LLM.models.llama.model_lora_pl import LlamaModelLoraPL
from src.LMBuilder.LLM.models.llama.model_args import LlamaModelArgs, LoraModelArgs
from src.LMBuilder.LLM.data_build import LlamaPTTrainDataArgs
from src.LMBuilder.LLM.data_build import LlamaPTDatasetPL


def main():
    model_args = LlamaModelArgs()
    lora_args = LoraModelArgs()
    data_args = LlamaPTTrainDataArgs()
    model_pl = LlamaModelLoraPL(
        model_args=model_args, use_peft=True, peft_args=lora_args
    )
    tokenizer = model_pl.tokenizer
    data_pl = LlamaPTDatasetPL(data_args, tokenizer=tokenizer)
    data_pl.setup("fit")
    train_dataloader = data_pl.train_dataloader()
    # strategy = DeepSpeedStrategy(config=f"{project_path}/configs/ds_config_zero2.json")
    model_checkpoint_callback = ModelCheckpoint(
        filename="epoch={epoch:02d}-train_loss={train/loss:.4f}",
        monitor="train/loss",
        verbose=True,
        save_last=True,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="train/loss",
        min_delta=0.01,
        patience=5,
        verbose=True,
        mode="min",
        strict=True,
    )
    # learn_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    rich_model_summary = RichModelSummary(max_depth=1)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=5,
        max_epochs=1000,
        strategy=DeepSpeedStrategy(
            stage=3, offload_optimizer=False, offload_parameters=True, overlap_comm=True
        ),
        precision="bf16-mixed",
        callbacks=[
            model_checkpoint_callback,
            early_stopping_callback,
            # learn_rate_monitor,
            # rich_model_summary,
        ],
        logger=WandbLogger(save_dir="logs", project="llama_13b_no_loss_mask_pt1_test", log_model=False),
        log_every_n_steps=50,
    )
    trainer.fit(
        model_pl, train_dataloaders=train_dataloader
    )


if __name__ == "__main__":
    # strategy = DeepSpeedStrategy(config=f"{project_path}/configs/ds_config_zero2.json")
    main()
