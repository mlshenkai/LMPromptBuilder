# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/13 5:06 PM
# @File: model_pl
# @Email: mlshenkai@163.com
import os
from typing import Any, Optional

from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch.optim import AdamW, lr_scheduler
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer
import lightning.pytorch as pl
import torchmetrics
from loguru import logger


class LlamaModelPL(pl.LightningModule):
    def __init__(self, model_args):
        super().__init__()
        # self.save_hyperparameters()
        self.model_args = model_args
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.load_backbone_model()
        acc_metrics = torchmetrics.Accuracy(
            "multiclass", num_classes=len(self.model_config.vocab_size)
        )
        self.train_acc = acc_metrics.clone()
        self.valid_acc = acc_metrics.clone()
        self.valid_acc_best = torchmetrics.MaxMetric()

    def unfreeze(self) -> None:
        logger.info("un-freeze model param")
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze(self) -> None:
        logger.info("freeze model para")
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, batch) -> Any:
        outputs = self.model(**batch)
        return outputs

    def on_train_start(self) -> None:
        self.train_acc.reset()
        self.valid_acc.reset()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outputs = self.forward(batch)
        train_loss = outputs.loss
        logits = outputs.logits
        labels = batch["labels"]
        self.train_acc(logits, labels)
        self.log("train/loss", train_loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return {"loss": train_loss, "preds": logits, "targets": labels}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outputs = self.forward(batch)
        valid_loss = outputs.loss
        logits = outputs.logits
        labels = batch["labels"]
        self.valid_acc(logits, labels)
        self.log("valid/loss", valid_loss, on_step=False, on_epoch=True)
        self.log("valid/acc", self.valid_acc, on_step=False, on_epoch=True)
        return {"loss": valid_loss, "preds": logits, "targets": labels}

    def on_validation_epoch_end(self) -> None:
        acc = self.train_acc.compute()
        self.valid_acc_best(acc)
        self.log("valid/ac_best", self.valid_acc_best.compute())

    def configure_optimizers(self) -> Any:
        weight_decay = self.model_args.weight_decay
        if weight_decay:
            optimizer = AdamW(
                self.model.parameters(),
                self.model_args.learning_rate,
                weight_decay=self.model_args.weight_decay,
            )
        else:
            optimizer = AdamW(
                self.model.parameters(),
                self.model_args.learning_rate,
            )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.model_args.warmup_epochs,
            max_epochs=self.model_args.max_epochs,
        )
        return [optimizer], [scheduler]

    def load_backbone_model(self):
        model_name_or_path = self.model_args.model_name_or_path
        self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, config=self.model_config
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path
        )

    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = self.model_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.model_args.save(output_dir)
