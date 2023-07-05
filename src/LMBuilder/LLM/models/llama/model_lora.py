# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/28 8:01 PM
# @File: model_lora
# @Email: mlshenkai@163.com

import os
from typing import Any, Optional

from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch.optim import AdamW, lr_scheduler
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer
import lightning.pytorch as pl
import torch
import torchmetrics
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from loguru import logger


class LlamaModelLoraPL(pl.LightningModule):
    def __init__(self, model_args, use_peft=True, peft_args=None):
        super().__init__()
        # self.save_hyperparameters()
        self.model_args = model_args
        self.use_peft = use_peft
        self.peft_args = peft_args
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.load_backbone_model()
        self.load_peft_model()
        acc_metrics = torchmetrics.Accuracy(num_classes=self.model_config.vocab_size, task="multiclass")
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
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        self.train_acc(preds, labels)
        self.log("train/loss", train_loss, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)
        return {"loss": train_loss, "preds": preds, "targets": labels}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outputs = self.forward(batch)
        valid_loss = outputs.loss
        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        self.valid_acc(preds, labels)
        self.log("valid/loss", valid_loss, on_step=False, on_epoch=True)
        self.log("valid/acc", self.valid_acc, on_step=False, on_epoch=True)
        return {"loss": valid_loss, "preds": preds, "targets": labels}

    def on_validation_epoch_end(self) -> None:
        acc = self.train_acc.compute()
        self.valid_acc_best(acc)
        self.log("valid/ac_best", self.valid_acc_best.compute())

    def configure_optimizers(self) -> Any:
        weight_decay = self.model_args.weight_decay
        # optimizer = FusedAdam(
        #     self.model.parameters(),
        #     self.model_args.learning_rate,
        #     weight_decay=self.model_args.weight_decay
        # )
        # optimizer = DeepSpeedCPUAdam(self.model.parameters(),
        #                              self.model_args.learning_rate,
        #                              weight_decay=self.model_args.weight_decay,)
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

    def load_peft_model(self):
        if self.use_peft and self.peft_args is not None:
            target_modules = self.peft_args.lora_target_modules
            lora_r = self.peft_args.lora_r
            lora_alpha = self.peft_args.lora_alpha
            lora_dropout = self.peft_args.lora_dropout
            model_to_save = self.peft_args.module_to_save
            peft_config = LoraConfig(
                peft_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                modules_to_save=model_to_save,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            old_state_dict = self.model.state_dict
            self.model.state_dict = (
                lambda x, *_, **__: get_peft_model_state_dict(x, old_state_dict())
            ).__get__(self.model, type(self.model))
            # print(self.model)

    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = self.model_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.model_args.save(output_dir)



    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        torch.cuda.empty_cache()
