# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/16 11:19 AM
# @File: model_args
# @Email: mlshenkai@163.com
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import Optional, Any
from src.models.peft.config.base_model_args import ModelArgs
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig


MODEL_CLASSES = {"llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer)}


@dataclass
class LlamaArgs(ModelArgs):
    """
    Model args for a LlamaModel
    """

    model_class: str = "LlamaArgs"
    dataset_class: Dataset = None
    learning_rate: float = 5e-6
    fp16: bool = True
    int8: bool = False
    quantization_bit: int = None  # if use quantization bit, set 4, else None
    debug: bool = False
    max_seq_length: int = 256  # max length of input sequence
    max_length = 384  # max length of the sequence to be generated
    do_sample: bool = True
    early_stopping: bool = True
    evaluate_generated_text: bool = True
    is_train_on_prompt: bool = True
    warmup_steps: int = 50
    report_to = "tensorboard"
    optimizer: str = "adamw_torch"
    save_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 400
    pad_to_multiple_of: int = 8
    max_eval_samples: int = 20
    length_penalty: float = 2.0
    num_beams: int = 1
    num_return_sequences: int = 1
    repetition_penalty: float = 1.3
    temperature: float = 0.2
    special_tokens_list: list = field(default_factory=list)
    top_k: float = 40
    top_p: float = 0.9
    model_name_or_path: Optional[str] = field(default="decapoda-research/llama-7b-hf")
    use_peft: bool = True
    peft_type: str = "LORA"
    peft_bin_name: str = "adapter_model.bin"
    lora_r: int = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_bias = "none"
    adalora_init_r: int = 12
    adalora_tinit: int = 200
    adalora_tfinal: int = 1000
    adalora_delta_t: int = 10
    lora_beta: float = 0.85
    num_virtual_tokens: int = 20
    prompt_encoder_hidden_size: int = 128
    num_train_epochs = 3
    max_steps = -1
    per_device_train_batch_size = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps = 1
    save_total_limit = 3
    remove_unused_columns = False
    logging_steps = 50
    resume_from_checkpoint: str = None
    enable_torch_compile: bool = False
    # enable_torch_compile: bool = False
