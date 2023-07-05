# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/6/14 9:35 AM
# @File: model_args
# @Email: mlshenkai@163.com
from dataclasses import dataclass, field


@dataclass
class LlamaModelArgs:
    model_name_or_path = "/llm/base_model_resources/chinese_llama/merge_chinese_llama_lora_7b"
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
    warmup_epochs: int = 5
    warmup_steps = 100
    report_to = "wandb"
    optimizer: str = "adamw_torch"
    save_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 1000
    max_epochs: int = 100
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
    num_virtual_tokens: int = 20
    prompt_encoder_hidden_size: int = 128
    num_train_epochs = 300
    max_steps = 6000000
    per_device_train_batch_size = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps = 1
    save_total_limit = 3
    remove_unused_columns = False
    logging_steps = 50
    resume_from_checkpoint: str = None
    enable_torch_compile: bool = False
    weight_decay: float = 0.01


class LoraModelArgs:
    lora_r: int = 8
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ]
    lora_bias = "none"
    lora_beta: float = 0.85
    module_to_save = ["embed_tokens", "lm_head"]
