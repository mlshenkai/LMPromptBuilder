# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/7/10 9:36 AM
# @File: shikra_deepspeed
# @Email: mlshenkai@163.com
training_args = dict(
    # run
    output_dir=None,  # required. must be filled by derived configs.
    overwrite_output_dir=True,
    report_to='wandb',
    seed=42,

    # datasets
    remove_unused_columns=False,

    # train
    do_train=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    weight_decay=0.,
    warmup_ratio=0.03,
    evaluation_strategy='no',


    # train ddp
    # tf32=False,
    # bf16=True,
    # gradient_checkpointing=True,
    # fsdp="full_shard auto_wrap",
    # fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer',
    # bf16=True,
    fp16=True,
    deepspeed={
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "bf16": {
            "enabled": False
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "last_batch_iteration": -1,
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 1e9,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 1e9,
            "contiguous_gradients": True
        },


        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
        "master_port": 29501
    },

    # train logging
    logging_steps=10,
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=3,
    # report_to="wandb",

    # eval and predict
    do_eval=False,
    do_predict=False,
    predict_with_generate=True,
    per_device_eval_batch_size=2,
    dataloader_num_workers=4,
)
