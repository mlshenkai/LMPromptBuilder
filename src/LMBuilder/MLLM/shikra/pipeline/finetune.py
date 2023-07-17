# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/7/5 3:30 PM
# @File: finetune
# @Email: mlshenkai@163.com
from pyrootutils import pyrootutils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,7"
import sys
import wandb
import torch
# import torch.multiprocessing as mp
import pathlib
# from loguru import logger
import typing

from src.LMBuilder.MLLM.shikra.utils.check_dataset import check_data
from src.test.clip_model_test import test_clip_version

project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
from loguru import logger
from src.LMBuilder.MLLM.shikra.configs.config import prepare_args
from src.LMBuilder.MLLM.shikra.builder import load_pretrained
from src.LMBuilder.MLLM.shikra.utils import print_trainable_params
from src.LMBuilder.MLLM.shikra.data_build import prepare_data, prepare_target_processor
from src.LMBuilder.MLLM.shikra.engine import prepare_trainer_collator


def main(arg=None):
    cfg, training_args = prepare_args()
    training_args.overwrite_output_dir = True
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    # Some ugly codes to inject target_processor into preprocessor.
    # maybe effect model. (e.g. add special token; resize embedding)
    model, preprocessor = prepare_target_processor(model, preprocessor, cfg.model_args, training_args)
    print_trainable_params(model)

    # Prepare data_collator
    collator_kwargs = cfg.data_args.collator_kwargs
    trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)
    # test_clip_version(preprocessor["image"], model.model.vision_tower[0])


    # check_data(dataset["train"], data_collator_dict['train_collator'])
    # Initialize Trainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=preprocessor['text'],
        train_dataset=dataset['train'] if training_args.do_train else None,
        eval_dataset=dataset['validation'] if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        **data_collator_dict,
    )

    # Training
    if training_args.do_train:
        try:
            if (not training_args.overwrite_output_dir) and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
                train_result = trainer.train(resume_from_checkpoint=True)
            else:
                train_result = trainer.train()
            trainer.log_metrics("train", train_result.metrics)  # noqa
            trainer.save_metrics("train", train_result.metrics)  # noqa
            trainer.save_model()
        except RuntimeError as e:
            print(f"got RuntimeError: {e.args}")
            try:
                print(f"#### device {training_args.local_rank} summary ####\n{torch.cuda.memory_summary(training_args.local_rank)}")
            except Exception as inner_e:
                print(f"get Exception when show cuda summary: {inner_e.args}")
            raise e
        finally:
            trainer.save_state()  # noqa
            trainer.plot_loss()

    # save cfg to output_dir
    try:
        output_dir = training_args.output_dir
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg.dump(os.path.join(output_dir, "cfg.py"))
    except Exception as e:
        logger.warning(f'try to save cfg to output_dir, but get exception {e.args}')

    # Keyword arguments for `model.generate`
    gen_kwargs = dict(cfg.data_args.gen_kwargs)
    gen_kwargs.setdefault('use_cache', True)
    # important for use model.generate in batch mode. some model config with wrong special_token_id
    # (e.g. shikra generationConfig set pad_token_id to -1)
    if hasattr(cfg.model_args, 'gen_kwargs_set_pad_token_id') and cfg.model_args.gen_kwargs_set_pad_token_id:
        gen_kwargs['pad_token_id'] = preprocessor['text'].pad_token_id
    if hasattr(cfg.model_args, 'gen_kwargs_set_bos_token_id') and cfg.model_args.gen_kwargs_set_bos_token_id:
        gen_kwargs['bos_token_id'] = preprocessor['text'].bos_token_id
    if hasattr(cfg.model_args, 'gen_kwargs_set_eos_token_id') and cfg.model_args.gen_kwargs_set_eos_token_id:
        gen_kwargs['eos_token_id'] = preprocessor['text'].eos_token_id

    # Evaluation
    if training_args.do_eval:
        if hasattr(trainer, '_test_collator') and hasattr(trainer, '_eval_collator') \
                and trainer._test_collator != trainer._eval_collator:  # noqa
            logger.warning('[WARNING!!!] use different collator for eval and test. but do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.)')
        eval_results = trainer.predict(dataset['validation'], metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_prediction(eval_results, file_key_prefix='eval')

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset['test'], metric_key_prefix="test", **gen_kwargs)
        trainer.log_metrics("test", predict_results.metrics)  # noqa
        trainer.save_metrics("test", predict_results.metrics)  # noqa
        trainer.save_prediction(predict_results, file_key_prefix='test')

    # Multi Predict
    if training_args.do_multi_predict:
        old_compute_metrics = trainer.compute_metrics
        multitest = dataset['multitest']
        multitest = typing.cast(dict, multitest)
        for _idx, (k, item) in enumerate(multitest.items()):
            print(f'processing multitest set {_idx}/{len(multitest)}: {k}')
            _ds = item['dataset']
            _compute_metrics = item['compute_metric']
            _prefix = f"multitest_{k}"

            trainer.compute_metrics = _compute_metrics
            _pred_results = trainer.predict(_ds, metric_key_prefix=_prefix, **gen_kwargs)
            trainer.log_metrics(_prefix, _pred_results.metrics)  # noqa
            trainer.save_metrics(_prefix, _pred_results.metrics)  # noqa
            trainer.save_prediction(_pred_results, file_key_prefix=_prefix)
        trainer.compute_metrics = old_compute_metrics


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    wandb.init(project="llava", group="llava")
    # import torch.multiprocessing as mp
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # mp.spawn(main, args=(), nprocs=3)
    main()
    wandb.finish()


