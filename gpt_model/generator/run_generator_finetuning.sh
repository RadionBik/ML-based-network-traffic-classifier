#!/bin/bash

PYTHONPATH=../.. python train_generator.py \
--model_name_or_path=/media/raid_store/pretrained_traffic/gpt2_model_2epochs_classes \
--finetune_on_class=Telegram \
--output_dir=/media/raid_store/pretrained_traffic/gpt2_model_telegram \
--do_train \
--train_data_file=../../datasets/train_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv \
--do_eval \
--eval_data_file=../../datasets/test_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv \
--overwrite_output_dir \
--per_device_train_batch_size=128 \
--per_device_eval_batch_size=224 \
--fp16 \
--fp16_opt_level=O2 \
--logging_steps=1 \
--save_steps=1000 \
--eval_steps=1000 \
--gradient_accumulation_steps=30 \
--num_train_epochs=10 \
--learning_rate=0.00005 \
--save_total_limit=10 \
--logging_dir=fine_runs/7