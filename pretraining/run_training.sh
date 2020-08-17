#!/bin/bash

PYTHONPATH=.. python run_packet_modeling.py \
--config_name=model_config.json \
--output_dir=/media/raid_store/pretrained_traffic/gpt2_model_2epochs_classes \
--do_train \
--train_data_file=/media/raid_store/pretrained_traffic/train_csv \
--do_eval \
--eval_data_file=/media/raid_store/pretrained_traffic/val_csv \
--overwrite_output_dir \
--per_device_train_batch_size=128 \
--per_device_eval_batch_size=224 \
--fp16 \
--fp16_opt_level=O2 \
--logging_steps=1 \
--save_steps=350 \
--eval_steps=1000 \
--gradient_accumulation_steps=30 \
--num_train_epochs=2 \
--warmup_steps=500 \
--learning_rate=0.001 \
--save_total_limit=10 \
--train_with_targets