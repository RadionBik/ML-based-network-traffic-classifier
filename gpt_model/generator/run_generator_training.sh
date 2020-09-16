#!/bin/bash

PYTHONPATH=../.. python train_generator.py \
--config_name=model_config.json \
--quantizer_path=trained_quantizers/quantizer_2^14_train_shuffled_0 \
--output_dir=/media/raid_store/pretrained_traffic/gpt2_model_4epochs_classes_external \
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
--save_steps=150 \
--eval_steps=1000 \
--gradient_accumulation_steps=30 \
--num_train_epochs=4 \
--warmup_steps=200 \
--learning_rate=0.001 \
--save_total_limit=10 \
--file_patterns_to_exclude=mawi_iot_home \
--train_with_targets
#--model_name_or_path=/media/raid_store/pretrained_traffic/gpt2_model_4epochs_classes_home_iot \