#!/bin/bash

PYTHONPATH=.. python run_packet_modeling.py \
--model_name_or_path=/media/raid_store/pretrained_traffic/gpt2_model_2epochs \
--finetune_on_class=Telegram \
--output_dir=/media/raid_store/pretrained_traffic/gpt2_model_telegram \
--do_train \
--train_data_file=/home/radion/Apps/PyCharmServer/classifier/datasets/train_78c109eedb12f4e9a7f91fec6a7621f2.csv \
--do_eval \
--eval_data_file=/home/radion/Apps/PyCharmServer/classifier/datasets/test_78c109eedb12f4e9a7f91fec6a7621f2.csv \
--overwrite_output_dir \
--per_device_train_batch_size=128 \
--per_device_eval_batch_size=224 \
--fp16 \
--fp16_opt_level=O2 \
--logging_steps=1 \
--save_steps=1000 \
--eval_steps=1000 \
--gradient_accumulation_steps=30 \
--num_train_epochs=200 \
--learning_rate=0.0001 \
--save_total_limit=10 \
--logging_dir=fine_runs/5