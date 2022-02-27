#!/bin/bash

export PYTHONPATH=../..
#export PRETRAINED_MODEL_PATH=/media/raid_store/pretrained_traffic/gpt2_model_4epochs_classes_external

export TRAIN_DATASET=../../datasets/train_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv
export TEST_DATASET=../../datasets/test_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNzVjNWYwZGUtYzQ0MC00ZWQ3LWEzNzItYjk0MDFkZmYzNWZjIn0="


#python train_classifier.py \
#--pretrained_path=$PRETRAINED_MODEL_PATH \
#--train_dataset=$TRAIN_DATASET \
#--test_dataset=$TEST_DATASET \
#--freeze_pretrained_model \
#--mask_first_token

#python train_classifier.py \
#--pretrained_path=$PRETRAINED_MODEL_PATH \
#--train_dataset=$TRAIN_DATASET \
#--test_dataset=$TEST_DATASET \
#--mask_first_token
#
#python train_classifier.py \
#--pretrained_path=$PRETRAINED_MODEL_PATH \
#--train_dataset=$TRAIN_DATASET \
#--test_dataset=$TEST_DATASET \
#--freeze_pretrained_model
#
#python train_classifier.py \
#--pretrained_path=$PRETRAINED_MODEL_PATH \
#--train_dataset=$TRAIN_DATASET \
#--test_dataset=$TEST_DATASET \
#
#python train.py \
#--pretrained_path=$PRETRAINED_MODEL_PATH \
#--train_dataset=$TRAIN_DATASET \
#--test_dataset=$TEST_DATASET \
#--freeze_pretrained_model \
#--reinitialize
#
#python train_classifier.py \
#--pretrained_path=$PRETRAINED_MODEL_PATH \
#--train_dataset=$TRAIN_DATASET \
#--test_dataset=$TEST_DATASET \
#--reinitialize

#python train_classifier.py \
#--pretrained_path=/media/raid_store/pretrained_traffic/gpt2_model_5epochs_no_classes_no_mawi_unswnb_iscxvpn_1layer \
#--train_dataset=$TRAIN_DATASET \
#--test_dataset=$TEST_DATASET \
#--freeze_pretrained_model \
#--mask_first_token \
#--log_neptune
#
#python train_classifier.py \
#--pretrained_path=/media/raid_store/pretrained_traffic/gpt2_model_5epochs_no_classes_no_mawi_unswnb_iscxvpn_2layer \
#--train_dataset=$TRAIN_DATASET \
#--test_dataset=$TEST_DATASET \
#--freeze_pretrained_model \
#--mask_first_token \
#--log_neptune

python train_classifier.py \
--pretrained_path=/media/raid_store/pretrained_traffic/gpt2_model_5epochs_no_classes_no_mawi_unswnb_iscxvpn_3layer \
--train_dataset=$TRAIN_DATASET \
--test_dataset=$TEST_DATASET \
--freeze_pretrained_model \
--mask_first_token \
--log_neptune

python train_classifier.py \
--pretrained_path=/media/raid_store/pretrained_traffic/gpt2_model_5epochs_no_classes_no_mawi_unswnb_iscxvpn_4layer \
--train_dataset=$TRAIN_DATASET \
--test_dataset=$TEST_DATASET \
--freeze_pretrained_model \
--mask_first_token \
--log_neptune

python train_classifier.py \
--pretrained_path=/media/raid_store/pretrained_traffic/gpt2_model_5epochs_no_classes_no_mawi_unswnb_iscxvpn_5layer \
--train_dataset=$TRAIN_DATASET \
--test_dataset=$TEST_DATASET \
--freeze_pretrained_model \
--mask_first_token \
--log_neptune
