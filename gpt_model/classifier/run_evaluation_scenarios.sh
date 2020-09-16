#!/bin/bash

export PYTHONPATH=../..
export PRETRAINED_MODEL_PATH=/media/raid_store/pretrained_traffic/gpt2_model_4epochs_classes_external

export TRAIN_DATASET=../../datasets/train_78c109eedb12f4e9a7f91fec6a7621f2.csv
export TEST_DATASET=../../datasets/test_78c109eedb12f4e9a7f91fec6a7621f2.csv


python train_classifier.py \
--pretrained_path=$PRETRAINED_MODEL_PATH \
--train_dataset=$TRAIN_DATASET \
--test_dataset=$TEST_DATASET \
--freeze_pretrained_model \
--mask_first_token

python train_classifier.py \
--pretrained_path=$PRETRAINED_MODEL_PATH \
--train_dataset=$TRAIN_DATASET \
--test_dataset=$TEST_DATASET \
--mask_first_token

python train_classifier.py \
--pretrained_path=$PRETRAINED_MODEL_PATH \
--train_dataset=$TRAIN_DATASET \
--test_dataset=$TEST_DATASET \
--freeze_pretrained_model

python train_classifier.py \
--pretrained_path=$PRETRAINED_MODEL_PATH \
--train_dataset=$TRAIN_DATASET \
--test_dataset=$TEST_DATASET \

python train.py \
--pretrained_path=$PRETRAINED_MODEL_PATH \
--train_dataset=$TRAIN_DATASET \
--test_dataset=$TEST_DATASET \
--freeze_pretrained_model \
--reinitialize

python train_classifier.py \
--pretrained_path=$PRETRAINED_MODEL_PATH \
--train_dataset=$TRAIN_DATASET \
--test_dataset=$TEST_DATASET \
--reinitialize
