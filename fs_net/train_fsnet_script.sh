export PYTHONPATH=..
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNzVjNWYwZGUtYzQ0MC00ZWQ3LWEzNzItYjk0MDFkZmYzNWZjIn0="

TRAIN_DS=../datasets/train_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv
TEST_DS=../datasets/test_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv

python train_fsnet.py \
--train_dataset=$TRAIN_DS \
--test_dataset=$TEST_DS \
--tokenizer_path=/media/raid_store/pretrained_traffic/gpt2_model_2epochs_no_classes_no_mawi \
--use_packet_size_only \
--dynamic_ps_range=10000 \
--log_neptune