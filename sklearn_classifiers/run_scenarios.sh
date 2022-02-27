
export PYTHONPATH=..
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNzVjNWYwZGUtYzQ0MC00ZWQ3LWEzNzItYjk0MDFkZmYzNWZjIn0="

TRAIN_DS=../datasets/train_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv
TEST_DS=../datasets/test_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv


#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--raw \
#--use_iat \
#--search_hyper_parameters \
#--log_neptune
#
#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--search_hyper_parameters \
#--raw \
#--log_neptune

#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--continuous \
#--use_iat \
#--search_hyper_parameters \
#--log_neptune
#
#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--continuous \
#--search_hyper_parameters \
#--log_neptune
#
#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--continuous \
#--use_iat \
#--search_hyper_parameters \
#--raw \
#--log_neptune
#
#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--continuous \
#--search_hyper_parameters \
#--raw \
#--log_neptune

python run_training.py \
--train_dataset=$TRAIN_DS \
--test_dataset=$TEST_DS \
--transformer_model_path=/media/raid_store/pretrained_traffic/gpt2_model_2epochs_no_classes_no_mawi \
--log_neptune

python run_training.py \
--train_dataset=$TRAIN_DS \
--test_dataset=$TEST_DS \
--transformer_model_path=/media/raid_store/pretrained_traffic/gpt2_model_3epochs_no_classes_no_mawi_iot_home \
--log_neptune

python run_training.py \
--train_dataset=$TRAIN_DS \
--test_dataset=$TEST_DS \
--transformer_model_path=/media/raid_store/pretrained_traffic/gpt2_model_5epochs_no_classes_no_mawi_unswnb_iscxvpn \
--log_neptune

#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--transformer_model_path=/media/raid_store/pretrained_traffic/gpt2_model_2epochs_classes \
#--log_neptune
#
#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--transformer_model_path=/media/raid_store/pretrained_traffic/gpt2_model_2epochs_classes \
#--mask_first_token \
#--log_neptune
#
#
#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--transformer_model_path=/media/raid_store/pretrained_traffic/gpt2_model_4epochs_classes_external \
#--log_neptune
#
#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--transformer_model_path=/media/raid_store/pretrained_traffic/gpt2_model_4epochs_classes_external \
#--mask_first_token \
#--log_neptune
#
#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--transformer_model_path=/media/raid_store/pretrained_traffic/gpt2_model_4_6epochs_classes_home_iot \
#--log_neptune
#
#python run_training.py \
#--train_dataset=$TRAIN_DS \
#--test_dataset=$TEST_DS \
#--transformer_model_path=/media/raid_store/pretrained_traffic/gpt2_model_4_6epochs_classes_home_iot \
#--mask_first_token \
#--log_neptune
