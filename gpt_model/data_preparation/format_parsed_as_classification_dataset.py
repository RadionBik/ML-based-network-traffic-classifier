import logging
import pathlib
from typing import Iterable, Optional

import pandas as pd

from flow_parsing.utils import get_hash, read_dataset, check_filename_in_patterns, save_dataset
from gpt_model.data_preparation.preprocess_target_pcaps import IOT_DEVICES
from settings import TARGET_CLASS_COLUMN, LOWER_BOUND_CLASS_OCCURRENCE, FilePatterns, DATASET_DIR

"""
task-specific module, provided for the sake of reproducibility
formats labels from outputs of nDPI and in case of IoT traffic, assigns labels from filenames
"""

logger = logging.getLogger(__name__)

# signalling protos are common among all devices and it doesn't make sense to treat them separately
COMMON_PROTOCOLS = ['DNS', 'NTP', 'STUN']
GARBAGE_PROTOCOLS = ['ICMP', 'ICMPV6', 'DHCPV6', 'DHCP', 'Unknown', 'IGMP', 'SSDP']


def _load_parsed_results(dir_with_parsed_csvs, filename_patterns_to_exclude: Optional[Iterable[str]]):
    dir_with_parsed_csvs = pathlib.Path(dir_with_parsed_csvs)

    parsed_csvs = list(dir_with_parsed_csvs.glob('*.csv'))

    iot_datasets = []
    usual_traffic = []

    iot_categories = set(item.category for item in IOT_DEVICES)
    for csv_file in parsed_csvs:
        # skip non-home and IoT files
        if check_filename_in_patterns(csv_file, filename_patterns_to_exclude):
            continue

        traffic_df = read_dataset(csv_file)

        if csv_file.name.startswith('train'):
            base_name = csv_file.name.split('train_')[-1]
        elif csv_file.name.startswith('val'):
            base_name = csv_file.name.split('val_')[-1]
        elif csv_file.name.startswith('test'):
            base_name = csv_file.name.split('test_')[-1]
        else:
            base_name = csv_file.name

        traffic_df['source_file'] = base_name

        if base_name.split('_')[0] in iot_categories:
            iot_datasets.append(traffic_df)
        else:
            usual_traffic.append(traffic_df)

    try:
        iot_traffic = pd.concat(iot_datasets, ignore_index=True)
    except ValueError:
        iot_traffic = pd.DataFrame([])
        logger.warning('no IoT files were found!')
    usual_traffic = pd.concat(usual_traffic, ignore_index=True)
    logger.info(f'found: {len(iot_traffic)} IoT flows, and {len(usual_traffic)} usual')
    return iot_traffic, usual_traffic


def _set_common_protos_targets(dataset):
    for proto in COMMON_PROTOCOLS:
        dataset.loc[dataset['ndpi_app'].str.startswith(proto), TARGET_CLASS_COLUMN] = proto
    return dataset


def _set_iot_devices_targets(dataset):
    """ assigns target class according to the category of an IoT device """
    common_indexer = dataset[TARGET_CLASS_COLUMN].isin(COMMON_PROTOCOLS)
    iot_category = dataset.loc[~common_indexer, 'source_file'].str.split('_').apply(lambda x: 'IoT_' + x[0])
    dataset.loc[~common_indexer, TARGET_CLASS_COLUMN] = iot_category
    logger.info(str(dataset[TARGET_CLASS_COLUMN].value_counts()))
    return dataset


def _set_application_targets(dataset):
    """ assigns target class according to the 'Y' application from nDPI's 'X.Y' label """
    common_indexer = dataset[TARGET_CLASS_COLUMN].isin(COMMON_PROTOCOLS)
    cleaned_up_applications = dataset.loc[~common_indexer, 'ndpi_app'].str.split('.').apply(lambda x: x[-1])
    dataset.loc[~common_indexer, TARGET_CLASS_COLUMN] = cleaned_up_applications
    logger.info(str(dataset[TARGET_CLASS_COLUMN].value_counts()))
    return dataset


def _rm_garbage(dataset, garbage: list = None, column_from='ndpi_app'):
    """ rm irrelevant targets for classification at an upstream device """
    if garbage is None:
        garbage = GARBAGE_PROTOCOLS
    garbage_indexer = dataset[column_from].isin(garbage)
    logger.info(f'found {garbage_indexer.sum()} objects of garbage protos')
    return dataset[~garbage_indexer]


def prune_targets(dataset, lower_bound=LOWER_BOUND_CLASS_OCCURRENCE, underrepresented_protos: list = None):
    """ rm infrequent targets """
    proto_counts = dataset[TARGET_CLASS_COLUMN].value_counts()
    if underrepresented_protos is None:
        underrepresented_protos = proto_counts[proto_counts < lower_bound].index.tolist()
    if underrepresented_protos:
        logger.info(f'pruning the following targets: {underrepresented_protos}')
        dataset = dataset.loc[~dataset[TARGET_CLASS_COLUMN].isin(underrepresented_protos)]
    return dataset.reset_index(drop=True), underrepresented_protos


def delete_duplicating_flows(dataset):
    def to_session_id(flow_id):
        proto, conn1, conn2 = flow_id.split(' ')
        return proto, frozenset([conn1, conn2])

    dataset['session_id'] = dataset['flow_id'].apply(to_session_id)
    dataset = dataset.drop_duplicates(subset=['session_id'])
    dataset.drop(columns='session_id', inplace=True)
    logger.info(f'{dataset.shape[0]} flows left after deduplication')
    return dataset


def prepare_classification_data(csv_dir, remove_garbage=True, filename_patterns_to_exclude=None):
    """ the order of operations matters """
    iot_traffic, usual_traffic = _load_parsed_results(csv_dir, filename_patterns_to_exclude)

    if len(iot_traffic) > 0:
        iot_traffic = _set_common_protos_targets(iot_traffic)
        iot_traffic = _set_iot_devices_targets(iot_traffic)
        if remove_garbage:
            iot_traffic = _rm_garbage(iot_traffic,
                                      column_from='ndpi_app')

    usual_traffic = _set_common_protos_targets(usual_traffic)
    usual_traffic = _set_application_targets(usual_traffic)

    if remove_garbage:
        usual_traffic = _rm_garbage(usual_traffic,
                                    garbage=GARBAGE_PROTOCOLS + ['Amazon'],
                                    column_from=TARGET_CLASS_COLUMN)

    merged_traffic = pd.concat([usual_traffic, iot_traffic], ignore_index=True)
    return merged_traffic


def main():
    pattern_name = 'mawi_unswnb_iscxvpn'
    excluded_patterns = getattr(FilePatterns, pattern_name)
    train_df = prepare_classification_data(DATASET_DIR / 'pretraining/train_csv',
                                           filename_patterns_to_exclude=excluded_patterns)
    eval_df = prepare_classification_data(DATASET_DIR / 'pretraining/val_csv',
                                          filename_patterns_to_exclude=excluded_patterns)
    test_df = prepare_classification_data(DATASET_DIR / 'pretraining/test_csv',
                                          filename_patterns_to_exclude=excluded_patterns)
    tr_val_df = pd.concat([train_df, eval_df], ignore_index=True)
    tr_val_df = delete_duplicating_flows(tr_val_df)
    tr_val_df, underrepresented_protos = prune_targets(tr_val_df)

    test_df = delete_duplicating_flows(test_df)
    test_df, _ = prune_targets(test_df, underrepresented_protos=underrepresented_protos)

    suffix = get_hash(tr_val_df)
    save_dataset(tr_val_df, save_to=DATASET_DIR / f'train_{suffix}_no_{pattern_name}.csv')
    save_dataset(test_df, save_to=DATASET_DIR / f'test_{suffix}_no_{pattern_name}.csv')


if __name__ == '__main__':
    main()
