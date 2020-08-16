import hashlib
import logging
import pathlib

import pandas as pd
from typing import Iterable

from pcap_files.preprocess_lan_pcaps import IOT_DEVICES
from settings import TARGET_CLASS_COLUMN, LOWER_BOUND_CLASS_OCCURRENCE, BASE_DIR

""" task-specific module, provided for the sake of reproducibility """

logger = logging.getLogger(__name__)

# signalling protos are common among all devices and it doesn't make sense to treat them separately
COMMON_PROTOCOLS = ['DNS', 'NTP', 'STUN']
GARBAGE_PROTOCOLS = ['ICMP', 'ICMPV6', 'DHCPV6', 'DHCP', 'Unknown', 'IGMP', 'SSDP']


def read_dataset(filename, fill_na=False) -> pd.DataFrame:
    """ a simple wrapper for pandas """
    dataset = pd.read_csv(filename, na_filter=True)
    if fill_na:
        dataset = dataset.fillna(0)
    logger.info(f'read {len(dataset)} flows from {filename}')
    return dataset


def _load_parsed_results(dir_with_parsed_csvs, filename_patterns_to_exclude: Iterable[str]):
    dir_with_parsed_csvs = pathlib.Path(dir_with_parsed_csvs)

    parsed_csvs = list(dir_with_parsed_csvs.glob('*.csv'))

    iot_datasets = []
    usual_traffic = []

    iot_categories = set(item.category for item in IOT_DEVICES)
    for csv_file in parsed_csvs:
        # skip non-home and IoT files
        if filename_patterns_to_exclude and any(pattern in csv_file.name for pattern in filename_patterns_to_exclude):
            logger.info(f'skipping {csv_file} due to "filename_patterns_to_exclude"')
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


def get_hash(df):
    def _hash_df(df):
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    def _get_current_commit_hash():
        """ get commit hash at HEAD """
        from git import Repo
        repo = Repo(BASE_DIR)
        return repo.head.commit.hexsha

    try:
        df_hash = _get_current_commit_hash()
    except Exception:
        df_hash = _hash_df(df)
    return df_hash


def save_dataset(dataset, save_to=None):
    """ simple data tracking/versioning via hash suffixes """

    if save_to is None:
        save_to = BASE_DIR / f'datasets/dataset_{get_hash(dataset)}.csv'
    dataset.to_csv(save_to, index=False)
    logger.info(f'saved dataset to {save_to}')
    return save_to


def delete_duplicating_flows(dataset):
    def to_session_id(flow_id):
        proto, conn1, conn2 = flow_id.split(' ')
        return proto, frozenset([conn1, conn2])

    dataset['session_id'] = dataset['flow_id'].apply(to_session_id)
    dataset = dataset.drop_duplicates(subset=['session_id'])
    dataset.drop(columns='session_id', inplace=True)
    logger.info(f'{dataset.shape[0]} flows left after deduplication')
    return dataset


def prepare_data(csv_dir, remove_garbage=True, filename_patterns_to_exclude=None):
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
    excluded_patterns = ('raw_csv', '202004')
    train_df = prepare_data('/media/raid_store/pretrained_traffic/train_csv',
                            filename_patterns_to_exclude=excluded_patterns)
    eval_df = prepare_data('/media/raid_store/pretrained_traffic/val_csv',
                           filename_patterns_to_exclude=excluded_patterns)
    test_df = prepare_data('/media/raid_store/pretrained_traffic/test_csv',
                           filename_patterns_to_exclude=excluded_patterns)
    tr_val_df = pd.concat([train_df, eval_df], ignore_index=True)
    tr_val_df = delete_duplicating_flows(tr_val_df)
    tr_val_df, underrepresented_protos = prune_targets(tr_val_df)

    test_df = delete_duplicating_flows(test_df)
    test_df, _ = prune_targets(test_df, underrepresented_protos=underrepresented_protos)

    suffix = get_hash(tr_val_df)
    save_dataset(tr_val_df, save_to=BASE_DIR / f'datasets/train_{suffix}.csv')
    save_dataset(test_df, save_to=BASE_DIR / f'datasets/test_{suffix}.csv')


if __name__ == '__main__':
    main()
