import hashlib
import logging
import pathlib

import pandas as pd

from settings import TARGET_CLASS_COLUMN, PCAP_OUTPUT_DIR, LOWER_BOUND_CLASS_OCCURRENCE, BASE_DIR
from pcap_files.preprocess_lan_pcaps import IOT_DEVICES

""" task-specific module, provided for the sake of reproducibility """

logger = logging.getLogger(__name__)

# signalling protos are common among all devices and it doesn't make sense to treat them separately
COMMON_PROTOCOLS = ['DNS', 'NTP', 'STUN']
GARBAGE_PROTOCOLS = ['ICMP', 'ICMPV6', 'DHCPV6', 'DHCP', 'Unknown', 'IGMP', 'SSDP']


def _load_parsed_results(dir_with_parsed_csvs=None):
    dir_with_parsed_csvs = pathlib.Path(dir_with_parsed_csvs)

    parsed_csvs = list(dir_with_parsed_csvs.glob('train_*.csv'))
    parsed_csvs += list(dir_with_parsed_csvs.glob('val_*.csv'))

    iot_datasets = []
    usual_traffic = []

    iot_categories = set(item.category for item in IOT_DEVICES)
    for csv_file in parsed_csvs:
        traffic_df = pd.read_csv(csv_file, na_filter='')
        if csv_file.name.startswith('train'):
            base_name = csv_file.name.split('train_')[-1]
        elif csv_file.name.startswith('val'):
            base_name = csv_file.name.split('val_')[-1]
        else:
            base_name = csv_file.name

        traffic_df['source_file'] = base_name

        if base_name.split('_')[0] in iot_categories:
            iot_datasets.append(traffic_df)
        else:
            usual_traffic.append(traffic_df)

    iot_traffic = pd.concat(iot_datasets, ignore_index=True)
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


def prune_targets(dataset, lower_bound=LOWER_BOUND_CLASS_OCCURRENCE):
    """ rm infrequent targets """
    proto_counts = dataset[TARGET_CLASS_COLUMN].value_counts()
    underepresented_protos = proto_counts[proto_counts < lower_bound].index.tolist()
    if underepresented_protos:
        logger.info(f'pruning the following targets: {underepresented_protos}')
        dataset = dataset.loc[~dataset[TARGET_CLASS_COLUMN].isin(underepresented_protos)]
    return dataset.reset_index(drop=True)


def save_dataset(dataset, save_to=None):
    """ simple data tracking/versioning via hash suffixes """

    def _hash_df(df):
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    def _get_current_commit_hash():
        """ get commit hash at HEAD """
        from git import Repo
        repo = Repo(BASE_DIR)
        return repo.head.commit.hexsha

    df_hash = _hash_df(dataset)
    head_hash = _get_current_commit_hash()
    if save_to is None:
        save_to = BASE_DIR / f'datasets/dataset_git_{head_hash}_content_{df_hash}.csv'
    dataset.to_csv(save_to, index=False)
    logger.info(f'saved dataset to {save_to}')
    return save_to


def read_dataset(filename):
    """ a simple wrapper for pandas """
    dataset = pd.read_csv(filename, na_filter='')
    logger.info(f'read {len(dataset)} flows from {filename}')
    return dataset


def prepare_data(pcap_dir):
    """ the order of operations matters """
    iot_traffic, usual_traffic = _load_parsed_results(pcap_dir)

    iot_traffic = _set_common_protos_targets(iot_traffic)
    usual_traffic = _set_common_protos_targets(usual_traffic)

    iot_traffic = _set_iot_devices_targets(iot_traffic)
    usual_traffic = _set_application_targets(usual_traffic)

    iot_traffic = _rm_garbage(iot_traffic)
    usual_traffic = _rm_garbage(usual_traffic, garbage=GARBAGE_PROTOCOLS + ['Amazon'], column_from=TARGET_CLASS_COLUMN)
    merged_traffic = pd.concat([usual_traffic, iot_traffic], ignore_index=True)
    return merged_traffic


if __name__ == '__main__':
    train_df = prepare_data('/media/raid_store/pretrained_traffic/train_csv')
    eval_df = prepare_data('/media/raid_store/pretrained_traffic/val_csv')
    total_df = prune_targets(pd.concat([train_df, eval_df], ignore_index=True))
    save_dataset(total_df)
