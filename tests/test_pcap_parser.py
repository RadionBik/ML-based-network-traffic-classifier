import json

import pandas as pd

from flow_parsing import pcap_parser


def test_feature_persistence(pcap_example_path):
    features = pcap_parser.parse_pcap_to_dataframe(pcap_example_path, online_mode=False). \
        sort_values('flow_id', axis=0). \
        reset_index(drop=True)
    features2 = pcap_parser.parse_pcap_to_dataframe(pcap_example_path, online_mode=False). \
        sort_values('flow_id', axis=0). \
        reset_index(drop=True)
    pd.testing.assert_frame_equal(features, features2)


def _serialize_tcp_flag(x):
    indexer = x.index.str.endswith('tcp_flags')
    x.iloc[indexer] = x.iloc[indexer].apply(json.dumps)
    return x


def test_parser_output(dataset, pcap_example_path):
    parsed_features = pcap_parser.parse_pcap_to_dataframe(pcap_example_path, online_mode=False). \
        sort_values('flow_id', axis=0). \
        reset_index(drop=True)

    parsed_features = parsed_features.apply(_serialize_tcp_flag, axis=1)
    dataset = dataset.astype(parsed_features.dtypes). \
        sort_values('flow_id', axis=0). \
        reset_index(drop=True)
    pd.testing.assert_frame_equal(parsed_features, dataset,
                                  check_less_precise=2,
                                  check_like=True,
                                  check_categorical=False)


def test_raw_parser_output(raw_dataset_with_targets, pcap_example_path):
    parsed_features = pcap_parser.parse_pcap_to_dataframe(pcap_example_path,
                                                          derivative_features=False,
                                                          raw_features=20,
                                                          online_mode=False)
    parsed_features = parsed_features. \
        sort_values('flow_id', axis=0). \
        reset_index(drop=True). \
        filter(regex='raw')

    raw_dataset = raw_dataset_with_targets. \
        sort_values('flow_id', axis=0). \
        reset_index(drop=True). \
        filter(regex='raw')
    raw_dataset = raw_dataset.astype(parsed_features.dtypes)
    pd.testing.assert_frame_equal(parsed_features, raw_dataset,
                                  check_less_precise=2,
                                  check_like=True,
                                  check_categorical=False)
