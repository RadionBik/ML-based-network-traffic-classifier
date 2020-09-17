import json

import numpy as np
import pandas as pd

from flow_parsing import pcap_parser, features


def test_feature_persistence(pcap_example_path):
    features = pcap_parser.parse_pcap_to_dataframe(pcap_example_path)
    features2 = pcap_parser.parse_pcap_to_dataframe(pcap_example_path)
    assert features.equals(features2)


def _serialize_tcp_flag(x):
    indexer = x.index.str.endswith('tcp_flags')
    x.iloc[indexer] = x.iloc[indexer].apply(json.dumps)
    return x


def test_parser_output(dataset, pcap_example_path):
    parsed_features = pcap_parser.parse_pcap_to_dataframe(pcap_example_path, online_mode=False)
    parsed_features = parsed_features.apply(_serialize_tcp_flag, axis=1)
    dataset = dataset.astype(parsed_features.dtypes)
    pd.testing.assert_frame_equal(parsed_features, dataset,
                                  check_less_precise=2,
                                  check_like=True,
                                  check_categorical=False)


def test_raw_parser_output(raw_dataset, pcap_example_path):
    parsed_features = pcap_parser.parse_pcap_to_dataframe(pcap_example_path,
                                                          derivative_features=False,
                                                          raw_features=20,
                                                          online_mode=False)
    parsed_features = parsed_features.filter(regex='raw')
    pd.testing.assert_frame_equal(parsed_features, raw_dataset,
                                  check_less_precise=2,
                                  check_like=True,
                                  check_categorical=False)


def test_raw_model_features(raw_matrix):
    iat_features = features._get_iat(raw_matrix)
    packet_features = features._get_packet_features(raw_matrix)
    model_features = np.vstack([iat_features, packet_features]).T
    expected = np.array([[0, -13],
                         [1, 54],
                         [1, 345],
                         [1, -43],
                         [1, -44],
                         [1, 990],
                         [1, 1000],
                         [1, 23],
                         [1, 555],
                         [1, -1400],
                         ])
    assert (model_features == expected).all()
