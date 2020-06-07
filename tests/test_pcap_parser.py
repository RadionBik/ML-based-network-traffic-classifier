import json

import pandas as pd
import numpy as np

import flow_parser
import settings


def test_feature_persistence():
    pcap_filename = (settings.BASE_DIR / 'pcap_files/example.pcap').as_posix()
    features = flow_parser.parse_features_to_dataframe(pcap_filename)
    features2 = flow_parser.parse_features_to_dataframe(pcap_filename)
    assert features.equals(features2)


def _serialize_tcp_flag(x):
    indexer = x.index.str.endswith('tcp_flags')
    x.iloc[indexer] = x.iloc[indexer].apply(json.dumps)
    return x


def test_parser_output(dataset):

    pcap_filename = (settings.BASE_DIR / 'pcap_files/example.pcap').as_posix()
    parsed_features = flow_parser.parse_features_to_dataframe(pcap_filename)
    parsed_features = parsed_features.apply(_serialize_tcp_flag, axis=1)
    pd.testing.assert_frame_equal(parsed_features, dataset,
                                  check_less_precise=2,
                                  check_like=True,
                                  check_categorical=False)


def test_raw_parser_output(raw_dataset):
    pcap_filename = (settings.BASE_DIR / 'pcap_files/example.pcap').as_posix()
    parsed_features = flow_parser.parse_features_to_dataframe(pcap_filename, raw_features=True)
    parsed_features = parsed_features.apply(_serialize_tcp_flag, axis=1)
    pd.testing.assert_frame_equal(parsed_features, raw_dataset,
                                  check_less_precise=2,
                                  check_like=True,
                                  check_categorical=False)


def test_raw_model_features(raw_matrix):
    iat_features = flow_parser._get_iat(raw_matrix)
    packet_features = flow_parser._get_packet_features(raw_matrix)
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
