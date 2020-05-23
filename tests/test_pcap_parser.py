import json

import pandas as pd

import flow_parser
import settings


def test_feature_persistence(filename):
    pcap_filename = (settings.BASE_DIR / 'pcap_files/example.pcap').as_posix()
    features = flow_parser.parse_to_dataframe(pcap_filename)
    features2 = flow_parser.parse_to_dataframe(pcap_filename)
    assert features.equals(features2)


def test_parser_output(dataset):
    def _serialize(x):
        indexer = x.index.str.endswith('tcp_flags')
        x.iloc[indexer] = x.iloc[indexer].apply(json.dumps)
        return x

    pcap_filename = (settings.BASE_DIR / 'pcap_files/example.pcap').as_posix()
    parsed_features = flow_parser.parse_to_dataframe(pcap_filename)
    parsed_features = parsed_features.apply(_serialize, axis=1)
    pd.testing.assert_frame_equal(parsed_features, dataset,
                                  check_less_precise=2,
                                  check_like=True,
                                  check_categorical=False)
