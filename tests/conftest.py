import json

import numpy as np
import pandas as pd
import pytest

from flow_parsing import features
import settings
from gpt_model.tokenizer import PacketTokenizer


@pytest.fixture
def dataset():
    return pd.read_csv(settings.TEST_STATIC_DIR / 'example_20packets.csv', na_filter=False)


@pytest.fixture
def raw_dataset_folder():
    return settings.TEST_STATIC_DIR / 'raw_csv'


@pytest.fixture
def raw_dataset_file(raw_dataset_folder):
    return raw_dataset_folder / 'example_raw_20packets.csv'


@pytest.fixture
def raw_dataset(raw_dataset_folder):
    return pd.read_csv(raw_dataset_folder / 'example_raw_20packets.csv', na_filter=False).\
        filter(regex='raw').\
        astype(np.float64)


@pytest.fixture
def raw_dataset_with_targets(raw_dataset_folder):
    df = pd.read_csv(raw_dataset_folder / 'example_raw_20packets.csv', na_filter=False)
    df.filter(regex='raw').astype(np.float64, copy=False)
    return df


@pytest.fixture
def classif_config():
    return {'SVM': {'type': 'OneVsOneClassifier',
                    'params': {'estimator': {'type': 'LinearSVC',
                                             'params': {'tol': 1e-05}}, 'n_jobs': -1},
                    'param_search_space': {'estimator__C': [0.1, 1, 10], 'estimator__loss': ['squared_hinge'],
                                           'estimator__dual': [True, False]}},
            'DecTree': {'type': 'DecisionTreeClassifier',
                        'param_search_space': {'max_depth': [6, 9, 12, 15, 18], 'max_features': [10, 20, 30, 40],
                                               'criterion': ['entropy']}},
            'GradBoost': {'type': 'GradientBoostingClassifier',
                          'param_search_space': {'n_estimators': [50], 'max_depth': [2, 3, 4, 5],
                                                 'learning_rate': [0.01, 0.05, 0.1]}}}


@pytest.fixture
def raw_matrix():
    size = 10
    raw_feature_matrix = np.zeros((size, 7))
    raw_feature_matrix[:, features.RMI.TIMESTAMP] = np.array(range(12312, size + 12312))
    raw_feature_matrix[:, features.RMI.IP_LEN] = np.array([13, 54, 345, 43, 44, 990, 1000, 23, 555, 1400])
    raw_feature_matrix[:, features.RMI.IS_CLIENT] = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 0])
    return raw_feature_matrix


@pytest.fixture
def quantized_packets():
    with open(settings.TEST_STATIC_DIR / 'quantized_pkts.json', 'r') as js:
        pkts = json.load(js)
    return np.array(pkts).reshape(-1, 20)


@pytest.fixture
def quantizer_checkpoint():
    return settings.TEST_STATIC_DIR / 'quantizer_checkpoint'


@pytest.fixture
def pcap_example_path():
    return (settings.BASE_DIR / 'flow_parsing/static/example.pcap').as_posix()


@pytest.fixture()
def tokenizer(quantizer_checkpoint):
    return PacketTokenizer.from_pretrained(quantizer_checkpoint, flow_size=20)
