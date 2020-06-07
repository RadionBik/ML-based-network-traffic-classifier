import pytest
import pandas as pd
import numpy as np

import settings
import flow_parser


@pytest.fixture
def dataset():
    return pd.read_csv(settings.TEST_STATIC_DIR / 'example_20packets.csv', na_filter='')


@pytest.fixture
def raw_dataset():
    return pd.read_csv(settings.TEST_STATIC_DIR / 'raw_example_20packets.csv', na_filter='')


@pytest.fixture
def classif_config():
    return {'SVM': {'type': 'OneVsOneClassifier',
                    'params': {'estimator': {'type': 'LinearSVC',
                                             'params': {'tol': 1e-05}}, 'n_jobs': -1},
                    'param_search_space': {'estimator__C': [0.1, 1, 10], 'estimator__loss': ['squared_hinge'],
                                           'estimator__dual': [True, False]}},
            'DecTree': {'type': 'DecisionTreeClassifier',
                        'param_search_space': {'max_depth': [6, 9, 12, 15, 18], 'max_features': [10, 20, 30],
                                               'criterion': ['entropy']}},
            'GradBoost': {'type': 'GradientBoostingClassifier',
                          'param_search_space': {'n_estimators': [50], 'max_depth': [2, 3, 4, 5],
                                                 'learning_rate': [0.01, 0.05, 0.1]}}}


@pytest.fixture
def raw_matrix():
    size = 10
    raw_feature_matrix = np.zeros((size, 7))
    raw_feature_matrix[:, flow_parser.RMI.TIMESTAMP] = np.array(range(12312, size+12312))
    raw_feature_matrix[:, flow_parser.RMI.IP_LEN] = np.array([13, 54, 345, 43, 44, 990, 1000, 23, 555, 1400])
    raw_feature_matrix[:, flow_parser.RMI.IS_CLIENT] = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 0])
    return raw_feature_matrix
