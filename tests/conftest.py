import pytest
import pandas as pd

import settings


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