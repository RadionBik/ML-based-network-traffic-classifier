import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder

import flow_parser
from datasets import TARGET_CLASS_COLUMN

logger = logging.getLogger(__name__)

RANDOM_SEED = 1


class TransformNotFound(FileNotFoundError):
    def __init(self, filename):
        super().__init__('Transform {} was not found, please check it exists or run training first'.format(
            filename
        ))


class Featurizer:
    """
    Featurizer processes raw features from a pandas object by merging results from scalers and one-hot encoders
    and encodes target labels
    """

    def __init__(self,
                 cont_features=None,
                 categorical_features=None,
                 consider_tcp_flags=True,
                 consider_j3a=True,
                 target_column=TARGET_CLASS_COLUMN):

        self.target_encoder = LabelEncoder()
        self.target_column = target_column

        self.categorical_features = ['ip_proto'] if not categorical_features else categorical_features
        self.cont_features = self._get_cont_features() if not cont_features else cont_features
        self.consider_tcp_flags = consider_tcp_flags
        self.consider_j3a = consider_j3a

        feature_set = [
            ("scaler", StandardScaler(), self.cont_features),
            ("one_hot", OneHotEncoder(handle_unknown='ignore', sparse=False), self.categorical_features),
        ]
        if self.consider_j3a:
            feature_set.append(("one_hot_j3a",
                                OneHotEncoder(handle_unknown='ignore', sparse=False),
                                ['ndpi_j3ac', 'ndpi_j3as']))

        if self.consider_tcp_flags:
            feature_set.append(("one_hot_tcp_flags",
                                OneHotEncoder(handle_unknown='ignore', sparse=False),
                                ['client_found_tcp_flags', 'server_found_tcp_flags']))

        self.transformer = ColumnTransformer(feature_set)

    @staticmethod
    def _get_cont_features():
        # here we expect features to be consistent with flow_parser's
        base_features = [feat for feat in flow_parser.FEATURE_NAMES
                         if 'bulk' in feat or 'packet' in feat]
        cont_features = []
        for prefix in [flow_parser.FEATURE_PREFIX.client, flow_parser.FEATURE_PREFIX.server]:
            for feature in base_features:
                cont_features.append(prefix+feature)
        return cont_features

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        X_train = self.transformer.fit_transform(data)
        self.target_encoder.fit(data[self.target_column])
        return X_train

    def fit_transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_train = self.transformer.fit_transform(data)
        y_train = self.target_encoder.fit_transform(data[self.target_column])
        return X_train, y_train

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        X_test = self.transformer.transform(data)
        return X_test

    def transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_test = self.transformer.transform(data)
        y_test = self.target_encoder.transform(data[self.target_column])
        return X_test, y_test

    def encode(self, data: pd.DataFrame) -> np.ndarray:
        return self.target_encoder.transform(data[self.target_column])
