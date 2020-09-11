from typing import Tuple
import logging

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from flow_parsing.features import (
    FEATURE_PREFIX,
    FEATURE_NAMES,
    CONTINUOUS_NAMES,
    generate_raw_feature_names,
    calc_parameter_stats
)
from evaluation_utils.modeling import flows_to_packets
from settings import TARGET_CLASS_COLUMN, DEFAULT_PACKET_LIMIT_PER_FLOW


logger = logging.getLogger(__name__)


class Featurizer:
    """
    Featurizer processes features from a pandas object by merging results from scalers and one-hot encoders
    and encodes target labels
    """

    def __init__(self,
                 cont_features=None,
                 categorical_features=None,
                 consider_tcp_flags=True,
                 consider_j3a=True,
                 raw_feature_num=0,
                 consider_raw_feature_iat=False,
                 target_column=TARGET_CLASS_COLUMN):

        self.target_encoder = LabelEncoder()
        self.transformer = None
        self.target_column = target_column

        self.categorical_features = ['ip_proto'] if categorical_features is None else categorical_features
        self.cont_features = self._get_cont_features() if cont_features is None else cont_features
        self.consider_tcp_flags = consider_tcp_flags
        self.consider_j3a = consider_j3a

        if self.consider_j3a:
            self.categorical_features.extend(['ndpi_j3ac', 'ndpi_j3as'])

        if self.consider_tcp_flags:
            self.categorical_features.extend([f'{FEATURE_PREFIX.client}found_tcp_flags',
                                              f'{FEATURE_PREFIX.server}found_tcp_flags'])

        self.raw_features = generate_raw_feature_names(
            raw_feature_num,
            base_features=('packet', 'iat') if consider_raw_feature_iat else ('packet',)
        )

    @staticmethod
    def _get_cont_features():
        # here we expect features to be consistent with flow_parser's
        base_features = [feat for feat in FEATURE_NAMES
                         if 'bulk' in feat or 'packet' in feat]
        cont_features = []
        for prefix in [FEATURE_PREFIX.client, FEATURE_PREFIX.server]:
            for feature in base_features:
                cont_features.append(prefix + feature)
        return cont_features

    def _filter_non_existing_features(self, data: pd.DataFrame):
        data_features = set(data.columns)

        if set(self.raw_features) - data_features:
            found_features = list(set(self.raw_features) & data_features)
            logger.warning(f'skipping the following raw features: {set(self.raw_features) - data_features}')
            self.raw_features = found_features

        if set(self.cont_features) - data_features:
            found_features = list(set(self.cont_features) & data_features)
            logger.warning(f'skipping the following continuous features: {set(self.cont_features) - data_features}')
            self.cont_features = found_features

        if set(self.categorical_features) - data_features:
            found_features = list(set(self.categorical_features) & data_features)
            logger.warning(f'skipping the following categorical features: '
                           f'{set(self.categorical_features) - data_features}')
            self.categorical_features = found_features

    def _fit_transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """ init transformers upon actual fitting to check for non-existing columns """
        self._filter_non_existing_features(data)
        feature_set = []

        if self.cont_features:
            feature_set.append(("scaler", StandardScaler(),
                                self.cont_features))

        if self.categorical_features:
            feature_set.append(("one_hot", OneHotEncoder(handle_unknown='ignore', sparse=False),
                                self.categorical_features)),

        if self.raw_features:
            # TODO replace with PacketScaler
            feature_set.append(('raw_features', StandardScaler(), self.raw_features))

        self.transformer = ColumnTransformer(feature_set)

        X_train = self.transformer.fit_transform(data)
        y_train = self.target_encoder.fit_transform(data[self.target_column])
        logger.info(f'{X_train.shape[0]} train samples with {self.n_classes} classes')
        return X_train, y_train

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self._fit_transform_encode(data)[0]

    def fit_transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self._fit_transform_encode(data)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        X_test = self.transformer.transform(data)
        return X_test

    def transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_test = self.transformer.transform(data)
        y_test = self.target_encoder.transform(data[self.target_column])
        return X_test, y_test

    def encode(self, data: pd.DataFrame) -> np.ndarray:
        return self.target_encoder.transform(data[self.target_column])

    @property
    def n_classes(self):
        return len(self.target_encoder.classes_)

    def calc_packets_stats_from_raw(self, data: pd.DataFrame):

        def calc_flow_packet_stats(flow):
            subflow = flow[:2 * DEFAULT_PACKET_LIMIT_PER_FLOW]
            packets = flows_to_packets(subflow)
            packets_from = packets[packets[:, 0] > 0, 0]
            packets_to = packets[packets[:, 0] < 0, 0] * -1
            stats = {}
            for direction, packets in zip(
                    (FEATURE_PREFIX.server, FEATURE_PREFIX.client),
                    (packets_from, packets_to)
            ):
                try:
                    stats.update(calc_parameter_stats(packets, direction, 'packet'))
                except ValueError:
                    continue
            return stats

        if any(column in CONTINUOUS_NAMES for column in data.columns):
            logger.warning('packet stats has been found in dataframe, skipping')
            return data
        raw = data.filter(regex='raw_')
        pandarallel.initialize()
        packet_stats = raw.parallel_apply(calc_flow_packet_stats, axis=1, raw=True, result_type='expand').tolist()
        packet_stats = pd.DataFrame(packet_stats).fillna(0)
        logger.info('calculated packet stats from raw packet sizes')
        return data.join(packet_stats)
