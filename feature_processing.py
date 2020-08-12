import functools
import logging
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

from settings import TARGET_CLASS_COLUMN

from raw_packets_nfplugin import raw_packets_matrix as RMI

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


# These are non-complete subsets of handcrafted features
CONTINUOUS_NAMES = (
    'bulk0', 'bulk1', 'packet0', 'packet1', 'tcp_window_avg',
    'bulk_max', 'bulk_min', 'bulk_avg', 'bulk_median', 'bulk_25q', 'bulk_75q', 'bulk_bytes', 'bulk_number',
    'packet_max', 'packet_min', 'packet_avg', 'packet_median', 'packet_25q', 'packet_75q', 'packet_bytes',
    'packet_number',
)

CATEGORICAL_NAMES = (
    'found_tcp_flags',
)

FEATURE_NAMES = CONTINUOUS_NAMES + CATEGORICAL_NAMES


class FEATURE_PREFIX:
    client = 'client_'
    server = 'server_'


@functools.lru_cache(maxsize=2)
def create_empty_features(prefix: str, feature_list=FEATURE_NAMES) -> dict:
    return {f'{prefix}{feature}': 0. for feature in feature_list}


def _safe_vector_getter(vector, indexer) -> Union[int, float]:
    try:
        return vector[indexer]
    except IndexError:
        return np.nan


def calc_parameter_stats(feature_slice, prefix, feature_name):
    return {prefix + feature_name + '0': _safe_vector_getter(feature_slice, 0),
            prefix + feature_name + '1': _safe_vector_getter(feature_slice, 1),
            prefix + feature_name + '_max': np.max(feature_slice),
            prefix + feature_name + '_min': np.min(feature_slice),
            prefix + feature_name + '_avg': np.mean(feature_slice),
            prefix + feature_name + '_median': np.median(feature_slice),
            prefix + feature_name + '_25q': np.percentile(feature_slice, 25),
            prefix + feature_name + '_75q': np.percentile(feature_slice, 75),
            prefix + feature_name + '_bytes': np.sum(feature_slice),
            # counting non-empty bulks (packets with payload)
            prefix + feature_name + '_number': feature_slice[feature_slice > 0].shape[0]
            }


def _calc_unidirectional_flow_features(direction_slice, prefix='', features: Optional[list] = None) -> dict:
    # this asserts using of the listed features
    if features is None:
        features = create_empty_features(prefix)
    features[prefix + 'found_tcp_flags'] = sorted(set(direction_slice[:, RMI.TCP_FLAGS]))
    features[prefix + 'tcp_window_avg'] = np.mean(direction_slice[:, RMI.TCP_WINDOW])

    features.update(calc_parameter_stats(direction_slice[:, RMI.TRANSP_PAYLOAD], prefix, 'bulk'))
    features.update(calc_parameter_stats(direction_slice[:, RMI.IP_LEN], prefix, 'packet'))
    return features


def _get_iat(raw_matrix):
    """ calcs inter-packet times """
    timestamps = raw_matrix[:, RMI.TIMESTAMP]
    next_timestamps = np.roll(timestamps, 1)
    iat = timestamps - next_timestamps
    iat[0] = 0
    return iat


def _get_packet_features(raw_matrix):
    """ sets packet len features negative for server-side packets """
    packet_features = np.zeros(raw_matrix.shape[0])
    client_indexer = np.where(raw_matrix[:, RMI.IS_CLIENT] == 1)[0]
    server_indexer = np.where(raw_matrix[:, RMI.IS_CLIENT] == 0)[0]
    packet_features[client_indexer] = raw_matrix[client_indexer, RMI.IP_LEN]
    packet_features[server_indexer] = raw_matrix[server_indexer, RMI.IP_LEN] * -1
    return packet_features


def calc_raw_features(raw_matrix: np.ndarray, packet_limit) -> dict:
    """ estimates features for flow models that are used for data-augmentation purposes """
    iat_features = _get_iat(raw_matrix)
    packet_features = _get_packet_features(raw_matrix)

    features = dict.fromkeys(generate_raw_feature_names(packet_limit))
    for index in range(packet_limit):
        features['raw_packet' + str(index)] = _safe_vector_getter(packet_features, index)
        features['raw_iat' + str(index)] = _safe_vector_getter(iat_features, index)

    return features


def calc_flow_features(raw_features: np.ndarray) -> dict:
    """ estimates discriminative features for flow classification """
    client_slice = raw_features[raw_features[:, RMI.IS_CLIENT] == 1]
    if client_slice.shape[0] > 0:
        client_features = _calc_unidirectional_flow_features(client_slice, prefix=FEATURE_PREFIX.client)
    else:
        client_features = create_empty_features(prefix=FEATURE_PREFIX.client)

    server_slice = raw_features[raw_features[:, RMI.IS_CLIENT] == 0]
    if server_slice.shape[0] > 0:
        server_features = _calc_unidirectional_flow_features(server_slice, prefix=FEATURE_PREFIX.server)
    else:
        server_features = create_empty_features(prefix=FEATURE_PREFIX.server)

    total_features = dict(**client_features, **server_features)
    return total_features


def generate_raw_feature_names(flow_size, base_features: Tuple[str] = ('packet', 'iat')):
    return [f'raw_{feature}{index}'
            for index in range(flow_size)
            for feature in base_features]
