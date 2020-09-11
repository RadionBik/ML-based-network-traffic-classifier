import functools
import logging
from typing import Tuple, Union, Optional

import numpy as np

from .raw_packets_nfplugin import raw_packets_matrix as RMI

logger = logging.getLogger(__name__)

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


def calc_parameter_stats(feature_slice, prefix, feature_name) -> dict:
    return {
        prefix + feature_name + '0': _safe_vector_getter(feature_slice, 0),
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


def inter_packet_times_from_timestamps(timestamps):
    if len(timestamps) == 0:
        return timestamps
    next_timestamps = np.roll(timestamps, 1)
    ipt = timestamps - next_timestamps
    ipt[0] = 0
    return ipt


def _get_iat(raw_matrix):
    """ calcs inter-packet times """
    timestamps = raw_matrix[:, RMI.TIMESTAMP]
    return inter_packet_times_from_timestamps(timestamps)


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


def calc_stat_features(raw_features: np.ndarray) -> dict:
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
