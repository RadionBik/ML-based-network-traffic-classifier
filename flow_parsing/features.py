import functools
import logging
from typing import Tuple, Union, Optional

import numpy as np
from nfstream.flow import NFlow

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


def inter_packet_times_from_timestamps(timestamps):
    if len(timestamps) == 0:
        return timestamps
    next_timestamps = np.roll(timestamps, 1)
    ipt = timestamps - next_timestamps
    ipt[0] = 0
    return ipt


def generate_raw_feature_names(flow_size, base_features: Tuple[str] = ('packet', 'iat')):
    return [f'raw_{feature}{index}'
            for index in range(flow_size)
            for feature in base_features]


def calc_raw_features(flow: NFlow) -> dict:
    """ selects PS and IPT features  """
    packet_limit = len(flow.splt_ps)
    features = dict.fromkeys(generate_raw_feature_names(packet_limit))
    for index in range(packet_limit):
        ps = flow.splt_ps[index]
        ipt = flow.splt_piat_ms[index]

        if flow.splt_direction[index] == 1:
            ps = flow.splt_ps[index] * -1
        elif flow.splt_direction[index] == -1:
            ps = np.nan
            ipt = np.nan

        features['raw_packet' + str(index)] = ps
        features['raw_iat' + str(index)] = ipt

    return features


def _calc_unidirectional_flow_features(flow: NFlow, direction_idxs, prefix='', features: Optional[list] = None) -> dict:
    # this asserts using of the listed features
    if features is None:
        features = create_empty_features(prefix)

    features.update(calc_parameter_stats(np.array(flow.splt_ps)[direction_idxs], prefix, 'packet'))

    features[prefix + 'found_tcp_flags'] = sorted(set(flow.udps.tcp_flag[direction_idxs]))
    features[prefix + 'tcp_window_avg'] = np.mean(flow.udps.tcp_window[direction_idxs])
    features.update(calc_parameter_stats(flow.udps.bulk[direction_idxs], prefix, 'bulk'))

    return features


def calc_stat_features(flow: NFlow) -> dict:
    """ estimates derivative discriminative features for flow classification from:
        packet size, payload size, TCP window, TCP flag
    """
    direction = np.array(flow.splt_direction)
    client_idxs = direction == 0
    server_idxs = direction == 1

    if client_idxs.sum() > 0:
        client_features = _calc_unidirectional_flow_features(flow, client_idxs, prefix=FEATURE_PREFIX.client)
    else:
        client_features = create_empty_features(prefix=FEATURE_PREFIX.client)

    if server_idxs.sum() > 0:
        server_features = _calc_unidirectional_flow_features(flow, server_idxs, prefix=FEATURE_PREFIX.server)
    else:
        server_features = create_empty_features(prefix=FEATURE_PREFIX.server)

    total_features = dict(**client_features, **server_features)
    return total_features
