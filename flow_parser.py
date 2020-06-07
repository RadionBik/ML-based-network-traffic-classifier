import argparse
import functools
import logging

import dpkt
import pandas as pd
import nfstream
import numpy as np

import settings

logger = logging.getLogger('flow_parser')


class RawFeatureMatrixIndexes:
    TIMESTAMP = 0
    IP_LEN = 1
    TRANSP_PAYLOAD = 2
    TCP_FLAGS = 3
    TCP_WINDOW = 4
    IP_PROTO = 5
    IS_CLIENT = 6


RMI = RawFeatureMatrixIndexes

FEATURE_NAMES = [
    'found_tcp_flags', 'bulk0', 'bulk1', 'packet0', 'packet1', 'tcp_window_avg',
    'bulk_max', 'bulk_min', 'bulk_avg', 'bulk_median', 'bulk_25q', 'bulk_75q', 'bulk_bytes', 'bulk_number',
    'packet_max', 'packet_min', 'packet_avg', 'packet_median', 'packet_25q', 'packet_75q', 'packet_bytes',
    'packet_number'
]


class FEATURE_PREFIX:
    client = 'client_'
    server = 'server_'


@functools.lru_cache(maxsize=2)
def _create_empty_features(prefix: str) -> dict:
    return {f'{prefix}{feature}': 0. for feature in FEATURE_NAMES}


def _safe_matrix_getter(matrix, row_indexer, column_indexer):
    try:
        return matrix[row_indexer, column_indexer]
    except IndexError:
        return 0


def _safe_vector_getter(matrix, indexer):
    try:
        return matrix[indexer]
    except IndexError:
        return 0


def _calc_unidirectional_flow_features(direction_slice, prefix='') -> dict:
    # this asserts using of the listed features
    features = _create_empty_features(prefix)
    features[prefix + 'found_tcp_flags'] = sorted(set(direction_slice[:, RMI.TCP_FLAGS]))

    features[prefix + 'bulk0'] = _safe_matrix_getter(direction_slice, 0, RMI.TRANSP_PAYLOAD)
    features[prefix + 'bulk1'] = _safe_matrix_getter(direction_slice, 1, RMI.TRANSP_PAYLOAD)

    features[prefix + 'packet0'] = _safe_matrix_getter(direction_slice, 0, RMI.IP_LEN)
    features[prefix + 'packet1'] = _safe_matrix_getter(direction_slice, 1, RMI.IP_LEN)

    features[prefix + 'tcp_window_avg'] = np.mean(direction_slice[:, RMI.TCP_WINDOW])

    features[prefix + 'bulk_max'] = np.max(direction_slice[:, RMI.TRANSP_PAYLOAD])
    features[prefix + 'bulk_min'] = np.min(direction_slice[:, RMI.TRANSP_PAYLOAD])
    features[prefix + 'bulk_avg'] = np.mean(direction_slice[:, RMI.TRANSP_PAYLOAD])
    features[prefix + 'bulk_median'] = np.median(direction_slice[:, RMI.TRANSP_PAYLOAD])
    features[prefix + 'bulk_25q'] = np.percentile(direction_slice[:, RMI.TRANSP_PAYLOAD], 25)
    features[prefix + 'bulk_75q'] = np.percentile(direction_slice[:, RMI.TRANSP_PAYLOAD], 75)
    features[prefix + 'bulk_bytes'] = np.sum(direction_slice[:, RMI.TRANSP_PAYLOAD])
    # counting non-empty packets (with payload)
    features[prefix + 'bulk_number'] = direction_slice[direction_slice[:, RMI.TRANSP_PAYLOAD] > 0].shape[0]

    features[prefix + 'packet_max'] = np.max(direction_slice[:, RMI.IP_LEN])
    features[prefix + 'packet_min'] = np.min(direction_slice[:, RMI.IP_LEN])
    features[prefix + 'packet_avg'] = np.mean(direction_slice[:, RMI.IP_LEN])
    features[prefix + 'packet_median'] = np.median(direction_slice[:, RMI.IP_LEN])
    features[prefix + 'packet_25q'] = np.percentile(direction_slice[:, RMI.IP_LEN], 25)
    features[prefix + 'packet_75q'] = np.percentile(direction_slice[:, RMI.IP_LEN], 75)
    features[prefix + 'packet_bytes'] = np.sum(direction_slice[:, RMI.IP_LEN])
    features[prefix + 'packet_number'] = direction_slice[:, RMI.IP_LEN].shape[0]
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


def calc_raw_features(raw_matrix: np.ndarray) -> dict:
    """ estimates features for flow models that are used for data-augmentation purposes """
    iat_features = _get_iat(raw_matrix)
    packet_features = _get_packet_features(raw_matrix)

    features = {}
    for index in range(settings.PACKET_LIMIT_PER_FLOW):
        features['raw_packet' + str(index)] = _safe_vector_getter(packet_features, index)
        features['raw_iat' + str(index)] = _safe_vector_getter(iat_features, index)

    return features


def calc_flow_features(raw_features: np.ndarray) -> dict:
    """ estimates discriminative features for flow classification """
    client_slice = raw_features[raw_features[:, RMI.IS_CLIENT] == 1]
    if client_slice.shape[0] > 0:
        client_features = _calc_unidirectional_flow_features(client_slice, prefix=FEATURE_PREFIX.client)
    else:
        client_features = _create_empty_features(prefix=FEATURE_PREFIX.client)

    server_slice = raw_features[raw_features[:, RMI.IS_CLIENT] == 0]
    if server_slice.shape[0] > 0:
        server_features = _calc_unidirectional_flow_features(server_slice, prefix=FEATURE_PREFIX.server)
    else:
        server_features = _create_empty_features(prefix=FEATURE_PREFIX.server)

    total_features = dict(**client_features, **server_features)
    return total_features


class raw_packets_matrix(nfstream.NFPlugin):
    @staticmethod
    def _fill_flow_stats(obs, raw_feature_matrix, counter=0):
        raw_feature_matrix[counter, RMI.TIMESTAMP] = obs.time
        raw_feature_matrix[counter, RMI.IP_LEN] = obs.ip_size
        raw_feature_matrix[counter, RMI.TRANSP_PAYLOAD] = obs.payload_size
        raw_feature_matrix[counter, RMI.TCP_FLAGS] = int(''.join(str(i) for i in obs.tcpflags), 2)
        if obs.protocol == 6 and obs.version == 4:
            packet = dpkt.ip.IP(obs.ip_packet)
            raw_feature_matrix[counter, RMI.TCP_WINDOW] = packet.data.win
        raw_feature_matrix[counter, RMI.IP_PROTO] = obs.protocol
        raw_feature_matrix[counter, RMI.IS_CLIENT] = 1 if obs.direction == 0 else 0
        return raw_feature_matrix

    def on_init(self, obs):
        raw_feature_matrix = np.zeros((settings.PACKET_LIMIT_PER_FLOW, 7))
        return self._fill_flow_stats(obs, raw_feature_matrix)

    def on_update(self, obs, entry):
        if entry.bidirectional_packets > settings.PACKET_LIMIT_PER_FLOW:
            return entry.raw_packets_matrix
        return self._fill_flow_stats(obs, entry.raw_packets_matrix, counter=entry.bidirectional_packets - 1)

    def on_expire(self, entry):
        # rm unfilled matrix rows
        entry.raw_packets_matrix = entry.raw_packets_matrix[entry.raw_packets_matrix[:, RMI.TIMESTAMP] > 0]


def _init_streamer(source, online_mode=False):
    # since we decide and set routing policy upon first occurrence of a flow we don't care about its re-export
    idle_timeout = 10 if online_mode else 10e5
    return nfstream.NFStreamer(source=source,
                               statistics=False,
                               idle_timeout=60,
                               active_timeout=idle_timeout,
                               plugins=[raw_packets_matrix(volatile=False)],
                               enable_guess=True)


def flow_processor(source, raw_features: bool = False):
    def _make_flow_id(entry):
        return f'{settings.IP_PROTO_MAPPING[entry.protocol]} ' \
               f'{entry.src_ip}:{entry.src_port} ' \
               f'{entry.dst_ip}:{entry.dst_port}'

    for flow_number, entry in enumerate(_init_streamer(source)):
        label_features = {
            'flow_id': _make_flow_id(entry),
            'ndpi_app': entry.application_name,
            'ndpi_category': entry.category_name,
            'ndpi_client_info': entry.client_info,
            'ndpi_server_info': entry.server_info,
            'ndpi_j3ac': entry.j3a_client,
            'ndpi_j3as': entry.j3a_server,
            'ip_proto': settings.IP_PROTO_MAPPING[entry.protocol],
        }
        flow_features = calc_flow_features(entry.raw_packets_matrix)
        if raw_features:
            raw_packets = calc_raw_features(entry.raw_packets_matrix)
            total_features = dict(**label_features, **flow_features, **raw_packets)
        else:
            total_features = dict(**label_features, **flow_features)
        if flow_number > 0 == flow_number % 1000:
            logger.info(f'processed {flow_number} flows...')
        yield total_features


def parse_features_to_dataframe(pcap_file: str, raw_features: bool = False) -> pd.DataFrame:
    flows = []
    logger.info(f'started parsing file {pcap_file}')
    for flow in flow_processor(pcap_file, raw_features):
        flows.append(flow)
    return pd.DataFrame(flows)


def _get_output_csv_filename(args) -> str:
    core_name = args.pcapfile.split('/')[-1].split('.')[0]
    if args.raw:
        core_name = core_name + '_raw'
    output_csv = settings.PCAP_OUTPUT_DIR / f'{core_name}_{settings.PACKET_LIMIT_PER_FLOW}packets.csv'
    logger.info(f'target .csv path: {output_csv}')
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pcapfile",
        help="pcap file",
        default='pcap_files/example.pcap',
    )
    parser.add_argument(
        "-o", "--output",
        help="output .csv file destination",
    )
    parser.add_argument(
        "--raw",
        dest='raw',
        action='store_true',
        help="when enabled, in addition to feature statistics, raw features (packet lengths and IATs) of first "
             "PACKET_LIMIT_PER_FLOW packets are exported, which are used by traffic augmenters/models.",
        default=False
    )

    args = parser.parse_args()

    flow_df = parse_features_to_dataframe(args.pcapfile, args.raw)
    output_csv = args.output if args.output else _get_output_csv_filename(args)
    flow_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    main()
