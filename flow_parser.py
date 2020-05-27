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
    'found_tcp_flags', 'bulk0', 'bulk1', 'client_packet0', 'client_packet1', 'tcp_window_avg',
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


def _calc_unidirectional_features(direction_slice, prefix='') -> dict:
    def _item_getter(matrix, row_indexer, column_indexer):
        try:
            return matrix[row_indexer, column_indexer]
        except IndexError:
            return 0

    # this asserts using of the listed features
    features = _create_empty_features(prefix)
    features[f'{prefix}found_tcp_flags'] = sorted(set(direction_slice[:, RMI.TCP_FLAGS]))

    features[f'{prefix}bulk0'] = _item_getter(direction_slice, 0, RMI.TRANSP_PAYLOAD)
    features[f'{prefix}bulk1'] = _item_getter(direction_slice, 1, RMI.TRANSP_PAYLOAD)

    features[f'{prefix}client_packet0'] = _item_getter(direction_slice, 0, RMI.IP_LEN)
    features[f'{prefix}client_packet1'] = _item_getter(direction_slice, 1, RMI.IP_LEN)

    features[f'{prefix}tcp_window_avg'] = np.mean(direction_slice[:, RMI.TCP_WINDOW])

    features[f'{prefix}bulk_max'] = np.max(direction_slice[:, RMI.TRANSP_PAYLOAD])
    features[f'{prefix}bulk_min'] = np.min(direction_slice[:, RMI.TRANSP_PAYLOAD])
    features[f'{prefix}bulk_avg'] = np.mean(direction_slice[:, RMI.TRANSP_PAYLOAD])
    features[f'{prefix}bulk_median'] = np.median(direction_slice[:, RMI.TRANSP_PAYLOAD])
    features[f'{prefix}bulk_25q'] = np.percentile(direction_slice[:, RMI.TRANSP_PAYLOAD], 25)
    features[f'{prefix}bulk_75q'] = np.percentile(direction_slice[:, RMI.TRANSP_PAYLOAD], 75)
    features[f'{prefix}bulk_bytes'] = np.sum(direction_slice[:, RMI.TRANSP_PAYLOAD])
    # counting non-empty packets (with payload)
    features[f'{prefix}bulk_number'] = direction_slice[direction_slice[:, RMI.TRANSP_PAYLOAD] > 0].shape[0]

    features[f'{prefix}packet_max'] = np.max(direction_slice[:, RMI.IP_LEN])
    features[f'{prefix}packet_min'] = np.min(direction_slice[:, RMI.IP_LEN])
    features[f'{prefix}packet_avg'] = np.mean(direction_slice[:, RMI.IP_LEN])
    features[f'{prefix}packet_median'] = np.median(direction_slice[:, RMI.IP_LEN])
    features[f'{prefix}packet_25q'] = np.percentile(direction_slice[:, RMI.IP_LEN], 25)
    features[f'{prefix}packet_75q'] = np.percentile(direction_slice[:, RMI.IP_LEN], 75)
    features[f'{prefix}packet_bytes'] = np.sum(direction_slice[:, RMI.IP_LEN])
    features[f'{prefix}packet_number'] = direction_slice[:, RMI.IP_LEN].shape[0]
    return features


def calc_flow_features(raw_features: np.ndarray) -> dict:
    client_slice = raw_features[raw_features[:, RMI.IS_CLIENT] == 1]
    if client_slice.shape[0] > 0:
        client_features = _calc_unidirectional_features(client_slice, prefix=FEATURE_PREFIX.client)
    else:
        client_features = _create_empty_features(prefix=FEATURE_PREFIX.client)

    server_slice = raw_features[raw_features[:, RMI.IS_CLIENT] == 0]
    if server_slice.shape[0] > 0:
        server_features = _calc_unidirectional_features(server_slice, prefix=FEATURE_PREFIX.server)
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


def flow_processor(source):
    streamer = nfstream.NFStreamer(source=source,
                                   statistics=False,
                                   idle_timeout=60,
                                   active_timeout=10e5,
                                   plugins=[raw_packets_matrix(volatile=False)],
                                   enable_guess=True)
    for flow_number, entry in enumerate(streamer):
        label_features = {
            'flow_id': f'{settings.IP_PROTO_MAPPING[entry.protocol]} '
                       f'{entry.src_ip}:{entry.src_port} '
                       f'{entry.dst_ip}:{entry.dst_port}',
            'ndpi_app': entry.application_name,
            'ndpi_category': entry.category_name,
            'ndpi_client_info': entry.client_info,
            'ndpi_server_info': entry.server_info,
            'ndpi_j3ac': entry.j3a_client,
            'ndpi_j3as': entry.j3a_server,
            'ip_proto': settings.IP_PROTO_MAPPING[entry.protocol],
        }
        flow_features = calc_flow_features(entry.raw_packets_matrix)

        if flow_number > 0 == flow_number % 1000:
            logger.info(f'processed {flow_number} flows...')

        yield dict(**label_features, **flow_features)


def parse_to_dataframe(pcap_file: str) -> pd.DataFrame:
    flows = []
    logger.info(f'started parsing file {pcap_file}')
    for flow in flow_processor(pcap_file):
        flows.append(flow)

    return pd.DataFrame(flows)


def _get_output_csv_filename(pcap_filename) -> str:
    core_name = pcap_filename.split('/')[-1].split('.')[0]
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

    args = parser.parse_args()

    flow_df = parse_to_dataframe(args.pcapfile)
    output_csv = args.output if args.output else _get_output_csv_filename(args.pcapfile)
    flow_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    main()
