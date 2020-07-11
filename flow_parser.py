import argparse
import csv
import logging

import dpkt
import pandas as pd
import nfstream
import numpy as np
from typing import Optional

import settings
from feature_processing import calc_raw_features, calc_flow_features, RMI

logger = logging.getLogger('flow_parser')


class raw_packets_matrix(nfstream.NFPlugin):
    def __init__(self, volatile=False, packet_limit=None, raw_matrix_indexer=RMI):
        super().__init__(volatile=volatile, user_data=None)
        self.packet_limit = packet_limit if packet_limit else settings.PACKET_LIMIT_PER_FLOW
        self.rmi = raw_matrix_indexer

    def _fill_flow_stats(self, obs, raw_feature_matrix, counter=0):
        raw_feature_matrix[counter, self.rmi.TIMESTAMP] = obs.time
        raw_feature_matrix[counter, self.rmi.IP_LEN] = obs.ip_size
        raw_feature_matrix[counter, self.rmi.TRANSP_PAYLOAD] = obs.payload_size
        raw_feature_matrix[counter, self.rmi.TCP_FLAGS] = int(''.join(str(i) for i in obs.tcpflags), 2)
        if obs.protocol == 6 and obs.version == 4:
            packet = dpkt.ip.IP(obs.ip_packet)
            try:
                raw_feature_matrix[counter, self.rmi.TCP_WINDOW] = packet.data.win
            except AttributeError:
                logger.warning(f'unexpected packet format: {packet}')
                raw_feature_matrix[counter, self.rmi.TCP_WINDOW] = 0
        raw_feature_matrix[counter, self.rmi.IP_PROTO] = obs.protocol
        raw_feature_matrix[counter, self.rmi.IS_CLIENT] = 1 if obs.direction == 0 else 0
        return raw_feature_matrix

    def on_init(self, obs):
        raw_feature_matrix = np.zeros((self.packet_limit, 7))
        return self._fill_flow_stats(obs, raw_feature_matrix)

    def on_update(self, obs, entry):
        if entry.bidirectional_packets > self.packet_limit:
            return entry.raw_packets_matrix
        return self._fill_flow_stats(obs, entry.raw_packets_matrix, counter=entry.bidirectional_packets - 1)

    def on_expire(self, entry):
        # rm unfilled matrix rows
        entry.raw_packets_matrix = entry.raw_packets_matrix[entry.raw_packets_matrix[:, self.rmi.TIMESTAMP] > 0]


def init_streamer(source, plugins: list, online_mode=False, provide_labels=True):
    # since we decide and set routing policy upon first occurrence of a flow we don't care about its re-export
    active_timeout = settings.ACTIVE_TIMEOUT_ONLINE if online_mode else settings.ACTIVE_TIMEOUT_OFFLINE
    logger.info(f'mode set to {"online" if online_mode else "offline"}')

    return nfstream.NFStreamer(source=source,
                               statistics=False,
                               idle_timeout=settings.IDLE_TIMEOUT,
                               active_timeout=active_timeout,
                               plugins=plugins,
                               enable_guess=provide_labels,
                               dissect=provide_labels)


def flow_processor(source,
                   derivative_features: bool = True,
                   raw_features: Optional[int] = None,
                   provide_labels=True,
                   online_mode=True
                   ) -> dict:

    def _make_flow_id(entry):
        return f'{settings.IP_PROTO_MAPPING[entry.protocol]} ' \
               f'{entry.src_ip}:{entry.src_port} ' \
               f'{entry.dst_ip}:{entry.dst_port}'

    for flow_number, entry in enumerate(init_streamer(source,
                                                      plugins=[raw_packets_matrix(volatile=False)],
                                                      provide_labels=provide_labels,
                                                      online_mode=online_mode)):
        flow_ids = {
            'flow_id': _make_flow_id(entry),
            'ip_proto': settings.IP_PROTO_MAPPING[entry.protocol]}

        ndpi_features = {
            'ndpi_app': entry.application_name,
            'ndpi_category': entry.category_name,
            'ndpi_client_info': entry.client_info,
            'ndpi_server_info': entry.server_info,
            'ndpi_j3ac': entry.j3a_client,
            'ndpi_j3as': entry.j3a_server,
        } if provide_labels else {}

        flow_features = calc_flow_features(entry.raw_packets_matrix) if derivative_features else {}

        raw_packets = calc_raw_features(entry.raw_packets_matrix) if raw_features else {}

        if flow_number > 0 == flow_number % 5000:
            logger.info(f'processed {flow_number} flows...')
        yield dict(**flow_ids, **ndpi_features, **flow_features, **raw_packets)


def parse_features_to_csv(pcap_file_path,
                          target_csv_path,
                          derivative_features: bool = True,
                          raw_features: Optional[int] = None,
                          provide_labels=True):

    logger.info(f'started parsing file {pcap_file_path}')
    with open(target_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for index, flow in enumerate(flow_processor(pcap_file_path,
                                                    derivative_features=derivative_features,
                                                    raw_features=raw_features,
                                                    provide_labels=provide_labels)):
            if index == 0:
                writer.writerow(flow.keys())
            writer.writerow(flow.values())


def parse_features_to_dataframe(pcap_file: str,
                                derivative_features: bool = True,
                                raw_features: bool = False,
                                provide_labels=True,
                                online_mode=True) -> pd.DataFrame:
    flows = []
    logger.info(f'started parsing file {pcap_file}')
    for flow in flow_processor(pcap_file,
                               derivative_features=derivative_features,
                               raw_features=raw_features,
                               provide_labels=provide_labels,
                               online_mode=online_mode):
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
        type=int,
        help="when provided, in addition to feature statistics, specified N number of raw features "
             "(packet lengths and IATs) for first N packets are exported, which are used by traffic augmenters/models.",
        default=None
    )
    parser.add_argument('--derivative', dest='derivative', action='store_true',
                        help="when enabled, derivative feature statistics "
                             "(e.g. such as percentiles, sums, etc. of packet size) "
                             "of first PACKET_LIMIT_PER_FLOW or provided via arg '--raw' packets are exported")
    parser.add_argument('--no-derivative', dest='derivative', action='store_false')
    parser.set_defaults(derivative=True)

    args = parser.parse_args()

    output_csv = args.output if args.output else _get_output_csv_filename(args)
    parse_features_to_csv(args.pcapfile, output_csv, args.derivative, args.raw)


if __name__ == '__main__':
    main()
