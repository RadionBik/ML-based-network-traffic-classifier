import argparse
import logging

import dpkt
import pandas as pd
import nfstream
import numpy as np

import settings
from feature_processing import calc_raw_features, calc_flow_features, RMI

logger = logging.getLogger('flow_parser')


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
