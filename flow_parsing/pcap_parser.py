import argparse
import csv
import logging

import pandas as pd
import nfstream
from typing import Optional

import settings
from flow_parsing.features import calc_raw_features, calc_stat_features
from flow_parsing.raw_packets_nfplugin import raw_packets_matrix

logger = logging.getLogger('flow_parser')


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


def get_ip_protocol_by_int(proto: int) -> str:
    try:
        return settings.IP_PROTO_MAPPING[proto]
    except KeyError:
        logger.warning(f'encountered unknown IP proto number: {proto}')
        return 'UNKNOWN'


def flow_processor(source,
                   derivative_features: bool = True,
                   raw_features: Optional[int] = None,
                   provide_labels=True,
                   online_mode=True
                   ) -> dict:

    def _make_flow_id():
        return f'{get_ip_protocol_by_int(entry.protocol)} ' \
               f'{entry.src_ip}:{entry.src_port} ' \
               f'{entry.dst_ip}:{entry.dst_port}'

    packet_limit = raw_features if raw_features else settings.DEFAULT_PACKET_LIMIT_PER_FLOW

    for flow_number, entry in enumerate(init_streamer(source,
                                                      plugins=[raw_packets_matrix(volatile=False,
                                                                                  packet_limit=packet_limit)],
                                                      provide_labels=provide_labels,
                                                      online_mode=online_mode)):
        flow_ids = {
            'flow_id': _make_flow_id(),
            'ip_proto': get_ip_protocol_by_int(entry.protocol)}

        ndpi_features = {
            'ndpi_app': entry.application_name,
            'ndpi_category': entry.category_name,
            'ndpi_client_info': entry.client_info,
            'ndpi_server_info': entry.server_info,
            'ndpi_j3ac': entry.j3a_client,
            'ndpi_j3as': entry.j3a_server,
        } if provide_labels else {}

        flow_features = calc_stat_features(entry.raw_packets_matrix) if derivative_features else {}

        raw_packets = calc_raw_features(entry.raw_packets_matrix, packet_limit) if raw_features else {}

        if flow_number > 0 == flow_number % 5000:
            logger.info(f'processed {flow_number} flows...')
        yield dict(**flow_ids, **ndpi_features, **flow_features, **raw_packets)


def parse_pcap_to_csv(pcap_file_path,
                      target_csv_path,
                      derivative_features: bool = True,
                      raw_features: Optional[int] = None,
                      provide_labels=True,
                      online_mode=True):

    logger.info(f'started parsing file {pcap_file_path}')
    logger.info(f'saving to {target_csv_path}')
    with open(target_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for index, flow in enumerate(flow_processor(pcap_file_path,
                                                    derivative_features=derivative_features,
                                                    raw_features=raw_features,
                                                    provide_labels=provide_labels,
                                                    online_mode=online_mode)):
            if index == 0:
                writer.writerow(flow.keys())
            writer.writerow(flow.values())


def parse_pcap_to_dataframe(pcap_file: str,
                            derivative_features: bool = True,
                            raw_features: Optional[int] = None,
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
    pkt_lim = args.raw if args.raw else settings.DEFAULT_PACKET_LIMIT_PER_FLOW
    output_csv = settings.PCAP_OUTPUT_DIR / f'{core_name}_{pkt_lim}packets.csv'
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pcapfile",
        help="pcap file",
        default=(settings.BASE_DIR / 'flow_parsing/static/example.pcap').as_posix(),
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
                             "of first DEFAULT_PACKET_LIMIT_PER_FLOW or provided via arg '--raw' packets are exported")
    parser.add_argument('--no-derivative', dest='derivative', action='store_false')
    parser.set_defaults(derivative=True)

    parser.add_argument('--online-mode', dest='online_mode', action='store_true',
                        help="when enabled, active flow expiration timeout is decreased to the one defined in settings,"
                             "to suite online monitoring, alternatively (default), active timeout is set to be large "
                             "enough to avoid flow fragmentation due to the timeout",
                        default=False)

    args = parser.parse_args()

    output_csv = args.output if args.output else _get_output_csv_filename(args)
    parse_pcap_to_csv(args.pcapfile,
                      target_csv_path=output_csv,
                      derivative_features=args.derivative,
                      raw_features=args.raw,
                      online_mode=args.online_mode)


if __name__ == '__main__':
    main()
