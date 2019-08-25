#!/usr/bin/env python

import argparse
import configparser
import logging
import os
import re
import socket
from subprocess import Popen, PIPE
import typing
import dpkt
import pandas as pd
import numpy as np


logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


class UnknownProtocol(Exception):
    pass


class Endpoint(typing.NamedTuple):
    address: str
    port: int


class Connection(typing.NamedTuple):
    proto: str
    peers: typing.FrozenSet[Endpoint]


def ip_to_string(inet) -> str:
    """
    Convert inet object to a string
    :param inet: (inet struct): inet network address
    :return: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def ip4_from_string(ip: str) -> bytes:
    """
    Convert symbolic IP-address into a 4-byte string
    :param ip: IP-address as a string (e.g.: '10.0.0.1')
    :return: a 4-byte string
    """
    return bytes(map(int, ip.split('.')))


def get_percentile(parameter, percentile):
    return np.percentile(parameter, percentile) if len(parameter) > 0 else 0


def _extract_rawflow_features(df: pd.DataFrame) -> dict:

    client_indexes = df['is_client'] == 1
    server_indexes = df['is_client'] == 0

    client_bulks = df[(df['transp_payload'] > 0) & client_indexes]['transp_payload']

    server_bulks = df[(df['transp_payload'] > 0) & server_indexes]['transp_payload']

    client_packets = df[client_indexes]['ip_payload']

    server_packets = df[server_indexes]['ip_payload']

    client_ts = df[client_indexes].index
    iat_client = pd.to_timedelta(pd.Series(client_ts).diff().fillna('0')) / pd.offsets.Second(1)
    iat_client.index = client_ts

    server_ts = df[server_indexes].index
    iat_server = pd.to_timedelta(pd.Series(server_ts).diff().fillna('0')) / pd.offsets.Second(1)
    iat_server.index = server_ts

    df['IAT'] = pd.concat([iat_server, iat_client], ignore_index=True)
    server_iats = df[server_indexes]['IAT']
    client_iats = df[client_indexes]['IAT']

    fault_avoider = (
        lambda values, index=0: values.iloc[index] if len(values) > index else 0)

    stats = {
        'proto': df['proto'].iloc[0],
        'is_tcp': df['is_tcp'].iloc[0],

        'client_found_tcp_flags': sorted(set(df[client_indexes]['tcp_flags'])),
        'server_found_tcp_flags': sorted(set(df[server_indexes]['tcp_flags'])),

        'client_tcp_window_mean': df[client_indexes]['tcp_win'].mean(),
        'server_tcp_window_mean': df[server_indexes]['tcp_win'].mean(),

        'client_bulk0': fault_avoider(client_bulks, 0),
        'client_bulk1': fault_avoider(client_bulks, 1),
        'server_bulk0': fault_avoider(server_bulks, 0),
        'server_bulk1': fault_avoider(server_bulks, 1),

        'client_packet0': fault_avoider(client_packets, 0),
        'client_packet1': fault_avoider(client_packets, 1),
        'server_packet0': fault_avoider(server_packets, 0),
        'server_packet1': fault_avoider(server_packets, 1),

        'server_bulk_max': server_bulks.max(),
        'server_bulk_min': server_bulks.min(),
        'server_bulk_mean': server_bulks.mean(),
        'server_bulk_median': get_percentile(server_bulks, 50),
        'server_bulk_25q': get_percentile(server_bulks, 25),
        'server_bulk_75q': get_percentile(server_bulks, 75),
        'server_bulks_bytes': server_bulks.sum(),
        'server_bulks_number': len(server_bulks),

        'client_bulk_max': client_bulks.max(),
        'client_bulk_min': client_bulks.min(),
        'client_bulk_mean': client_bulks.mean(),
        'client_bulk_median': get_percentile(client_bulks, 50),
        'client_bulk_25q': get_percentile(client_bulks, 25),
        'client_bulk_75q': get_percentile(client_bulks, 75),
        'client_bulks_bytes': client_bulks.sum(),
        'client_bulks_number': len(client_bulks),

        'server_packet_max': server_packets.max(),
        'server_packet_min': server_packets.min(),
        'server_packet_mean': server_packets.mean(),
        'server_packet_median': get_percentile(server_packets, 50),
        'server_packet_25q': get_percentile(server_packets, 25),
        'server_packet_75q': get_percentile(server_packets, 75),
        'server_packets_bytes': server_packets.sum(),
        'server_packets_number': len(server_packets),

        'client_packet_max': client_packets.max(),
        'client_packet_min': client_packets.min(),
        'client_packet_mean': client_packets.mean(),
        'client_packet_median': get_percentile(client_packets, 50),
        'client_packet_25q': get_percentile(client_packets, 25),
        'client_packet_75q': get_percentile(client_packets, 75),
        'client_packets_bytes': client_packets.sum(),
        'client_packets_number': len(client_packets),

        'client_iat_mean': client_iats.mean(),
        'client_iat_median': get_percentile(client_iats, 50),
        'client_iat_25q': get_percentile(client_iats, 25),
        'client_iat_75q': get_percentile(client_iats, 75),

        'server_iat_mean': server_iats.mean(),
        'server_iat_median': get_percentile(server_iats, 50),
        'server_iat_25q': get_percentile(server_iats, 25),
        'server_iat_75q': get_percentile(server_iats, 75),
    }

    return stats


def _raw_flow_to_df(flow):
    df = pd.DataFrame(flow, columns=['TS', 'ip_payload', 'transp_payload', 'tcp_flags',
                                     'tcp_win', 'is_tcp', 'is_client', 'proto'])
    df.set_index(pd.to_datetime(df['TS'], unit='s'), inplace=True)
    return df.drop(['TS'], axis=1)


def _pure_filename(full_path):
    basename = os.path.basename(full_path)
    return os.path.splitext(basename)[0]


PROTOCOL = r'(UDP|TCP)'
IP4 = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
PORT = r'(\d{1,5})'
APPPROTO = r'\[proto: [\d+\.]*\d+\/(\w+\.?\w+)*\]'


def _get_ndpi_protocol_mapping(ndpi_filename: str) -> dict:
    pipe = Popen([ndpi_filename,
                  '-h'],
                 stdout=PIPE,
                 universal_newlines=True)
    stdout, stderr = pipe.communicate()
    protocol_mapping = dict()
    for line in stdout.split('\n'):
        if line.startswith('['):
            number_part, protocol_part = line.split(']')
            number = int(number_part.split('[')[1].strip())
            protocol = protocol_part.strip()
            protocol_mapping.update({number: protocol})
            protocol_mapping.update({protocol: number})

    return protocol_mapping


def _get_ndpi_output(ndpi_filename: str, pcap_filename: str) -> str:
    pipe = Popen([ndpi_filename,
                  '-i', pcap_filename, "-v2"],
                 stdout=PIPE,
                 universal_newlines=True)
    stdout, stderr = pipe.communicate()
    return stdout


def _parse_ndpi_output(raw: str, protocol_map: dict) -> dict:
    regex = f'{PROTOCOL} {IP4}:{PORT} <?->? {IP4}:{PORT} {APPPROTO}'
    reg = re.compile(regex)

    apps = {}
    for captures in re.findall(reg, raw):
        transp_proto, ip1, port1, ip2, port2, app_proto = captures

        port1 = int(port1)
        port2 = int(port2)
        connection = Connection(transp_proto.lower(),
                                frozenset([Endpoint(ip1, port1), Endpoint(ip2, port2)]))
        apps[connection] = protocol_map[app_proto.split('.')[0]]
    return apps


def _filter_packets(source):
    for timestamp, raw in source:
        eth = dpkt.ethernet.Ethernet(raw)
        ip = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            continue
        seg = ip.data
        if isinstance(seg, (dpkt.tcp.TCP, dpkt.udp.UDP)):
            yield timestamp, ip, seg


def _get_raw_flows(apps: dict, filename: str, max_packets_per_flow: typing.Optional[int] = None) -> dict:
    """ transform packets to flows for each app """
    flows = dict.fromkeys(apps)
    packet_counter = dict.fromkeys(apps)
    client_tuple = dict.fromkeys(apps.keys())
    with open(filename, "rb") as pcap_file:
        for pkt_number, (timestamp, ip, seg) in enumerate(_filter_packets(dpkt.pcap.Reader(pcap_file))):
            if isinstance(seg, dpkt.tcp.TCP):
                transp_proto = "tcp"
            elif isinstance(seg, dpkt.udp.UDP):
                transp_proto = "udp"
            else:
                raise UnknownProtocol(seg.__class__.__name__)
            source = Endpoint(ip_to_string(ip.src), seg.sport)
            destination = Endpoint(ip_to_string(ip.dst), seg.dport)
            connection = Connection(transp_proto, frozenset([source, destination]))

            assert connection in client_tuple

            # if client tuple is empty, then no packets from the flow has been seen so far
            if not client_tuple[connection]:
                client_tuple[connection] = source
                flows[connection] = np.zeros((max_packets_per_flow, 8))
                packet_counter[connection] = 0

            assert client_tuple[connection] in (source, destination)

            if max_packets_per_flow and packet_counter[connection] >= max_packets_per_flow:
                continue

            # 'TS', 'ip_payload', 'transp_payload', 'tcp_flags', 'tcp_win', 'is_tcp', 'is_client', 'proto'
            flows[connection][packet_counter[connection], 0] = timestamp
            flows[connection][packet_counter[connection], 1] = len(ip.data)
            flows[connection][packet_counter[connection], 2] = len(seg.data)
            flows[connection][packet_counter[connection], 3] = seg.flags if transp_proto == 'tcp' else 0
            flows[connection][packet_counter[connection], 4] = seg.win if transp_proto == 'tcp' else 0
            flows[connection][packet_counter[connection], 5] = transp_proto == 'tcp'
            flows[connection][packet_counter[connection], 6] = client_tuple[connection] == source
            flows[connection][packet_counter[connection], 7] = apps[connection]

            packet_counter[connection] += 1

    return flows


def _format_connection(connection: Connection) -> str:
    peers = list(sorted(connection.peers))
    return '{} {}:{} {}:{}'.format(connection.proto.upper(),
                                   peers[0].address, peers[0].port,
                                   peers[1].address, peers[1].port)


def _get_labeled_flows(ndpi_filename, traffic_filename, max_packets_per_flow=None):
    logging.info('Started extracting ground truth labels for flows...')
    output = _get_ndpi_output(ndpi_filename, traffic_filename)
    protocol_map = _get_ndpi_protocol_mapping(ndpi_filename)
    apps = _parse_ndpi_output(output, protocol_map)
    logging.info('Got {} unique flows!'.format(len(apps)))
    raw_flows = _get_raw_flows(apps, traffic_filename, max_packets_per_flow=max_packets_per_flow)
    return raw_flows


def _get_flows_features(raw_flows: dict) -> pd.DataFrame:
    flows = {}
    logging.info('Started extracting features of packets...')
    for flow_counter, (connection, flow) in enumerate(raw_flows.items()):
        key = _format_connection(connection)
        chronological_df = _raw_flow_to_df(flow)
        flow_features = _extract_rawflow_features(chronological_df)
        flows[key] = flow_features
        if flow_counter % 100 == 0:
            logging.info(f'Processed {flow_counter} flows...')
    else:
        logging.info(f'Processed {len(flows)} flows total')
    return pd.DataFrame(flows).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pcapfiles",
        nargs="+",
        help="pcap file")

    parser.add_argument(
        "-c", "--config",
        help="configuration file, defaults to config.ini",
        default='config.ini')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    max_packets_per_flow = int(config['parser']['packetLimitPerFlow'])
    pcap_filenames = args.pcapfiles or [config['parser']['PCAPfilename']]
    for pcap_filename in pcap_filenames:
        flows = _get_labeled_flows(
            config['parser']['nDPIfilename'],
            pcap_filename,
            max_packets_per_flow=max_packets_per_flow
        )
        features = _get_flows_features(flows)

        csv_output_folder = config['offline']['csv_folder']
        pure_filename = _pure_filename(pcap_filename)
        csv_output_filename = os.path.join(
            csv_output_folder,
            'flows_{}split_{}.csv'.format(
                max_packets_per_flow,
                pure_filename)
        )
        logging.info('Saving features to {}...'.format(csv_output_filename))
        protocol_mapping = _get_ndpi_protocol_mapping(config['parser']['nDPIfilename'])
        features['proto'] = features['proto'].apply(lambda proto: protocol_mapping[proto])
        features.to_csv(csv_output_filename, index=True, sep='|', na_rep=0, columns=sorted(features.columns))


if __name__ == "__main__":
    main()
