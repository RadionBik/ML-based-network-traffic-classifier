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


def _extract_rawflow_features(df: pd.DataFrame) -> dict:

    client_bulks = df[(df['transp_payload'] > 0) &
                      (df['is_client'] == 1)
                      ]['transp_payload']

    server_bulks = df[(df['transp_payload'] > 0) &
                      (df['is_client'] == 0)
                      ]['transp_payload']

    client_packets = df[df['is_client'] == 1
                        ]['ip_payload']

    server_packets = df[df['is_client'] == 0
                        ]['ip_payload']

    client_index = df[df['is_client'] == 1].index
    iat_client = pd.to_timedelta(pd.Series(client_index).diff().fillna('0')) / pd.offsets.Second(1)
    iat_client.index = client_index

    server_index = df[df['is_client'] == 0].index

    iat_server = pd.to_timedelta(pd.Series(server_index).diff().fillna('0')) / pd.offsets.Second(1)
    iat_server.index = server_index

    df['IAT'] = pd.concat([iat_server, iat_client])

    server_iats = df[df['is_client'] == 0]['IAT']
    client_iats = df[df['is_client'] == 1]['IAT']

    fault_avoider = (
        lambda values, index=0: values.iloc[index] if len(values) > index else 0)

    stats = {
        'proto': df['proto'].iloc[0],
        'subproto': df['subproto'].iloc[0],
        'is_tcp': df['is_tcp'].iloc[0],

        'client_found_tcp_flags': sorted(set(df[df['is_client'] == 1]['tcp_flags'])),
        'server_found_tcp_flags': sorted(set(df[df['is_client'] == 0]['tcp_flags'])),

        'client_tcp_window_mean': df[df['is_client'] == 1]['tcp_win'].mean(),
        'server_tcp_window_mean': df[df['is_client'] == 0]['tcp_win'].mean(),

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
        'server_bulk_median': server_bulks.quantile(.5),
        'server_bulk_25q': server_bulks.quantile(.25),
        'server_bulk_75q': server_bulks.quantile(.75),
        'server_bulks_bytes': server_bulks.sum(),
        'server_bulks_number': len(server_bulks),

        'client_bulk_max': client_bulks.max(),
        'client_bulk_min': client_bulks.min(),
        'client_bulk_mean': client_bulks.mean(),
        'client_bulk_median': client_bulks.quantile(.5),
        'client_bulk_25q': client_bulks.quantile(.25),
        'client_bulk_75q': client_bulks.quantile(.75),
        'client_bulks_bytes': client_bulks.sum(),
        'client_bulks_number': len(client_bulks),

        'server_packet_max': server_packets.max(),
        'server_packet_min': server_packets.min(),
        'server_packet_mean': server_packets.mean(),
        'server_packet_median': server_packets.quantile(.5),
        'server_packet_25q': server_packets.quantile(.25),
        'server_packet_75q': server_packets.quantile(.75),
        'server_packets_bytes': server_packets.sum(),
        'server_packets_number': len(server_packets),

        'client_packet_max': client_packets.max(),
        'client_packet_min': client_packets.min(),
        'client_packet_mean': client_packets.mean(),
        'client_packet_median': client_packets.quantile(.5),
        'client_packet_25q': client_packets.quantile(.25),
        'client_packet_75q': client_packets.quantile(.75),
        'client_packets_bytes': client_packets.sum(),
        'client_packets_number': len(client_packets),

        'client_iat_mean': client_iats.mean(),
        'client_iat_median': client_iats.quantile(.5),
        'client_iat_25q': client_iats.quantile(.25),
        'client_iat_75q': client_iats.quantile(.75),

        'server_iat_mean': server_iats.mean(),
        'server_iat_median': server_iats.quantile(.5),
        'server_iat_25q': server_iats.quantile(.25),
        'server_iat_75q': server_iats.quantile(.75),
    }

    return stats


def _raw_flow_to_df(flow):
    df = pd.DataFrame(flow)
    df.set_index(pd.to_datetime(df['TS'], unit='s'), inplace=True)
    return df.drop(['TS'], axis=1)


def _pure_filename(full_path):
    basename = os.path.basename(full_path)
    return os.path.splitext(basename)[0]


PROTOCOL = r'(UDP|TCP)'
IP4 = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
PORT = r'(\d{1,5})'
APPPROTO = r'\[proto: [\d+\.]*\d+\/(\w+\.?\w+)*\]'


def _parse_ndpi_output(raw: str) -> dict:
    regex = f'{PROTOCOL} {IP4}:{PORT} <?->? {IP4}:{PORT} {APPPROTO}'
    reg = re.compile(regex)

    apps = {}
    for captures in re.findall(reg, raw):
        transp_proto, ip1, port1, ip2, port2, app_proto = captures

        port1 = int(port1)
        port2 = int(port2)
        connection = Connection(transp_proto.lower(),
                                frozenset([Endpoint(ip1, port1), Endpoint(ip2, port2)]))
        apps[connection] = app_proto
    return apps


def _get_flow_apps(ndpi_filename: str, pcap_filename: str) -> dict:
    pipe = Popen(['./' + ndpi_filename,
                  '-i', pcap_filename, "-v2"],
                 stdout=PIPE,
                 universal_newlines=True)
    stdout, stderr = pipe.communicate()
    return _parse_ndpi_output(stdout)


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
    client_tuple = dict.fromkeys(apps.keys())
    with open(filename, "rb") as pcap_file:
        for timestamp, ip, seg in _filter_packets(dpkt.pcap.Reader(pcap_file)):
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
                flows[connection] = []

            assert client_tuple[connection] in (source, destination)

            flow = flows[connection]
            if max_packets_per_flow and len(flow) > max_packets_per_flow:
                continue

            app = apps[connection].split('.')

            packet = {
                'TS': timestamp,
                'ip_payload': len(ip.data),
                'transp_payload': len(seg.data),
                'proto': app[0],
                'subproto': app[1] if len(app) > 1 else '',
                'tcp_flags': seg.flags if transp_proto == 'tcp' else 0,
                'tcp_win': seg.win if transp_proto == 'tcp' else 0,
                'is_tcp': transp_proto == 'tcp',
                'is_client': client_tuple[connection] == source
            }

            flows[connection].append(packet)
    return flows


def _format_connection(connection: Connection) -> str:
    peers = list(connection.peers)
    return '{} {}:{} {}:{}'.format(connection.proto.upper(),
                                   peers[0].address, peers[0].port,
                                   peers[1].address, peers[1].port)


def _get_flows_features(ndpi_filename: str,
                        traffic_filename: str,
                        max_packets_per_flow: typing.Optional[int] = None) -> pd.DataFrame:
    logging.info('Started extracting ground truth labels for flows...')
    apps = _get_flow_apps(ndpi_filename, traffic_filename)
    logging.info('Got {} unique flows!'.format(len(apps)))
    flows = {}
    logging.info('Started extracting features of packets...')
    raw_flows = _get_raw_flows(apps, traffic_filename, max_packets_per_flow=max_packets_per_flow)
    for flow_counter, (connection, flow) in enumerate(raw_flows.items()):
        key = _format_connection(connection)
        chronoligical_df = _raw_flow_to_df(flow)
        flow_features = _extract_rawflow_features(chronoligical_df)
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
        features = _get_flows_features(
            config['parser']['nDPIfilename'],
            pcap_filename,
            max_packets_per_flow=max_packets_per_flow
        )

        csv_output_folder = config['offline']['csv_folder']
        pure_filename = _pure_filename(pcap_filename)
        csv_output_filename = os.path.join(
            csv_output_folder,
            'flows_{}split_{}.csv'.format(
                max_packets_per_flow,
                pure_filename)
        )
        logging.info('Saving features to {}...'.format(csv_output_filename))
        features.to_csv(csv_output_filename, index=True, sep='|')


if __name__ == "__main__":
    main()
