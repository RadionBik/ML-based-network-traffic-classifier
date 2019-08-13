#!/usr/bin/env python

import argparse
import configparser
import os
import re
import socket
from subprocess import Popen, PIPE

import dpkt
import pandas as pd


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


def _extract_feature_stats(raw_df):
    stats = {}

    client_bulks = raw_df[(raw_df['transp_payload'] > 0) &
                          (raw_df['is_client'] == 1)
                          ]['transp_payload']

    server_bulks = raw_df[(raw_df['transp_payload'] > 0) &
                          (raw_df['is_client'] == 0)
                          ]['transp_payload']

    client_packets = raw_df[raw_df['is_client'] == 0
                            ]['ip_payload']

    server_packets = raw_df[raw_df['is_client'] == 0
                            ]['ip_payload']

    fault_avoider = (
        lambda values, index=0: values.iloc[index] if len(values) > index else 0)

    stats.update({
        'proto': raw_df['proto'].iloc[0],
        'subproto': raw_df['subproto'].iloc[0],
        'is_tcp': raw_df['is_tcp'].iloc[0],

        'client_found_tcp_flags': sorted(set(raw_df[raw_df['is_client'] == 1]['tcp_flags'])),
        'server_found_tcp_flags': sorted(set(raw_df[raw_df['is_client'] == 0]['tcp_flags'])),

        'client_tcp_window_mean': raw_df[raw_df['is_client'] == 1]['tcp_win'].mean(),
        'server_tcp_window_mean': raw_df[raw_df['is_client'] == 0]['tcp_win'].mean(),

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
    })

    iat_client = pd.to_timedelta(pd.Series(
        raw_df[raw_df['is_client'] == 1].index).diff().fillna('0')) / pd.offsets.Second(1)
    iat_client.index = raw_df[raw_df['is_client'] == 1].index

    iat_server = pd.to_timedelta(pd.Series(
        raw_df[raw_df['is_client'] == 0].index).diff().fillna('0')) / pd.offsets.Second(1)
    iat_server.index = raw_df[raw_df['is_client'] == 0].index

    raw_df['IAT'] = pd.concat([iat_server, iat_client])

    client_iats = raw_df[raw_df['is_client'] == 1
                         ]['IAT']
    server_iats = raw_df[raw_df['is_client'] == 0
                         ]['IAT']

    stats.update({
        'client_iat_mean': client_iats.mean(),
        'client_iat_median': client_iats.quantile(.5),
        'client_iat_25q': client_iats.quantile(.25),
        'client_iat_75q': client_iats.quantile(.75),

        'server_iat_mean': server_iats.mean(),
        'server_iat_median': server_iats.quantile(.5),
        'server_iat_25q': server_iats.quantile(.25),
        'server_iat_75q': server_iats.quantile(.75),
    })

    return stats


def _get_raw_flow_df(flow):
    df = pd.DataFrame(flow)
    df.set_index(pd.to_datetime(df['TS'], unit='s'), inplace=True)
    return df.drop(['TS'], axis=1)


def _pure_filename(full_path):
    basename = os.path.basename(full_path)
    return os.path.splitext(basename)[0]


class PCAPParser:
    def __init__(self, config, traffic_filename=None):
        super().__init__()
        self._config = config
        self.traffic_filename = traffic_filename
        self.strip = int(self._config['parser']['packetLimitPerFlow'])
        self._apps = {}
        self._flows = {}
        self.flow_features = pd.DataFrame()
        self.csv_filename = ''

    def __repr__(self):
        if self.flow_features.shape[0] == 0:
            return '{}, strip={}, 0 flows with features so far.'.format(self.traffic_filename,
                                                                        self.strip)
        else:
            return '{}, strip={}, {} flows with features such as:\n{}'.format(self.traffic_filename,
                                                                              self.strip,
                                                                              self.flow_features.shape[0],
                                                                              self.flow_features.head())

    def _get_flow_labels(self):
        pipe = Popen(['./' + self._config['parser']['nDPIfilename'],
                      '-i', self.traffic_filename, "-v2"], stdout=PIPE)
        raw = pipe.communicate()[0].decode("utf-8")
        reg = re.compile(
            r'(UDP|TCP) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) <?->? (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) \[proto: [\d+\.]*\d+\/(\w+\.?\w+)*\]')

        apps = {}
        for captures in re.findall(reg, raw):
            transp_proto, ip1, port1, ip2, port2, app_proto = captures

            port1 = int(port1)
            port2 = int(port2)
            key = (transp_proto.lower(),
                   frozenset(((ip1, port1), (ip2, port2))))
            apps[key] = app_proto
        return apps

    def _get_raw_flows(self):
        flows = dict.fromkeys(self._apps.keys())
        client_tuple = dict.fromkeys(self._apps.keys())
        with open(self.traffic_filename, "rb") as pcap_file:
            for ts, raw in dpkt.pcap.Reader(pcap_file):
                eth = dpkt.ethernet.Ethernet(raw)

                ip = eth.data
                seg = ip.data

                # check if the packet is IP, TCP, UDP
                if not isinstance(ip, dpkt.ip.IP):
                    continue

                if isinstance(seg, dpkt.tcp.TCP):
                    # 2 and 18 correspond to active SYN and SYN&ACK flags
                    # if (seg.flags & dpkt.tcp.TH_SYN):
                    #    print(seg.flags)
                    transp_proto = "tcp"

                elif isinstance(seg, dpkt.udp.UDP):
                    transp_proto = "udp"
                else:
                    continue

                key = (transp_proto, frozenset(
                    ((ip_to_string(ip.src), seg.sport),
                     (ip_to_string(ip.dst), seg.dport))))

                assert key in client_tuple

                # if client tuple is empty, then no packets from the flow has been seen so far
                if not client_tuple[key]:
                    client_tuple[key] = (ip_to_string(ip.src), seg.sport)
                    flows[key] = {feature: [] for feature in ['is_client',
                                                              'TS',
                                                              'ip_payload',
                                                              'transp_payload',
                                                              'tcp_flags',
                                                              'tcp_win',
                                                              'proto',
                                                              'subproto',
                                                              'is_tcp',
                                                              ]}

                if self.strip != 0 and (len(flows[key]['TS'])) > self.strip:
                    continue

                flows[key]['TS'].append(ts)
                flows[key]['ip_payload'].append(len(ip.data))
                flows[key]['transp_payload'].append(len(seg.data))

                if client_tuple[key] == (ip_to_string(ip.src), seg.sport):
                    flows[key]['is_client'].append(True)
                elif client_tuple[key] == (ip_to_string(ip.dst), seg.dport):
                    flows[key]['is_client'].append(False)
                else:
                    raise ValueError

                if transp_proto == 'tcp':
                    flows[key]['tcp_flags'].append(seg.flags)
                    flows[key]['tcp_win'].append(seg.win)
                    flows[key]['is_tcp'].append(True)

                else:
                    flows[key]['tcp_flags'].append(0)
                    flows[key]['tcp_win'].append(0)
                    flows[key]['is_tcp'].append(False)

                app = self._apps[key].split('.')
                if len(app) == 1:
                    flows[key]['proto'].append(app[0])
                    flows[key]['subproto'].append('')
                else:
                    flows[key]['proto'].append(app[0])
                    flows[key]['subproto'].append(app[1])

        return flows

    def get_flows_features(self):
        print('Started extracting ground truth labels for flows...')
        self._apps = self._get_flow_labels()
        print('Got {} unique flows!'.format(len(self._apps)))

        flow_counter = 0
        print('Started extracting features of packets...')
        raw_flows = self._get_raw_flows()
        for key in raw_flows:
            raw_df = _get_raw_flow_df(raw_flows[key])
            # format with a nice key
            key = '{} {}:{} {}:{}'.format(key[0].upper(),
                                          *list(key[1])[0],
                                          *list(key[1])[1])

            self._flows.update({key: _extract_feature_stats(raw_df)})
            flow_counter += 1
            if flow_counter % 100 == 0:
                print('Processed {} flows...'.format(flow_counter))

        self.flow_features = pd.DataFrame(self._flows).T
        return self.flow_features

    def save_to_file(self):
        pure_filename = _pure_filename(self.traffic_filename)
        self.csv_filename = os.path.join(
            self._config['offline']['csv_folder'],
            'flows_{}split_{}.csv'.format(
                self.strip,
                pure_filename)
        )
        print('Saving features to {}...'.format(self.csv_filename))
        self.flow_features.to_csv(self.csv_filename, index=True, sep='|')


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
    pcap_files = args.pcapfiles or [config['parser']['PCAPfilename']]
    for pcap_file in pcap_files:
        parsed_pcap = PCAPParser(config, pcap_file)
        parsed_pcap.get_flows_features()
        parsed_pcap.save_to_file()


if __name__ == "__main__":
    main()
