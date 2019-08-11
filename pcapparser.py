#!/usr/bin/env python

import argparse
import configparser
import os
import re
import socket
from subprocess import Popen, PIPE

import dpkt
import pandas as pd

from feature_processing import Feature_Extractor


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


class PCAPParser(Feature_Extractor):

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

    def _get_raw_flow_df(self, flow):

        df = pd.DataFrame(flow)
        df.set_index(pd.to_datetime(df['TS'], unit='s'), inplace=True)
        return df.drop(['TS'], axis=1)

    def get_flows_features(self, save_to_file=False):
        print('Started extracting ground truth labels for flows...')
        self._apps = self._get_flow_labels()
        print('Got {} unique flows!'.format(len(self._apps)))

        flow_counter = 0
        print('Started extracting features of packets...')
        raw_flows = self._get_raw_flows()
        for key in raw_flows:
            raw_df = self._get_raw_flow_df(raw_flows[key])
            # format with a nice key
            key = '{} {}:{} {}:{}'.format(key[0].upper(),
                                          *list(key[1])[0],
                                          *list(key[1])[1])

            self._flows.update({key:
                                    self.extract_features(raw_df)})
            flow_counter += 1
            if flow_counter % 100 == 0:
                print('Processed {} flows...'.format(flow_counter))

        self.flow_features = pd.DataFrame(self._flows).T
        if save_to_file:
            self.csv_filename = '{}{}flows_{}split_{}.csv'.format(
                self._config['offline']['csv_folder'],
                os.sep,
                self.strip,
                self.traffic_filename.split(os.sep)[-1].split('.')[0])

            print('Saving features to {}...'.format(self.csv_filename))
            self.flow_features.to_csv(self.csv_filename, index=True, sep='|')

        return self.flow_features


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
        parsed_pcap.get_flows_features(save_to_file=True)


if __name__ == "__main__":
    main()
