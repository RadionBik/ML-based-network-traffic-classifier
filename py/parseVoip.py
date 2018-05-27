#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re
import argparse
from subprocess import Popen, PIPE
import cProfile
import dpkt
import pcap
import pandas as ps
import numpy as np
import sklearn
import socket
import copy
import matplotlib.pyplot as plt

#from pypacker import ppcap
#from pypacker.layer12 import ethernet
#from pypacker.layer3 import ip
#from pypacker.layer4 import tcp, udp


def ip_to_string(inet):
    """Convert inet object to a string
        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def ip_from_string(ips):
    '''
        Convert symbolic IP-address into a 4-byte string
        Args:
            ips - IP-address as a string (e.g.: '10.0.0.1')
        returns:
            a 4-byte string
    '''
    return b''.join([bytes([int(n)]) for n in ips.split('.')])


def parseVoIPpacker(pcapfile, filter):
    for ts, raw in ppcap.Reader(filename=pcapfile):
        eth = ethernet.Ethernet(raw)

        # create the keys for IP UDP/TCP flows
        if eth[ip.IP] is not None:
            # if eth[tcp.TCP] is not None:
            #    continue
                # key = ('tcp', frozenset(((eth.ip.src_s, eth.ip.tcp.sport),(eth.ip.dst_s, eth.ip.tcp.dport))))
            if eth[udp.UDP] is not None:
                # and ((eth.ip.udp.sport in portsOfInterest) or (eth.ip.udp.dport in portsOfInterest)):
                if ((eth.ip.src_s in filter['IP']) or (eth.ip.dst_s in filter['IP'])):
                    if ((eth.ip.udp.sport in filter['port']) or (eth.ip.udp.dport in filter['port'])):
                        key = (ts, 'udp', frozenset(
                            ((eth.ip.src_s, eth.ip.udp.sport), (eth.ip.dst_s, eth.ip.udp.dport))))
                        print(key)
        else:
            continue


def get_IAT(TS):

    iteration = 0
    IAT = [0]
    for ts in TS:
        if iteration == 0:
            tempIAT = ts
            iteration = iteration + 1
        else:
            IAT.append(ts - tempIAT)
            tempIAT = ts
    return IAT


def parseVoIP(pcapfile, filter=None, isTunInt=False, manualFiltering=False):

    pipe = Popen(["./ndpiReader", "-i", pcapfile, "-v2"], stdout=PIPE)
    raw = pipe.communicate()[0].decode("utf-8")

    reg = re.compile(
        r'(UDP) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) <?->? (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) \[proto: [\d+\.]*\d+\/(RTP)*\]')

    if not manualFiltering:

        filter = {
            'IPserver': [],
            'IPclient': [],
            'port': [],
            'length': []
        }

        print('Found the following RTP flows:')
        for captures in re.findall(reg, raw):
            print(captures)
            transp_proto, ip1, port1, ip2, port2, app_proto = captures
            filter['IPserver'].append(ip1)
            filter['IPclient'].append(ip2)
            filter['port'].append(int(port1))
            filter['port'].append(int(port2))

    clientToServerPkts = {
        'ts': [],
        'pktLen': [],
        'IAT': []
    }

    serverToClientPkts = copy.deepcopy(clientToServerPkts)

    pktNum = 0
    t = 1
    iteration = 0
    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        if (t == 1):
            firstTime = ts
            t = t-1
        if (isTunInt):
            ip = dpkt.ip.IP(raw)
        else:
            eth = dpkt.ethernet.Ethernet(raw)
            ip = eth.data
        # check if the packet is IP, TCP, UDP
        if not isinstance(ip, dpkt.ip.IP):
            continue
        seg = ip.data
        if isinstance(seg, dpkt.udp.UDP):
            timeFromStart = ts - firstTime
            if (ip_to_string(ip.src) in filter['IPserver']) or (ip_to_string(ip.dst) in filter['IPserver']):
                if ((ip.data.sport in filter['port']) and (ip.data.dport in filter['port'])) and len(raw) not in filter['length']:
                    if ip_to_string(ip.src) == filter['IPclient'][0]:
                        clientToServerPkts['ts'].append(timeFromStart)
                        clientToServerPkts['pktLen'].append(len(raw))

                    else:
                        serverToClientPkts['ts'].append(timeFromStart)
                        serverToClientPkts['pktLen'].append(len(raw))

                    pktNum = pktNum+1

        else:
            continue

    clientToServerPkts['IAT'] = get_IAT(clientToServerPkts['ts'])
    serverToClientPkts['IAT'] = get_IAT(serverToClientPkts['ts'])
    return clientToServerPkts, serverToClientPkts


def plot_IAT(list, title, fig_properties):
    f1 = plt.figure(figsize=fig_properties['size'])
    plt.hist(
        list, bins=fig_properties['bin_number'], range=fig_properties['range'])
    plt.title(title)
    plt.xlabel('IAT, s')
    plt.ylabel('number')
    plt.grid(True)
    f1.show()


def plot_PL(list, title, fig_properties):
    f2 = plt.figure(figsize=fig_properties['size'])
    plt.hist(list, bins=fig_properties['bin_number'])
    plt.title(title)
    plt.xlabel('bytes')
    plt.ylabel('number')
    plt.grid(True)
    f2.show()


def plotPSOfTime(dict, fig_properties, title):
    f3 = plt.figure(figsize=fig_properties['size'])
    plt.plot(dict['ts'][::fig_properties['sampling']],
             dict['pktLen'][::fig_properties['sampling']])
    plt.title(title)
    plt.ylabel('Packet size, bytes')
    plt.xlabel('time, s')
    plt.grid(True)
    f3.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="pcap file")
    args = parser.parse_args()

    filter = {
        'IPserver': ['192.168.0.105'],
        'IPclient': ['192.168.0.102'],
        'port': [26454, 18826],
        'length': []
    }

    isTunInt = False
    useManualFilter = True

    clientToServerPkts, serverToClientPkts = parseVoIP(
        args.file, filter, isTunInt, useManualFilter)

    fig_properties = {
        'size': (5, 4),
        'bin_number': 50,
        'range': (0, 0.1),
        'sampling': 1
    }

    plot_IAT(clientToServerPkts['IAT'],
             'Inter-Arrival Time Client -> Server', fig_properties)
    plot_IAT(serverToClientPkts['IAT'],
             'Inter-Arrival Time Server -> Client', fig_properties)
    plot_PL(clientToServerPkts['pktLen'],
            'Packet Length Client -> Server', fig_properties)
    plot_PL(serverToClientPkts['pktLen'],
            'Packet Length Server -> Client', fig_properties)

    plotPSOfTime(clientToServerPkts, fig_properties,
                 'PS of time Client -> Server')

    plotPSOfTime(serverToClientPkts, fig_properties,
                 'PS of time Server -> Client')

    input()


if __name__ == "__main__":
    main()
