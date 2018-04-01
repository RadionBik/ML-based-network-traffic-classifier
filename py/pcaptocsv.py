#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

#forked from vnetserg/traffic-v2 @ github.com

import argparse, re, dpkt, pcap
from pypacker import ppcap
from pypacker.layer12 import ethernet
from pypacker.layer3 import ip
from pypacker.layer4 import tcp,udp
from subprocess import Popen, PIPE
import pandas as ps
import numpy as np
import socket
import configparser
import os
import datetime
import feature_extractor

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

def parseFlowsDpkt(pcapfile):
    '''
        parse_flows() reads the PCAP-file, splits into flows,
        defines the application for each flow.
        Args:
            pcapfile - path to PCAP (string)
        returns (генерирует):
            The list of tuples as follows:
            (
                transport layer application,
                complementary protocol,
                the list of Ethernet-frames
            )
    '''

    pipe = Popen(["./ndpiReader", "-i", pcapfile, "-v2"], stdout=PIPE)
    raw = pipe.communicate()[0].decode("utf-8")
    #raw=open('logAtata').read()
    reg = re.compile(r'(UDP|TCP) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) <?->? (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) \[proto: [\d+\.]*\d+\/(\w+\.?\w+)*\]')
    flows = {}
    apps = {}
    f=open('captures.txt','w')
    for captures in re.findall(reg, raw):
        f.writelines("%s %s %s %s %s %s\n" % captures)
        
        transp_proto, ip1, port1, ip2, port2, app_proto = captures
        ip1 = ip_from_string(ip1)
        ip2 = ip_from_string(ip2)
        port1 = int(port1)
        port2 = int(port2)
        key = (transp_proto.lower(),frozenset(((ip1, port1), (ip2, port2))))
        flows[key] = []
        apps[key] = app_proto.split(".")
        if len(apps[key]) == 1:
            apps[key].append(None)
    f.close()

    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        eth = dpkt.ethernet.Ethernet(raw)
        ip = eth.data
        #check if the packet is IP, TCP, UDP
        if not isinstance(ip, dpkt.ip.IP):
            continue
        seg = ip.data
        if isinstance(seg, dpkt.tcp.TCP):
            transp_proto = "tcp"
        elif isinstance(seg, dpkt.udp.UDP):
            transp_proto = "udp"
        else:
            continue

        key = (transp_proto, frozenset(((ip.src, seg.sport),(ip.dst, seg.dport))))
        try:
            assert key in flows
        except AssertionError:
            print(ip.src,ip_to_string(ip.src))
            raise
        flows[key].append(eth)

    for key, flow in flows.items():
        yield apps[key][0], apps[key][1], flow

def parseFlowsPacker(pcapfile):
    '''
        parseFlowsPacker() reads the PCAP-file, splits into flows,
        defines the application for each flow.
        Args:
            pcapfile - path to PCAP (string)
        returns (генерирует):
            The list of tuples as follows:
            (
                transport layer application,
                complementary protocol,
                the list of Ethernet-frames
            )
    '''

    pipe = Popen(["./ndpiReader", "-i", pcapfile, "-v2"], stdout=PIPE)
    raw = pipe.communicate()[0].decode("utf-8")
    #raw=open('logAtata').read()
    reg = re.compile(r'(UDP|TCP) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) <?->? (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) \[proto: [\d+\.]*\d+\/(\w+\.?\w+)*\]')
    flows = {}
    apps = {}
    f=open('captures.txt','w')
    for captures in re.findall(reg, raw):
        f.writelines("%s %s %s %s %s %s\n" % captures)
        transp_proto, ip1, port1, ip2, port2, app_proto = captures
        ip1 = ip_from_string(ip1)
        ip2 = ip_from_string(ip2)
        port1 = int(port1)
        port2 = int(port2)
        key = (transp_proto.lower(),frozenset(((ip1, port1), (ip2, port2))))
        flows[key] = []
        apps[key] = app_proto.split(".")
        if len(apps[key]) == 1:
            apps[key].append(None)
    f.close()

    for ts, raw in ppcap.Reader(filename=pcapfile):
        eth = ethernet.Ethernet(raw)

        #create the keys for IP UDP/TCP flows
        if eth[ip.IP] is not None:
            if eth[tcp.TCP] is not None:
                key = ('tcp', frozenset(((eth.ip.src, eth.ip.tcp.sport),(eth.ip.dst, eth.ip.tcp.dport))))
            elif eth[udp.UDP] is not None:
                key = ('udp', frozenset(((eth.ip.src, eth.ip.udp.sport),(eth.ip.dst, eth.ip.udp.dport))))
            else:
                continue
            try:
                assert key in flows
            except AssertionError:
                print(eth.ip.src,eth.ip.src_s)
                raise
            flows[key].append(eth)
        else:
            continue

    for key, flow in flows.items():
        yield apps[key][0], apps[key][1], flow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", help="pcap file")
    parser.add_argument("-o", "--output", help="output csv file", default="flows.csv")
    parser.add_argument("-s", "--strip", help="leave only first N datagramms", metavar = "N", default=0, type=int)
    args = parser.parse_args()
    flows = {feature: [] for feature in feature_extractor.FEATURES}
    for pcapfile in args.file:
        if len(args.file) > 1:
            print(pcapfile)
        for proto, subproto, flow in parseFlowsDpkt(pcapfile):
            stats = feature_extractor.getFlowStatsDpkt(flow, args.strip)
            if stats:
                stats.update({"proto": proto, "subproto": subproto})
                for feature in feature_extractor.FEATURES:
                    flows[feature].append(stats[feature])
    data = ps.DataFrame(flows)
    print(data.proto.value_counts())
    config = configparser.ConfigParser()
    config.read(os.pardir+os.sep+'config.ini')
    data.to_csv(os.pardir+os.sep+config['OfflineMode']['folderWithCSVfiles']+os.sep+args.output, index=False)

if __name__ == "__main__":
    main()