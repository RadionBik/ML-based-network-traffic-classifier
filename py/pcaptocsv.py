#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

#forked from vnetserg/traffic-v2 @ github.com

import argparse, re, dpkt
from subprocess import Popen, PIPE
import pandas as ps
import numpy as np
import socket
import configparser
import os
import datetime
# Full feature list of a flow:
FEATURES = [
    "proto", # app layer protocol
    "subproto", # complementary protocol field of the nDPI output
                # not used for now
    "bulk0", # the size of the first bulk of client
    "bulk1", # the size of the first bulk of server 
    "bulk2", # the size of the second bulk of client
    "bulk3", # the size of the second bulk of server 
    "client_packet0", # the size of the first segment of client
    "client_packet1", # the size of the second segment of client
    "server_packet0", # the size of the first segment of server
    "server_packet1", # the size of the second segment of server
    "client_bulksize_avg", # avg bulk size of client
    "client_bulksize_dev", # dev of the bulk size of client
    "server_bulksize_avg", # avg bulk size of server
    "server_bulksize_dev", # dev of the bulk size of server
    "client_packetsize_avg", # avg segment size of client
    "client_packetsize_dev", # dev of the segment size of client
    "server_packetsize_avg", # avg segment size of server
    "server_packetsize_dev", # dev of the segment size of server
#    "client_packets_per_bulk", # avg segment number per bulk from client 
#    "server_packets_per_bulk", # avg segment number per bulk from server 
#    "client_effeciency", # 
#    "server_efficiency", # 
#    "byte_ratio", # how many times more the client transmitted than the server in bytes 
#    "payload_ratio", # how many times more the client transmitted than the server in bytes of payload
#    "packet_ratio", # how many times more the client transmitted than the server in number of pckts
    "client_bytes", # bytes transmitted by client in total
    "client_payload", # payload transmitted by client in total
    "client_packets", # number of segments transmitted by client in total
    "client_bulks", # number of bulks transmitted by client in total
    "server_bytes", # bytes transmitted by server in total
    "server_payload", # payload transmitted by server in total
    "server_packets", # number of segments transmitted by server in total
    "server_bulks", # number of bulks transmitted by client in total
    "is_tcp" # whether it is TCP (only UDP/TCP are considered)
]

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

def parse_flows(pcapfile):
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
        #print(app_proto)
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
        #print(str(datetime.datetime.utcfromtimestamp(ts)),'\n')
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
        #print(key,'here')
        try:
            assert key in flows
        except AssertionError:
            print(ip.src,ip_to_string(ip.src))
            raise
        flows[key].append(eth)

    for key, flow in flows.items():
        yield apps[key][0], apps[key][1], flow
        #print(key)

def forge_flow_stats(flow, strip = 0):
    '''
        forge_flow_stats() calculates stat. features of a flow.
        Args:
            flow - the list of Ethernet-frames
            strip - a number of first frames to calculate features with
			(if < 1, then frames are NOT discarded)
        returns:
            a dict, where keys are the names of features,
            items are the values.
            If there are no at least 2 buld of data in the flow,
			it returns None.
    '''
    ip = flow[0].data
    seg = ip.data
    '''
    if isinstance(seg, dpkt.tcp.TCP):
        
        try:
            seg2 = flow[1].data.data
        except IndexError:
            return None
        if not (seg.flags & dpkt.tcp.TH_SYN and seg2.flags & dpkt.tcp.TH_SYN):
            return None
        proto = "tcp"
        flow = flow[3:] # срезаем tcp handshake
    elif isinstance(seg, dpkt.udp.UDP):
        proto = "udp"
'''
    if isinstance(seg, dpkt.tcp.TCP):
        proto = "tcp"
		# check if there is SYN flag in first 2 packets:
        if (seg.flags & dpkt.tcp.TH_SYN and flow[1].data.data.flags & dpkt.tcp.TH_SYN):
            flow = flow[3:] # cut out the tcp handshake
    elif isinstance(seg, dpkt.udp.UDP):
        proto = "udp"

    else:
        raise ValueError("Unknown transport protocol: `{}`".format(
            seg.__class__.__name__))

    if strip > 0:
        flow = flow[:strip]

    client = (ip.src, seg.sport)
    server = (ip.dst, seg.dport)

    client_bulks = []
    server_bulks = []
    client_packets = []
    server_packets = []

    cur_bulk_size = 0
    cur_bulk_owner = "client"
    client_fin = False
    server_fin = False
    for eth in flow:
        ip = eth.data
        seg = ip.data
        if (ip.src, seg.sport) == client:
            if client_fin: continue
            if proto == "tcp":
                client_fin = bool(seg.flags & dpkt.tcp.TH_FIN)
            client_packets.append(len(seg))
            if cur_bulk_owner == "client":
                cur_bulk_size += len(seg.data)
            elif len(seg.data) > 0:
                server_bulks.append(cur_bulk_size)
                cur_bulk_owner = "client"
                cur_bulk_size = len(seg.data)
        elif (ip.src, seg.sport) == server:
            if server_fin: continue
            if proto == "tcp":
                server_fin = bool(seg.flags & dpkt.tcp.TH_FIN)
            server_packets.append(len(seg))
            if cur_bulk_owner == "server":
                cur_bulk_size += len(seg.data)
            elif len(seg.data) > 0:
                client_bulks.append(cur_bulk_size)
                cur_bulk_owner = "server"
                cur_bulk_size = len(seg.data)
        else:
            raise ValueError("There is more than one flow here!")

    if cur_bulk_owner == "client":
        client_bulks.append(cur_bulk_size)
    else:
        server_bulks.append(cur_bulk_size)

    stats = {
        "bulk0": client_bulks[0] if len(client_bulks) > 0 else 0,
        "bulk1": server_bulks[0] if len(server_bulks) > 0 else 0,
        "bulk2": client_bulks[1] if len(client_bulks) > 1 else 0,
        "bulk3": server_bulks[1] if len(server_bulks) > 1 else 0,
        "client_packet0": client_packets[0] if len(client_packets) > 0 else 0,
        "client_packet1": client_packets[1] if len(client_packets) > 1 else 0,
        "server_packet0": server_packets[0] if len(server_packets) > 0 else 0,
        "server_packet1": server_packets[1] if len(server_packets) > 1 else 0,
    }

    if client_bulks and client_bulks[0] == 0:
        client_bulks = client_bulks[1:]

#    if not client_bulks or not server_bulks:
#        return None

    stats.update({
        "client_bulksize_avg": np.mean(client_bulks),
        "client_bulksize_dev": np.std(client_bulks),
        "server_bulksize_avg": np.mean(server_bulks),
        "server_bulksize_dev": np.std(server_bulks),
        "client_packetsize_avg": np.mean(client_packets),
        "client_packetsize_dev": np.std(client_packets),
        "server_packetsize_avg": np.mean(server_packets),
        "server_packetsize_dev": np.std(server_packets),
#        "client_packets_per_bulk": len(client_packets)/len(client_bulks),
#        "server_packets_per_bulk": len(server_packets)/len(server_bulks),
#        "client_effeciency": sum(client_bulks)/sum(client_packets),
#        "server_efficiency": sum(server_bulks)/sum(server_packets),
#        "byte_ratio": sum(client_packets)/sum(server_packets),
#        "payload_ratio": sum(client_bulks)/sum(server_bulks),
#        "packet_ratio": len(client_packets)/len(server_packets),
        "client_bytes": sum(client_packets),
        "client_payload": sum(client_bulks),
        "client_packets": len(client_packets),
        "client_bulks": len(client_bulks),
        "server_bytes": sum(server_packets),
        "server_payload": sum(server_bulks),
        "server_packets": len(server_packets),
        "server_bulks": len(server_bulks),
        "is_tcp": int(proto == "tcp")
    })

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", help="pcap file")
    parser.add_argument("-o", "--output", help="output csv file", default="flows.csv")
    parser.add_argument("-s", "--strip", help="leave only first N datagramms", metavar = "N", default=0, type=int)
    args = parser.parse_args()
    flows = {feature: [] for feature in FEATURES}
    for pcapfile in args.file:
        if len(args.file) > 1:
            print(pcapfile)
        for proto, subproto, flow in parse_flows(pcapfile):
            #print(proto)
            stats = forge_flow_stats(flow, args.strip)
#            stats = forge_flow_stats(flow)
            if stats:
                stats.update({"proto": proto, "subproto": subproto})
                for feature in FEATURES:
                    flows[feature].append(stats[feature])
    data = ps.DataFrame(flows)
    print(data.proto.value_counts())
    config = configparser.ConfigParser()
    config.read(os.pardir+os.sep+'config.ini')
    data.to_csv(os.pardir+os.sep+config['GeneralSettings']['folderWithCSVfiles']+os.sep+args.output, index=False)

if __name__ == "__main__":
    main()