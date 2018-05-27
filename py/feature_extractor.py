'''
This module is responsible for feature extracting from a flow
Implements 2 versions based on PyPacker and DPKT modules.
DPKT is considered to be more stable. 
'''

import dpkt
#from pypacker.layer12 import ethernet
#from pypacker.layer3 import ip
#from pypacker.layer4 import tcp,udp
import numpy as np


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


def getFlowStatsPacker(flow, strip = 0):
    '''
        getFlowStatsPacker() calculates stat. features of a flow.
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
    ip_layer = flow[0].upper_layer
    transp_layer = ip_layer.upper_layer

    if ip_layer[tcp.TCP] is not None:
        proto = "tcp"
        try:
        	transp_layer2 = flow[1].upper_layer.upper_layer
        except IndexError:
        	return None
		# check if there is SYN flag in first 2 packets:
        if (transp_layer.flags & tcp.TH_SYN and transp_layer2.flags & tcp.TH_SYN):
        	flow = flow[3:] # cut out the tcp handshake
    elif ip_layer[udp.UDP] is not None:
        proto = "udp"

    else:
        raise ValueError("Unknown transport protocol: `{}`".format(
            transp_layer.__class__.__name__))

    if strip > 0:
        flow = flow[:strip]

    client = (ip_layer.src, transp_layer.sport)
    server = (ip_layer.dst, transp_layer.dport)

    client_bulks = []
    server_bulks = []
    client_packets = []
    server_packets = []

    cur_bulk_size = 0
    cur_bulk_owner = "client"
    client_fin = False
    server_fin = False
    for eth in flow:
        ip_layer = eth.upper_layer
        transp_layer = ip_layer.upper_layer
        if (ip_layer.src, transp_layer.sport) == client:
            if client_fin: continue
            if proto == "tcp":
                client_fin = bool(transp_layer.flags & tcp.TH_FIN)
            client_packets.append(len(transp_layer))
            if cur_bulk_owner == "client":
                cur_bulk_size += len(transp_layer.body_bytes)
            elif len(transp_layer.body_bytes) > 0:
                server_bulks.append(cur_bulk_size)
                cur_bulk_owner = "client"
                cur_bulk_size = len(transp_layer.body_bytes)
        elif (ip_layer.src, transp_layer.sport) == server:
            if server_fin: continue
            if proto == "tcp":
                server_fin = bool(transp_layer.flags & tcp.TH_FIN)
            server_packets.append(len(transp_layer))
            if cur_bulk_owner == "server":
                cur_bulk_size += len(transp_layer.body_bytes)
            elif len(transp_layer.body_bytes) > 0:
                client_bulks.append(cur_bulk_size)
                cur_bulk_owner = "server"
                cur_bulk_size = len(transp_layer.body_bytes)
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


def getFlowStatsDpkt(flow, strip = 0):
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
        try:
            seg2 = flow[1].data.data
        except IndexError:
            return None
        # check if there is SYN flag in first 2 packets:
        if (seg.flags & dpkt.tcp.TH_SYN and seg2.flags & dpkt.tcp.TH_SYN):
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

    if not client_bulks:
        client_bulks = [0]

    if not server_bulks:
        server_bulks = [0]

    if not client_packets:
        client_packets = [0]
    if not server_packets:
        server_packets = [0]
    
    
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