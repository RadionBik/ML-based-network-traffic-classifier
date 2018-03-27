'''
https://github.com/pynetwork/pypcap
http://pypcap.readthedocs.io/en/latest/

performance is ok, packets appear faster than in tshark, no dropped
'''
import numpy as np
import pcap
from pypacker.layer12 import ethernet
from pypacker.layer3 import ip
from pypacker.layer4 import tcp,udp
import pandas

def selectDevice():
	deviceList = pcap.findalldevs()
	print("We found the following devices:")
	deviceNumber=0
	for device in deviceList:
		deviceNumber=deviceNumber+1
		print(deviceNumber,"\t",device)

	print("Which one to use for capturing? Enter the number:")
	while True:
		try:
			userNumber = int(input())
		except ValueError:
			print("The number is not integer! Try again")
			continue
		else:
			if userNumber>len(deviceList):
				print("The input is out of the range! Try again")
				continue
			else:
				targetDevice=deviceList[userNumber-1]
				print(targetDevice," was selected")
				break
	return targetDevice

def liveFlowCapture(targetDevice,flows,packetLimit=5):
	#start the capture
	sniffer = pcap.pcap(name=targetDevice, promisc=True, immediate=True, timeout_ms=50)
	count=0
	for ts,raw_pkt in sniffer:
		count = count+1
		#convert raw bytes to an ethernet object
		eth = ethernet.Ethernet(raw_pkt)
		#create the keys for IP UDP/TCP flows
		if eth[ip.IP] is not None:
			if eth[tcp.TCP] is not None:
				key = ('TCP', frozenset(((eth.ip.src_s, eth.ip.tcp.sport),(eth.ip.dst_s, eth.ip.tcp.dport))))
			elif eth[udp.UDP] is not None:
				key = ('UDP', frozenset(((eth.ip.src_s, eth.ip.udp.sport),(eth.ip.dst_s, eth.ip.udp.dport))))
			else:
				continue
			#print(key,' ','payload size:',len(eth.upper_layer.upper_layer.body_bytes))
			#create an entry in the flow dict if one is absent. init with the unprocessed mark
			if key not in flows:
				#print("New flow detected:\n",key)
				flows[key]=[False]
				#print(flows[key])

			if (len(flows[key]) < packetLimit+1):
				#append packets to the flow entry
				flows[key].append(eth)

			#if there is enough packets and flow has not been processed --> yield the values
			elif (len(flows[key]) == packetLimit+1) and (flows[key][0] == False):
				flows[key][0] = True		
				yield key,flows[key][1:]

def getFlowStats(flow, strip = 0):
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


#packetLimit = 5
targetDevice = selectDevice()
#empty dict for flows, outside of captureFlows() to preserve the values
flows={}

#iterate infinitely to capture the flows 
while True:
	key,flow = next(liveFlowCapture(targetDevice,flows,10))
	stats = getFlowStats(flow)
	#print("Key is {}".format(key))
	#print(stats,'\n')
	for featName,featVal in stats.items():
		print(featName,featVal)


#print(flows)