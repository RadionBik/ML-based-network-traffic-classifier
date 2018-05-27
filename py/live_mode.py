#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

'''
https://github.com/pynetwork/pypcap
http://pypcap.readthedocs.io/en/latest/

performance is ok, packets appear faster than in tshark, no dropped
'''
import pcap, dpkt
import socket
#from pypacker.layer12 import ethernet
#from pypacker.layer3 import ip
#from pypacker.layer4 import tcp,udp
import pandas as pd
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

def liveFlowCaptureDpkt(targetDevice,flows,packetLimit=5):
    #start the capture
    sniffer = pcap.pcap(name=targetDevice, promisc=True, immediate=True, timeout_ms=50)
    count=0
    for ts,raw_pkt in sniffer:
        count = count+1
        #convert raw bytes to an ethernet object
        eth = dpkt.ethernet.Ethernet(raw_pkt)
        ip = eth.data
        #create the keys for IP UDP/TCP flows
        if isinstance(ip, dpkt.ip.IP):
            if isinstance(ip.data, dpkt.tcp.TCP):
                key = ('TCP', frozenset( ((ip_to_string(ip.src), ip.data.sport),(ip_to_string(ip.dst), ip.data.dport)) ))
            elif isinstance(ip.data, dpkt.udp.UDP):
                #hard-coded packet limit for UDP to allow for DNS
                packetLimit = 2
                key = ('UDP', frozenset( ((ip_to_string(ip.src), ip.data.sport),(ip_to_string(ip.dst), ip.data.dport)) ))
            else:
                continue

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

def statsToDataFrame(stats):
    flow = {feature: [] for feature in feature_extractor.FEATURES if
    (feature is not "proto" and feature is not "subproto") }
    if stats:
        for feature in feature_extractor.FEATURES:
            if feature is not "proto" and feature is not "subproto":
                flow[feature].append(stats[feature])

    return pd.DataFrame(flow)

def main():

    #packetLimit = 5
    targetDevice = selectDevice()
    #empty dict for flows, outside of captureFlows() to preserve the values
    flows={}
    #iterate infinitely to capture the flows 
    while True:
        key,flow = next(liveFlowCaptureDpkt(targetDevice,flows,10))
        stats = feature_extractor.getFlowStatsDpkt(flow)
        #convert to DataFrame with features as the header
        flowFeatures = pd.DataFrame.from_dict(stats,orient="index").T
        print(key,'\n',flowFeatures)

if __name__ == "__main__":
    main()