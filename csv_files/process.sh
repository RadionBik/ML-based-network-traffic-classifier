#!/bin/bash

ls -1d ~/Applications/traffic_dumps/separated_iot_devices/* | xargs -n1 -t python ../flow_parser.py --raw -p
python ../flow_parser.py -p ~/Applications/traffic_dumps/non_iot.pcap --raw