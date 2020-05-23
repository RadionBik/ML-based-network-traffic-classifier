import configparser
import logging
import pathlib

import pandas as pd


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(name)s %(message)s')


def _read_protocol_mapping() -> dict:
    map_file = BASE_DIR / 'static/ip_proto_map.csv'
    pairs = pd.read_csv(map_file, header=None)
    return dict(pairs.values.tolist())


BASE_DIR = pathlib.Path(__file__).resolve().parent
PCAP_OUTPUT_DIR = BASE_DIR / 'csv_files'

config = configparser.ConfigParser()
config.read(BASE_DIR / 'config.ini')

PACKET_LIMIT_PER_FLOW = int(config['parser']['packetLimitPerFlow'])

IP_PROTO_MAPPING = _read_protocol_mapping()
