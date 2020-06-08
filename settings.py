import logging
import pathlib
import os

import pandas as pd


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')


def _read_protocol_mapping() -> dict:
    map_file = BASE_DIR / 'static/ip_proto_map.csv'
    pairs = pd.read_csv(map_file, header=None)
    return dict(pairs.values.tolist())


BASE_DIR = pathlib.Path(__file__).resolve().parent
TEST_STATIC_DIR = BASE_DIR / 'tests' / 'static'

PCAP_OUTPUT_DIR = BASE_DIR / 'csv_files'


IP_PROTO_MAPPING = _read_protocol_mapping()
RANDOM_SEED = 1

PACKET_LIMIT_PER_FLOW = int(os.getenv('PACKET_LIMIT_PER_FLOW', 20))
LOWER_BOUND_CLASS_OCCURRENCE = int(os.getenv('LOWER_BOUND_CLASS_OCCURRENCE', 10))
