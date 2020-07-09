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

# set via analysis of packet number distribution within sessions @ mawi.wide.ad.jp/mawi/ditl/ditl2020/ pcaps,
# such that the limit is close to .99 percentile

PACKET_LIMIT_PER_FLOW = int(os.getenv('PACKET_LIMIT_PER_FLOW', 128))
LOWER_BOUND_CLASS_OCCURRENCE = int(os.getenv('LOWER_BOUND_CLASS_OCCURRENCE', 10))

# customize, if needed
TARGET_CLASS_COLUMN = 'target_class'

# nfstream params
# the idle timeout follows many papers on traffic identification
IDLE_TIMEOUT = 60
# active timeouts are set similarly to Cisco's JOY tool
ACTIVE_TIMEOUT_ONLINE = 30
ACTIVE_TIMEOUT_OFFLINE = 30
