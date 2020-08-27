import logging
import pathlib
import os
from dataclasses import dataclass

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
logger = logging.getLogger()


def _read_protocol_mapping() -> dict:
    map_file = BASE_DIR / 'static/ip_proto_map.csv'
    pairs = pd.read_csv(map_file, header=None)
    return dict(pairs.values.tolist())


BASE_DIR = pathlib.Path(__file__).resolve().parent
TEST_STATIC_DIR = BASE_DIR / 'tests' / 'static'

PCAP_OUTPUT_DIR = BASE_DIR / 'csv_files'
REPORT_DIR = BASE_DIR / 'reports'

IP_PROTO_MAPPING = _read_protocol_mapping()
RANDOM_SEED = 1

DEFAULT_PACKET_LIMIT_PER_FLOW = int(os.getenv('DEFAULT_PACKET_LIMIT_PER_FLOW', 20))
LOWER_BOUND_CLASS_OCCURRENCE = int(os.getenv('LOWER_BOUND_CLASS_OCCURRENCE', 50))

# customize, if needed
TARGET_CLASS_COLUMN = 'target_class'

# nfstream params
# the idle timeout follows many papers on traffic identification (JOY has 10 sec)
IDLE_TIMEOUT = 60
# active timeouts are set similarly, (Cisco's JOY tool has 30 sec)
ACTIVE_TIMEOUT_ONLINE = 60
ACTIVE_TIMEOUT_OFFLINE = 10e5

NEPTUNE_PROJECT = 'radion/traffic-classifier'


@dataclass
class FilePatterns:
    mawi = ('202004',)
    mawi_unswnb_iscxvpn = ('raw_csv', '202004')
    iot_home = ('electronics', 'camera', '2020', 'enivronment', 'healthcare', 'home', 'hub', 'light', 'trigger')
    mawi_iot_home = ('electronics', 'camera', '2020', 'enivronment', 'healthcare', 'home', 'hub', 'light',
                     'trigger', '202004')
