import logging

import dpkt
import nfstream
import numpy as np

logger = logging.getLogger('raw_packets_matrix')


class raw_packets_matrix(nfstream.NFPlugin):

    TIMESTAMP = 0
    IP_LEN = 1
    TRANSP_PAYLOAD = 2
    TCP_FLAGS = 3
    TCP_WINDOW = 4
    IP_PROTO = 5
    IS_CLIENT = 6

    def __init__(self, volatile=False, packet_limit=None):
        super().__init__(volatile=volatile, user_data=None)
        self.packet_limit = packet_limit

    def _fill_flow_stats(self, obs, raw_feature_matrix, counter=0):
        raw_feature_matrix[counter, self.TIMESTAMP] = obs.time
        raw_feature_matrix[counter, self.IP_LEN] = obs.ip_size
        raw_feature_matrix[counter, self.TRANSP_PAYLOAD] = obs.payload_size
        raw_feature_matrix[counter, self.TCP_FLAGS] = int(''.join(str(i) for i in obs.tcpflags), 2)
        if obs.protocol == 6 and obs.version == 4:
            packet = dpkt.ip.IP(obs.ip_packet)
            try:
                raw_feature_matrix[counter, self.TCP_WINDOW] = packet.data.win
            except AttributeError:
                logger.warning(f'unexpected packet format: {packet}')
                raw_feature_matrix[counter, self.TCP_WINDOW] = 0
        raw_feature_matrix[counter, self.IP_PROTO] = obs.protocol
        raw_feature_matrix[counter, self.IS_CLIENT] = 1 if obs.direction == 0 else 0
        return raw_feature_matrix

    def on_init(self, obs):
        raw_feature_matrix = np.zeros((self.packet_limit, 7))
        return self._fill_flow_stats(obs, raw_feature_matrix)

    def on_update(self, obs, entry):
        if entry.bidirectional_packets > self.packet_limit:
            return entry.raw_packets_matrix
        return self._fill_flow_stats(obs, entry.raw_packets_matrix, counter=entry.bidirectional_packets - 1)

    def on_expire(self, entry):
        # rm unfilled matrix rows
        entry.raw_packets_matrix = entry.raw_packets_matrix[entry.raw_packets_matrix[:, self.TIMESTAMP] > 0]
