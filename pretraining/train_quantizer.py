import json
import pathlib

import numpy as np
import pandas as pd
from libKMCUDA import kmeans_cuda

from settings import logger, BASE_DIR


def plot_packets(packet_features):
    pd.DataFrame(packet_features).plot(kind='scatter', x=0, y=1, alpha=0.3, figsize=(12, 7), grid=True)


def get_kmeans_mae(original, restored):
    s = np.abs(original - restored).sum()
    mae = np.abs(original - restored).mean()
    logger.info(f'MAE: {mae}, cumulative error: {s}')
    return mae


def drop_nan_packets(packet_features):
    return packet_features[~np.isnan(packet_features)].reshape(-1, 2)


class PacketScaler:
    def __init__(self, max_packet_len=1500):
        self.max_packet_len = max_packet_len

    def transform(self, packet_pairs):
        """
        :param packet_pairs: (N, 2), 0 -- packet_len, 1 -- IAT
        :return: transformed_packets (N, 2)
        """
        packet_pairs[:, 0] = packet_pairs[:, 0] / self.max_packet_len
        # avoids warning and -inf values. the scale here is in microseconds (?)
        zero_iats = np.isclose(packet_pairs[:, 1], 0.)
        packet_pairs[:, 1][zero_iats] = 0
        packet_pairs[:, 1][~zero_iats] = np.log10(packet_pairs[:, 1][~zero_iats])
        return packet_pairs

    def inverse_transform(self, packet_pairs):
        packet_pairs[:, 0] = packet_pairs[:, 0] * self.max_packet_len
        # to correctly rescale, we need to know which were initially zeros
        zero_iats = np.isclose(packet_pairs[:, 1], 0., atol=1e-8)
        packet_pairs[:, 1][zero_iats] = 0
        packet_pairs[:, 1][~zero_iats] = 10 ** packet_pairs[:, 1][~zero_iats]
        return packet_pairs


class PacketQuantizer:
    def __init__(self, n_cluster=16384, flow_size=128, packet_scaler=PacketScaler):
        self.n_clusters = n_cluster
        # hard-coded to the expected dataframe format (as in feature_processing.py)
        self.iat_columns = [f'raw_iat{index}' for index in range(flow_size)]
        self.packet_columns = [f'raw_packet{index}' for index in range(flow_size)]
        self.raw_columns = [f'raw_{feature}{index}'
                            for index in range(flow_size)
                            for feature in ['packet', 'iat']
                            ]
        self.scaler = packet_scaler()
        self.cluster_centers_ = None

    def _transform_packets(self, raw_batch, filter_single_packet_flows=False):
        if filter_single_packet_flows:
            # do not consider single-packet flows
            raw_batch = raw_batch[raw_batch.raw_packet1 != 0]
        packet_features = raw_batch[self.raw_columns].values.reshape(-1, 2)
        return packet_features

    def fit(self, raw_batch):
        """
        https://github.com/src-d/kmcuda#python-api
        :param raw_batch:
        :return:
        """
        packet_features = self._transform_packets(raw_batch, filter_single_packet_flows=True)
        # omit non_packet values
        packet_features = drop_nan_packets(packet_features)
        init_clusters = "k-means++" if self.cluster_centers_ is None else self.cluster_centers_
        logger.info('fitting on {} packets, init clusters from data: {}'.format(packet_features.shape[0],
                                                                                isinstance(init_clusters, str)))
        packet_features = self.scaler.transform(packet_features)

        cluster_centers_, assignments = kmeans_cuda(
            samples=packet_features,
            clusters=self.n_clusters,
            tolerance=0.01,
            init=init_clusters,
            yinyang_t=0,
            metric="L2",
            average_distance=False,
            seed=1, device=0, verbosity=1
        )
        self.cluster_centers_ = cluster_centers_
        self.evaluate(packet_features, cluster_centers_[assignments])

    def evaluate(self, packet_features, restored):
        n_unique_clusters = len(self.cluster_centers_[~np.isnan(self.cluster_centers_)]) / 2
        logger.info(f'found {n_unique_clusters} unique clusters')
        get_kmeans_mae(packet_features, restored)

    def save_pretrained(self, save_directory):
        save_directory = pathlib.Path(save_directory)
        save_directory.mkdir(exist_ok=True)
        quantizer_path = save_directory / 'clusters.json'
        with open(quantizer_path, 'w') as qf:
            json.dump(self.cluster_centers_.tolist(), qf)
        logger.info(f'saving checkpoint to {quantizer_path}')
        return quantizer_path.as_posix()


def main():
    quantizer = PacketQuantizer()
    raw_csv_dir = pathlib.Path('/media/raid_store/pretrained_traffic/raw_csv_outer')

    flow_limit = 300_000
    for file_idx, csv in enumerate(raw_csv_dir.glob('*.csv')):
        logger.info(f'processing {csv}')
        reader = pd.read_csv(csv, chunksize=flow_limit, usecols=quantizer.raw_columns, dtype=np.float32)
        for batch, raw_packets in enumerate(reader):
            quantizer.fit(raw_packets)
            if batch % 10 == 0:
                quantizer.save_pretrained(BASE_DIR / f'pretraining/trained_quantizers/quantizer_2^14_{csv.stem}_{batch}')

        quantizer.save_pretrained(BASE_DIR / f'pretraining/trained_quantizers/quantizer_2^14_{csv.stem}_final')


if __name__ == '__main__':
    main()
