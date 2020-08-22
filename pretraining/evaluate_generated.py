import pathlib

import numpy as np
import pandas as pd
import scipy

from datasets import read_dataset
from feature_processing import inter_packet_times_from_timestamps
from pretraining.dataset import load_modeling_data_with_classes
from settings import REPORT_DIR, logger


def convert_ipt_to_iat(flows):
    """ converts IPT (timing between 2 any packets) to IAT
        (timing between 2 consecutive packets within 1 direction) """
    def ipt_to_iat(flow):
        """
        :param flow: source flow of size (packet_num, feature_num=2)
        :return:
        """
        timestamps_like = np.cumsum(flow[:, 1])
        direction_from_mask = flow[:, 0] > 0
        direction_to_mask = flow[:, 0] < 0

        iat_flow = np.full(flow.shape, np.nan)
        iat_flow[direction_from_mask, 1] = inter_packet_times_from_timestamps(timestamps_like[direction_from_mask])
        iat_flow[direction_to_mask, 1] = inter_packet_times_from_timestamps(timestamps_like[direction_to_mask])
        iat_flow[:, 0] = flow[:, 0]
        return iat_flow

    source_shape = flows.shape
    raw_packets = flows.reshape(-1, 2)  # per-packet view
    raw_packets = raw_packets.reshape(-1, source_shape[1] // 2, 2)  # (n_flows, n_packets, features)
    iat_packets = np.empty_like(raw_packets)
    for i in range(source_shape[0]):
        iat_packets[i, :, :] = ipt_to_iat(raw_packets[i])
    iat_packets = iat_packets.reshape(source_shape)
    return iat_packets


def plot_packets(packet_features):
    pd.DataFrame(packet_features).plot(kind='scatter', x=0, y=1, alpha=0.3, figsize=(12, 7), grid=True)


def packets_per_flow(flows):
    non_packet_mask = ~np.isnan(flows)
    packets_per_flow = non_packet_mask.sum(1) / 2
    return packets_per_flow


def flows_to_packets(flows):
    return flows[~np.isnan(flows)].reshape(-1, 2)


def get_kl_divergence_continuous(orig_values, gen_values):
    try:
        x_values = np.linspace(0, max(orig_values), 100)
        kde_orig = scipy.stats.gaussian_kde(orig_values)(x_values)
        kde_gen = scipy.stats.gaussian_kde(gen_values)(x_values)
        return scipy.stats.entropy(kde_orig, kde_gen)
    except Exception as e:
        logger.error(f'cannot get KDE of value(s), reason: {e}')
        return np.nan


def packets_to_throughput(packets, resolution='1S'):
    # replace indexes with DateTime format
    df = pd.Series(
        packets[:, 0],
        index=pd.to_datetime(np.cumsum(packets[:, 1]), unit='ms')
    )
    throughput = df.resample(resolution).sum()
    return throughput.values


def evaluate_generated_traffic(src_flows: np.ndarray, gen_flows: np.ndarray) -> dict:
    distance_packets_per_flow = get_kl_divergence_continuous(
        packets_per_flow(src_flows),
        packets_per_flow(gen_flows)
    )

    common_metrics = {
        'KL div n_packets/flow': distance_packets_per_flow,
        'N flows': min(src_flows.shape[0], gen_flows.shape[0])
    }
    src_packets = flows_to_packets(convert_ipt_to_iat(src_flows))
    gen_packets = flows_to_packets(convert_ipt_to_iat(gen_flows))

    client_src_mask = src_packets[:, 0] > 0
    client_gen_mask = gen_packets[:, 0] > 0

    client_src_packets = src_packets[client_src_mask]
    server_src_packets = src_packets[~client_src_mask]

    client_gen_packets = gen_packets[client_gen_mask]
    server_gen_packets = gen_packets[~client_gen_mask]

    throughput = {
        'Src avg throughput bytes/s (client)': np.mean(packets_to_throughput(client_src_packets)),
        'Gen avg throughput bytes/s (client)': np.mean(packets_to_throughput(client_gen_packets)),
        'Src avg throughput bytes/s (server)': np.mean(packets_to_throughput(server_src_packets)),
        'Gen avg throughput bytes/s (server)': np.mean(packets_to_throughput(server_gen_packets)),
    }

    per_direction_kl_divergences = {
        'KL div PS  (client)': get_kl_divergence_continuous(client_src_packets[:, 0], client_gen_packets[:, 0]),
        'KL div IAT (client)': get_kl_divergence_continuous(client_src_packets[:, 1], client_gen_packets[:, 1]),
        'KL div PS  (server)': get_kl_divergence_continuous(server_src_packets[:, 0], server_gen_packets[:, 0]),
        'KL div IAT (server)': get_kl_divergence_continuous(server_src_packets[:, 1], server_gen_packets[:, 1]),
        'KL div throughput (client)': get_kl_divergence_continuous(packets_to_throughput(client_src_packets),
                                                                   packets_to_throughput(client_gen_packets)),
        'KL div throughput (server)': get_kl_divergence_continuous(packets_to_throughput(server_src_packets),
                                                                   packets_to_throughput(server_gen_packets)),
    }

    return dict(**common_metrics, **per_direction_kl_divergences, **throughput)


def main():

    generated_folder = pathlib.Path('/media/raid_store/pretrained_traffic/generated_flows_gpt2_model_2epochs_classes')
    all_source_flows, classes = load_modeling_data_with_classes('/media/raid_store/pretrained_traffic/train_csv')

    metrics = {}
    for file in generated_folder.glob('*.csv'):
        gen_flows = read_dataset(file)
        src_flows = all_source_flows[classes == file.stem]
        results = evaluate_generated_traffic(src_flows.values, gen_flows.values)
        metrics[file.stem] = results
    pd.DataFrame(metrics).to_csv(REPORT_DIR / ('report_' + generated_folder.stem))


if __name__ == '__main__':
    main()
