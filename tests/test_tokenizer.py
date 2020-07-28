import numpy as np

import flow_parser
from pretraining import tokenizer
from pretraining.quantizer import PacketScaler, init_sklearn_kmeans_from_checkpoint, PacketQuantizer

import settings

np.random.seed(1)


def test_transformer():
    n_packets = 1000
    pack_lens = np.random.uniform(-1500, 1500, n_packets)
    iats = np.random.gamma(0, scale=1e4, size=n_packets)
    indices = np.random.choice(np.arange(iats.size), replace=False, size=int(iats.size * 0.2))
    iats[indices] = 0.

    packets = np.stack([pack_lens, iats], axis=1)
    transformer = PacketScaler()
    transf_packets = transformer.transform(packets.copy())
    reverted_packets = transformer.inverse_transform(transf_packets)
    assert np.isclose(packets, reverted_packets, atol=10e-9).all()


def test_loading_quantizer(quantizer_checkpoint):
    q = init_sklearn_kmeans_from_checkpoint(quantizer_checkpoint)
    cluster = q.predict(np.array([[-1, 0]]))
    assert cluster[0] == 8


def test_quantizer_transform(quantizer_checkpoint, pcap_example_path):
    raw_dataset = flow_parser.parse_features_to_dataframe(pcap_example_path,
                                                          derivative_features=False,
                                                          raw_features=20,
                                                          online_mode=False)
    q = PacketQuantizer.from_checkpoint(quantizer_checkpoint, flow_size=20)

    # assert proper column ordering with packet features
    raw_packets = raw_dataset.filter(regex='raw')[q.raw_columns].values
    quantized = q.transform(raw_packets)
    assert quantized.shape == (raw_dataset.shape[0], 20)
    assert np.isnan(raw_packets).sum() == (quantized == -1).sum() * 2

    reverted_packets = q.inverse_transform(quantized)
    assert reverted_packets.shape == raw_packets.shape
    assert np.isnan(reverted_packets).sum() == np.isnan(raw_packets).sum()

    norm_diff = (reverted_packets - raw_packets) / reverted_packets
    norm_diff[np.isnan(norm_diff) | np.isinf(norm_diff)] = 0
    mean_diff = norm_diff.mean()
    assert mean_diff < 0.0003
