import numpy as np
from torch.utils.data import DataLoader

from pretraining.dataset import FlowDataset, FlowCollator
from pretraining.quantizer import PacketScaler, init_sklearn_kmeans_from_checkpoint, PacketQuantizer
from pretraining.tokenizer import PacketTokenizer

np.random.seed(1)


def test_packet_scaler():
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


def _estimate_normalized_packet_difference(raw_packets, reverted_packets):
    norm_diff = (reverted_packets - raw_packets) / reverted_packets
    norm_diff[np.isnan(norm_diff) | np.isinf(norm_diff)] = 0
    return norm_diff.mean()


def test_quantizer_transform(quantizer_checkpoint, raw_dataset):

    q = PacketQuantizer.from_checkpoint(quantizer_checkpoint, flow_size=20)
    # assert proper column ordering with packet features
    raw_packets = raw_dataset[q.raw_columns].values
    quantized = q.transform(raw_packets)
    assert quantized.shape == (raw_dataset.shape[0], 20)
    assert np.isnan(raw_packets).sum() == (quantized == -1).sum() * 2

    # test invariance
    assert np.isclose(quantized, q.transform(raw_packets)).all()

    # test inverting
    reverted_packets = q.inverse_transform(quantized)
    assert reverted_packets.shape == raw_packets.shape
    assert np.isnan(reverted_packets).sum() == np.isnan(raw_packets).sum()

    assert _estimate_normalized_packet_difference(raw_packets, reverted_packets) < 0.0003


def test_tokenize_detokenize(quantizer_checkpoint, raw_dataset):
    tokenizer = PacketTokenizer.from_pretrained(quantizer_checkpoint)
    encoded = tokenizer.batch_encode_plus(raw_dataset)
    tokens = encoded['input_ids']
    # since the model limit 128 > 20 in raw_features, we do not expect truncating
    decoded = tokenizer.batch_decode(tokens)
    assert _estimate_normalized_packet_difference(raw_dataset.values, decoded) < 0.0003


def test_flow_loader(raw_dataset_folder, quantizer_checkpoint):
    tokenizer = PacketTokenizer.from_pretrained(quantizer_checkpoint, flow_size=20)
    ds = FlowDataset(tokenizer, folder_path=raw_dataset_folder)
    loader = DataLoader(ds, batch_size=4, collate_fn=FlowCollator(tokenizer), drop_last=True)
    for flow in loader:
        assert flow['input_ids'].shape == (4, 22)
