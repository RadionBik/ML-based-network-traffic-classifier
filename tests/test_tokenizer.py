import numpy as np

from pretraining import tokenizer
from pretraining.train_quantizer import PacketScaler
import settings

np.random.seed(1)


def test_transformer():
    n_packets = 1000
    pack_lens = np.random.uniform(-1500, 1500, n_packets)
    iats = np.random.gamma(1, scale=1e4, size=n_packets)
    indices = np.random.choice(np.arange(iats.size), replace=False, size=int(iats.size * 0.2))
    iats[indices] = 0.

    packets = np.stack([pack_lens, iats], axis=1)
    transformer = PacketScaler()
    transf_packets = transformer.transform(packets.copy())
    reverted_packets = transformer.inverse_transform(transf_packets)
    assert np.isclose(packets, reverted_packets, atol=10e-9).all()


def test_from_pretrained():
    q = tokenizer.PacketTokenizer.from_pretrained(settings.TEST_STATIC_DIR)