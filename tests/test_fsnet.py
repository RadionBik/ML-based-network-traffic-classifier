from torch.utils.data import DataLoader
import torch

from fs_net.dataset import SimpleClassificationQuantizedDataset, ClassificationPacketSizeDataset
from fs_net.model import FSNETClassifier


def test_packet_ds(raw_dataset_file):
    ds = ClassificationPacketSizeDataset(raw_dataset_file, max_size_range=100, target_column='ndpi_app')
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    for flow, target in loader:
        assert flow.min() == torch.tensor(1) and \
               flow.max() == torch.tensor(99) and \
               flow.shape == torch.Size([4, 20])


def test_forward_packet_ds(raw_dataset_file):
    ds = ClassificationPacketSizeDataset(raw_dataset_file, max_size_range=100, target_column='ndpi_app')
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    n_classes = len(ds.target_encoder.classes_)
    model = FSNETClassifier({}, ds.target_encoder.classes_, 100)
    for flow, target in loader:
        output = model(flow)
        assert output.shape == torch.Size([4, n_classes])


def test_forward(tokenizer, raw_dataset_file):
    """ simple smoke-test """
    ds = SimpleClassificationQuantizedDataset(tokenizer, dataset_path=raw_dataset_file, target_column='ndpi_app')
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    n_classes = len(ds.target_encoder.classes_)
    model = FSNETClassifier({}, ds.target_encoder.classes_, len(tokenizer))
    for flow in loader:
        output = model(flow[0])
        assert output.shape == torch.Size([4, n_classes])
