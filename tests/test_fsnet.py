from torch.utils.data import DataLoader
import torch

from fs_net.dataset import SimpleClassificationQuantizedDataset
from fs_net.model import FSNETClassifier


def test_forward(tokenizer, raw_dataset_file):
    """ simple smoke-test """
    ds = SimpleClassificationQuantizedDataset(tokenizer, dataset_path=raw_dataset_file, target_column='ndpi_app')
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    n_classes = len(ds.target_encoder.classes_)
    model = FSNETClassifier({}, ds.target_encoder.classes_, len(tokenizer))
    for flow in loader:
        output = model(flow[0])
        assert output.shape == torch.Size([4, n_classes])
