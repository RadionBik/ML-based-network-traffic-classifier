import logging
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from gpt_model.classifier.dataset import ClassificationQuantizedDataset

logger = logging.getLogger(__name__)


def get_train_val_test_datasets(X_train, y_train, X_test, y_test, device='cpu', val_part=0.9) \
        -> Tuple[TensorDataset, TensorDataset, TensorDataset]:

    """
    converts sklearn-like dataset into torch-compatible one
    """
    def _tensor_dataset(X, y):
        return TensorDataset(torch.as_tensor(X, device=device, dtype=torch.float),
                             torch.as_tensor(y, device=device, dtype=torch.long))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        train_size=val_part,
        stratify=y_train,
        random_state=1
    )
    train_dataset = _tensor_dataset(X_train, y_train)
    val_dataset = _tensor_dataset(X_val, y_val)
    test_dataset = _tensor_dataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset


class ClassificationQuantizedDatasetNoAttention(ClassificationQuantizedDataset):
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_flow = self.tokenizer.batch_encode_packets(self.raw_flows[i].reshape(1, -1).astype(np.float64),
                                                       add_special_tokens=True,
                                                       return_attention_mask=False).data
        X = enc_flow['input_ids']
        y = torch.as_tensor(self.targets[i], dtype=torch.long)
        return X, y
