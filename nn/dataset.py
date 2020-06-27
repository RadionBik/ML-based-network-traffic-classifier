import logging
from typing import Tuple

import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def get_train_val_test_datasets(X_train, y_train, X_test, y_test, device='cpu', val_part=0.8) \
        -> Tuple[TensorDataset, TensorDataset, TensorDataset]:

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

    X_train = torch.as_tensor(X_train, dtype=torch.float)
    y_train = torch.as_tensor(y_train, dtype=torch.long)

    train_dataset = _tensor_dataset(X_train, y_train)
    val_dataset = _tensor_dataset(X_val, y_val)
    test_dataset = _tensor_dataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset
