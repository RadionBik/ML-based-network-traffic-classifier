import os
import pathlib
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import logging
import pandas as pd
from flow_parsing.features import generate_raw_feature_names
from gpt_model.classifier.dataset import ClassificationQuantizedDataset
from settings import TARGET_CLASS_COLUMN, DEFAULT_PACKET_LIMIT_PER_FLOW


logger = logging.getLogger(__name__)


class SimpleClassificationQuantizedDataset(ClassificationQuantizedDataset):
    """ no attention mask and no dict-like output """
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_flow = self.tokenizer.batch_encode_packets(self.raw_flows.reshape(1, -1).astype(np.float64),
                                                       add_special_tokens=False,
                                                       return_attention_mask=False).data
        X = enc_flow['input_ids']
        y = torch.as_tensor(self.targets[i], dtype=torch.long)
        return X, y


class ClassificationPacketSizeDataset(Dataset):
    """
    the sequences are expected to be passed through embedding layer first, thus they are encoded to be positive and
    the modified PS itself will serve as an index

    max_size_range sets max dynamic range for PS parameter and implicitly sets Embedding layer dim
    """
    def __init__(
            self,
            dataset_path: str,
            max_size_range=5000,
            label_encoder: LabelEncoder = None,
            target_column=TARGET_CLASS_COLUMN,
            flow_size=DEFAULT_PACKET_LIMIT_PER_FLOW
    ):
        assert os.path.isfile(dataset_path)

        dataset_path = pathlib.Path(dataset_path)
        self.source_file = dataset_path
        logger.info("initializing dataset from %s", dataset_path)

        self.packet_columns = generate_raw_feature_names(flow_size, base_features=('packet',))
        raw_flows = pd.read_csv(self.source_file,
                                usecols=self.packet_columns + [target_column])

        if label_encoder is None:
            self.target_encoder = LabelEncoder().fit(raw_flows[target_column].values)
        else:
            self.target_encoder = label_encoder

        self.targets = self.target_encoder.transform(raw_flows[target_column].values)
        raw_flows = raw_flows.loc[:, self.packet_columns]
        raw_flows = raw_flows.fillna(0)
        # truncate values outside the range
        offset = max_size_range // 2
        raw_flows[raw_flows <= -offset] = -offset + 1
        raw_flows[raw_flows >= offset] = offset - 1
        self.raw_flows = raw_flows + offset
        logger.info('initialized dataset')

    def __len__(self):
        return len(self.raw_flows)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.as_tensor(self.raw_flows.values[i], dtype=torch.long)
        y = torch.as_tensor(self.targets[i], dtype=torch.long)
        return X, y
