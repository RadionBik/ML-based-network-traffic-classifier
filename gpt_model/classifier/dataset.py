import os
import pathlib
from functools import partial
from typing import Dict, List, Tuple

import logging
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from gpt_model.tokenizer import PacketTokenizer
from settings import TARGET_CLASS_COLUMN

logger = logging.getLogger(__name__)


class ClassificationQuantizedDataset(Dataset):
    def __init__(
        self, tokenizer: PacketTokenizer,
        dataset_path: str,
        label_encoder: LabelEncoder = None,
        target_column=TARGET_CLASS_COLUMN
    ):
        assert os.path.isfile(dataset_path)

        dataset_path = pathlib.Path(dataset_path)
        self.source_file = dataset_path
        logger.info("initializing dataset from %s", dataset_path)

        self.tokenizer = tokenizer

        raw_flows = pd.read_csv(self.source_file,
                                usecols=self.tokenizer.packet_quantizer.raw_columns + [target_column])

        if label_encoder is None:
            self.target_encoder = LabelEncoder().fit(raw_flows[target_column].values)
        else:
            self.target_encoder = label_encoder

        self.targets = self.target_encoder.transform(raw_flows[target_column].values)
        self.raw_flows = raw_flows.loc[:, tokenizer.packet_quantizer.raw_columns].values
        logger.info('initialized dataset')

    def __len__(self):
        return len(self.raw_flows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        enc_flow = self.tokenizer.batch_encode_packets(self.raw_flows[i].reshape(1, -1).astype(np.float64),
                                                       add_special_tokens=True,
                                                       return_attention_mask=True).data

        enc_flow.update({'target': torch.as_tensor(self.targets[i], dtype=torch.long)})
        return enc_flow

    @classmethod
    def get_collator(cls, mask_first_token):
        return partial(classification_quantized_collator, mask_first_token=mask_first_token)


def classification_quantized_collator(examples: List[Dict[str, torch.Tensor]], mask_first_token=True) -> \
        Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """ Data collator used for traffic classification """

    length_of_first = examples[0]['input_ids'].size(0)
    are_tensors_same_length = all(x['input_ids'].size(0) == length_of_first for x in examples)
    assert are_tensors_same_length

    input_ids = torch.cat([item['input_ids'] for item in examples], dim=0)
    attention_masks = torch.cat([item['attention_mask'] for item in examples], dim=0)
    if mask_first_token:
        attention_masks[:, 0] = 0
    targets = torch.cat([item['target'].view(1) for item in examples])
    return {"input_ids": input_ids, "attention_mask": attention_masks}, targets
