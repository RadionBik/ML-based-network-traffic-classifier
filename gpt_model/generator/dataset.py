import os
import pathlib
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Dict, List, Tuple

import logging
import numpy as np
import pandas as pd
import sh
import torch
from torch.utils.data.dataset import IterableDataset, Dataset
from transformers import BatchEncoding

from flow_parsing import check_filename_in_patterns
from settings import TARGET_CLASS_COLUMN
from gpt_model.tokenizer import PacketTokenizer
from gpt_model.data_preparation.format_parsed_as_classification_dataset import prepare_classification_data

logger = logging.getLogger(__name__)


class PretrainIterDataset(IterableDataset):

    def __init__(self, tokenizer: PacketTokenizer, folder_path: str, train_mode=True):
        assert os.path.isdir(folder_path)
        # TODO feature caching, multiple workers?, filter out one-packet flows

        self.source_files = list(pathlib.Path(folder_path).glob('*.csv'))
        logger.info("initializing dataset from %s with %s files", folder_path, len(self.source_files))

        self.tokenizer = tokenizer
        self.train_mode = train_mode

    def __iter__(self) -> BatchEncoding:
        assert torch.utils.data.get_worker_info() is None
        for csv in self.source_files:
            # not really the best way, reading is gonna be slow
            logger.info(f'FlowDataset: reading {csv}')
            reader = pd.read_csv(csv, chunksize=1,
                                 usecols=self.tokenizer.packet_quantizer.raw_columns,
                                 dtype=float)
            for raw_flow in reader:
                # skip 1-packet and empty flows
                if self.train_mode and pd.isna(raw_flow.iloc[:, 3]).any():
                    continue

                encoded = self.tokenizer.batch_encode_packets(raw_flow,
                                                              add_special_tokens=True,
                                                              return_attention_mask=True)
                yield encoded

    @lru_cache(maxsize=2)
    def __len__(self):
        """ the files are too large to count their size via Python """
        line_counter = sh.Command('sed')
        total = 0
        for filename in self.source_files:
            found_lines = line_counter("-n", "$=", filename)
            # do not count .csv header
            total += int(found_lines) - 1
        return total


class PretrainDataset(Dataset):
    def __init__(self, tokenizer: PacketTokenizer, folder_path: str, filename_patterns_to_exclude: tuple = ()):
        assert os.path.isdir(folder_path)
        # TODO feature caching, multiple workers?, filter out one-packet flows

        source_files = list(pathlib.Path(folder_path).glob('*.csv'))
        file_matcher = partial(check_filename_in_patterns, patterns=filename_patterns_to_exclude)
        source_files = list(file for file in source_files if not file_matcher(file))
        print(source_files)
        self.source_files = source_files
        logger.info("initializing dataset from %s with %s files", folder_path, len(self.source_files))

        self.tokenizer = tokenizer
        # load as 32-bit to save RAM
        raw_flows = pd.concat((pd.read_csv(csv, usecols=self.tokenizer.packet_quantizer.raw_columns, dtype=np.float32)
                               for csv in self.source_files), ignore_index=True)

        raw_flows = raw_flows.loc[:, tokenizer.packet_quantizer.raw_columns].sample(frac=1, random_state=1)

        logger.info('concatenated dataframes within the folder')
        # skip 1-packet and empty flows
        raw_flows.dropna(axis=0, subset=['raw_packet0', 'raw_packet1'], inplace=True, how='any')
        self.raw_flows = raw_flows.values
        logger.info('initialized dataset')

    def __len__(self):
        return len(self.raw_flows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer.batch_encode_packets(self.raw_flows[i].reshape(1, -1).astype(np.float64),
                                                   add_special_tokens=True,
                                                   return_attention_mask=True).data


def load_modeling_data_with_classes(
        folder_path,
        shuffle=True,
        filename_patterns_to_exclude=None
) -> Tuple[pd.DataFrame, pd.Series]:
    assert os.path.isdir(folder_path)
    logger.info(f"initializing dataset from {folder_path}, excluding {filename_patterns_to_exclude}")
    folder_path = pathlib.Path(folder_path)

    raw_flows = prepare_classification_data(folder_path,
                                            remove_garbage=False,
                                            filename_patterns_to_exclude=filename_patterns_to_exclude)
    # skip 1-packet and empty flows
    raw_flows.dropna(axis=0, subset=['raw_packet0', 'raw_packet1'], inplace=True, how='any')
    if shuffle:
        raw_flows = raw_flows.sample(frac=1, random_state=1)

    return raw_flows.filter(regex='raw_'), raw_flows[TARGET_CLASS_COLUMN]


class PretrainDatasetWithClasses(Dataset):
    def __init__(self, tokenizer: PacketTokenizer, folder_path: str, filename_patterns_to_exclude: tuple = ()):
        self.tokenizer = tokenizer

        raw_flows, targets = load_modeling_data_with_classes(folder_path,
                                                             filename_patterns_to_exclude=filename_patterns_to_exclude)

        self.raw_flows: np.ndarray = raw_flows.loc[:, tokenizer.packet_quantizer.raw_columns].values
        self.targets: np.ndarray = targets.values
        logger.info('initialized dataset')
        tokenizer.add_class_tokens(self.target_classes)
        logger.info('added special tokens representing classes')

    @property
    def target_classes(self) -> list:
        return np.unique(self.targets).tolist()

    def __len__(self):
        return len(self.raw_flows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer.batch_encode_packets(self.raw_flows[i].reshape(1, -1).astype(np.float64),
                                                   target_class=self.targets[i],
                                                   add_special_tokens=True,
                                                   return_attention_mask=True).data


@dataclass
class PretrainCollator:
    """
    Data collator used for traffic flow modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PacketTokenizer

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Data collator used for packet modeling.
        - collates batches of tensors
        """

        length_of_first = examples[0]['input_ids'].size(0)
        are_tensors_same_length = all(x['input_ids'].size(0) == length_of_first for x in examples)
        assert are_tensors_same_length

        input_ids = torch.cat([item['input_ids'] for item in examples], dim=0)
        attention_masks = torch.cat([item['attention_mask'] for item in examples], dim=0)
        labels = input_ids.clone().detach()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }


class FinetuningDataset(Dataset):
    def __init__(self, tokenizer: PacketTokenizer, dataset_path: str, target_class: str, target_column: str = None):
        assert os.path.isfile(dataset_path)

        dataset_path = pathlib.Path(dataset_path)
        self.source_file = dataset_path
        logger.info("initializing dataset from %s with '%s' target class", dataset_path, target_class)

        self.tokenizer = tokenizer

        self.target_column = TARGET_CLASS_COLUMN if target_column is None else target_column

        raw_flows = pd.read_csv(self.source_file,
                                usecols=self.tokenizer.packet_quantizer.raw_columns + [self.target_column])
        raw_flows = raw_flows[raw_flows.loc[:, self.target_column] == target_class]

        self.raw_flows = raw_flows.loc[:, tokenizer.packet_quantizer.raw_columns].values
        logger.info('initialized dataset')

    def __len__(self):
        return len(self.raw_flows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer.batch_encode_packets(self.raw_flows[i].reshape(1, -1).astype(np.float64),
                                                   add_special_tokens=True,
                                                   return_attention_mask=True).data
