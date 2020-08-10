import bisect
import collections
import json
import pathlib
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy, PaddingStrategy, TensorType, BatchEncoding

from settings import logger
from pretraining.quantizer import PacketQuantizer


class PacketTokenizer(PreTrainedTokenizerBase):
    max_model_input_sizes = 128
    model_input_names = ["attention_mask"]

    def __init__(self,
                 packet_quantizer: PacketQuantizer,
                 unk_token="[UNK]",
                 bos_token="[BOF]",
                 eos_token="[EOF]",
                 pad_token="[PAD]",
                 **kwargs
                 ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
        self.packet_quantizer = packet_quantizer
        self.cluster_num = packet_quantizer.n_clusters
        # special token ids are inserted after all packet clusters (that start at 0)
        self.ids_to_tokens = collections.OrderedDict([(ids + self.cluster_num, tok)
                                                      for ids, tok in enumerate(self.all_special_tokens)])

        self.tokens_to_ids = {v: k for k, v in self.ids_to_tokens.items()}

        logger.info('initialized PacketTokenizer')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, flow_size=None):
        path_dir = pathlib.Path(pretrained_model_name_or_path)
        flow_size = cls.max_model_input_sizes if flow_size is None else flow_size
        quantizer = PacketQuantizer.from_checkpoint(path_dir, flow_size=flow_size)
        return cls(packet_quantizer=quantizer)

    def save_pretrained(self, save_directory):
        self.packet_quantizer.save_checkpoint(save_directory)

    def convert_ids_to_tokens(self, index):
        if isinstance(index, str):
            # exception indicates the bug
            return self.ids_to_tokens[index]
        else:
            raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.tokens_to_ids[tokens]
        else:
            raise NotImplementedError

    def _expand_with_special_tokens(self, flow: np.ndarray) -> np.ndarray:
        # truncate to account for the tokens
        flow = flow[:self.max_model_input_sizes - 2]
        flow = np.insert(flow, 0, self.bos_token_id)
        non_packets_mask = flow == self.packet_quantizer.non_packet_value
        flow[non_packets_mask] = self.pad_token_id
        # we either pick index of the first True value or append
        end_of_flow = non_packets_mask.argmax() if (non_packets_mask).any() else len(flow)
        flow = np.insert(flow, end_of_flow, self.eos_token_id)
        return flow

    def batch_encode_plus(
            self,
            flows: Union[pd.DataFrame, np.ndarray],
            add_special_tokens: bool = True,
            return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
            return_attention_mask: Optional[bool] = True,
            **kwargs
    ) -> BatchEncoding:

        if isinstance(flows, pd.DataFrame):
            flows = flows.values

        if flows.shape[1]//2 != self.max_model_input_sizes:
            logger.debug(f'input number of features ({flows.shape[1]//2}) does not match '
                           f'max_model_input_sizes ({self.max_model_input_sizes})')
        clusters = self.packet_quantizer.transform(flows)

        if add_special_tokens:
            clusters = np.apply_along_axis(self._expand_with_special_tokens, axis=1, arr=clusters)

        result = {'input_ids': clusters.astype(np.int64)}

        if return_attention_mask:
            token_mask = (clusters != self.pad_token_id).astype(np.int64)
            result.update({'attention_mask': token_mask})

        return BatchEncoding(result, tensor_type=TensorType(return_tensors), prepend_batch_axis=False)

    def _remove_special_tokens(self, flow):
        # rm bos token
        flow = flow[1:]
        # replace pad token with quantizer's non packet value for consistency
        unk_values = flow == self.pad_token_id
        flow[unk_values] = self.packet_quantizer.non_packet_value
        flow = np.delete(flow, np.where(flow == self.eos_token_id))
        return flow

    def batch_decode(self, tokenized_flows, **kwargs) -> np.ndarray:
        if isinstance(tokenized_flows, torch.Tensor):
            tokenized_flows = tokenized_flows.numpy()
        clusters_only = np.apply_along_axis(self._remove_special_tokens, axis=1, arr=tokenized_flows)
        packet_features = self.packet_quantizer.inverse_transform(clusters_only)
        return packet_features

    def __len__(self):
        return self.cluster_num + len(self.tokens_to_ids)

    @property
    def max_len(self):
        return self.max_model_input_sizes
