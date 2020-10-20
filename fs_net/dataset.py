from typing import Tuple

import numpy as np
import torch

from gpt_model.classifier.dataset import ClassificationQuantizedDataset


class SimpleClassificationQuantizedDataset(ClassificationQuantizedDataset):
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_flow = self.tokenizer.batch_encode_packets(self.raw_flows[i].reshape(1, -1).astype(np.float64),
                                                       add_special_tokens=False,
                                                       return_attention_mask=False).data
        X = enc_flow['input_ids']
        y = torch.as_tensor(self.targets[i], dtype=torch.long)
        return X, y
