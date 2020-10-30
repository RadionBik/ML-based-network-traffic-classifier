import logging

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import GPT2Model, GPT2Config
from transformers.optimization import AdamW

from nn_classifiers.models import BaseClassifier

logger = logging.getLogger(__name__)


class GPT2Classifier(BaseClassifier):
    def __init__(
            self,
            config,
            class_labels,
            pretrained_model_path,
            dropout=0.1,
            freeze_pretrained_part=True,
            reinitialize=False,
            n_layers=6,
    ):
        super().__init__(config, class_labels)

        if reinitialize:
            logger.info('resetting model weights')
            config = GPT2Config.from_json_file(pretrained_model_path + '/config.json')
            config = config.to_dict()
            config['n_layer'] = n_layers
            config = GPT2Config.from_dict(config)
            self.gpt2 = GPT2Model(config)
        else:
            self.gpt2 = GPT2Model.from_pretrained(pretrained_model_path)

        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(self.gpt2.config.n_embd, self.output_dim)
        if freeze_pretrained_part:
            for param in self.gpt2.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.gpt2(**x)
        output = output[0]  # last hidden state (batch_size, sequence_length, hidden_size)
        # average over temporal dimension
        output = output.mean(dim=1)
        output = self.dropout(output)
        return self.fc(output)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams.es_patience // 2)
        return [optimizer], [scheduler]
