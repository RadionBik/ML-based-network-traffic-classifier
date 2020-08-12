from typing import List

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import GPT2Model
from transformers.trainer_utils import set_seed


from report import Reporter
from settings import BASE_DIR

set_seed(1)


class BaseClassifier(LightningModule):
    def __init__(self, config, class_labels: List[str], *args, **kwargs):
        super().__init__()
        self.hparams = config
        self.class_labels = class_labels
        self.output_dim = len(class_labels)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        predictions = y_hat.max(axis=1)[1]
        loss = F.cross_entropy(y_hat, y)
        logs = {'test_loss': loss}
        return {'test_loss': loss,
                'predictions': predictions.to('cpu'),
                'targets': y.to('cpu'),
                'log': logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        predictions = torch.cat([x['predictions'] for x in outputs]).to('cpu').numpy()
        targets = torch.cat([x['targets'] for x in outputs]).to('cpu').numpy()
        rpt = Reporter(targets, predictions, self.__class__.__name__, target_classes=self.class_labels)
        self.logger.experiment.log_image('confusion_matrix', rpt.plot_conf_matrix())

        clf_report = rpt.clf_report()
        print(clf_report)
        clf_report.to_csv(BASE_DIR / 'clf_report.csv', index=True)
        self.logger.experiment.log_artifact((BASE_DIR / 'clf_report.csv').as_posix())

        logs = rpt.scores()
        logs.update({'test_loss': avg_loss})
        return {'test_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams.es_patience // 2)
        return [optimizer], [scheduler]


class DenseClassifier(BaseClassifier):
    def __init__(self, config, class_labels, input_size, hidden_size=40, activation=torch.nn.LeakyReLU, dropout=0.1):
        super().__init__(config, class_labels)

        self.net = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),
                                       activation(),
                                       torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_size, hidden_size),
                                       activation(),
                                       torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_size, hidden_size),
                                       activation(),
                                       torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_size, self.output_dim))


class BiGRUClassifier(BaseClassifier):
    def __init__(self, config, class_labels, input_size, num_layers=3, hidden_size=None, dropout=0.1, bidirectional=True):
        super().__init__(config, class_labels)

        if not hidden_size:
            hidden_size = self.output_dim

        self.gru = torch.nn.GRU(input_size,
                                hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=bidirectional)

        self.activation = torch.nn.LeakyReLU()
        gru_out_size = 2*hidden_size if bidirectional else hidden_size
        self.layer_norm = torch.nn.LayerNorm(gru_out_size)
        self.fc = torch.nn.Linear(gru_out_size, self.output_dim)

    def forward(self, x):
        gru_out, hidden_state = self.gru(x.unsqueeze_(2))
        out = self.activation(gru_out.max(axis=1)[0])
        out = self.layer_norm(out)
        return self.fc(out)


class GPT2Classifier(BaseClassifier):
    def __init__(self, config, class_labels, pretrained_model_path, dropout=0.1, freeze_pretrained_part=True):
        super().__init__(config, class_labels)

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
        from transformers.optimization import AdamW

        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams.es_patience // 2)
        return [optimizer], [scheduler]
