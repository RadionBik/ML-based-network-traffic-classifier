import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau

from report import Reporter


class BaseClassifier(LightningModule):
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
        rpt = Reporter(targets, predictions, self.__class__.__name__)
        logs = rpt.scores()
        logs.update({'test_loss': avg_loss})
        return {'test_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]



class DenseClassifier(BaseClassifier):
    def __init__(self, input_size, output_size, hidden_size=40, activation=torch.nn.LeakyReLU, dropout=0.1):
        super().__init__()

        self.net = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),
                                       activation(),
                                       torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_size, hidden_size),
                                       activation(),
                                       torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_size, hidden_size),
                                       activation(),
                                       torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_size, output_size))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]


class BiGRUClassifier(BaseClassifier):
    def __init__(self, input_size, output_size, num_layers=3, hidden_size=None, dropout=0.1):
        super().__init__()

        if not hidden_size:
            hidden_size = output_size

        self.gru = torch.nn.GRU(input_size,
                                hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)

        self.activation = torch.nn.LeakyReLU()
        self.layer_norm = torch.nn.LayerNorm(2*hidden_size)
        self.fc = torch.nn.Linear(2*hidden_size, output_size)

    def forward(self, x):
        gru_out, hidden_state =  self.gru(x.unsqueeze_(2))
        out = self.activation(gru_out.mean(axis=1))
        out = self.layer_norm(out)
        return self.fc(out)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]
