import torch

from nn_classifiers.models import BaseClassifier


class FSNETClassifier(BaseClassifier):
    def __init__(self,
                 config,
                 class_labels,
                 n_tokens,
                 embedding_dim=128,
                 hidden_size=128,
                 n_layers=2,
                 dropout=0.3):
        super().__init__(config, class_labels)

        self.embeddings = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.SELU()
        self.encoder = torch.nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.encoder_hidden_dim = 2 * n_layers * hidden_size
        self.compound_dim = self.encoder_hidden_dim * 4
        self.decoder = torch.nn.GRU(
            self.encoder_hidden_dim,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.compressor = torch.nn.Sequential(
            torch.nn.Linear(self.compound_dim, 2 * hidden_size),
            self.activation,
            self.dropout,
        )
        self.classifier = torch.nn.Linear(2 * hidden_size, self.output_dim)

    @staticmethod
    def _concat_hidden_states(hidden_states, batch_size):
        return hidden_states.permute([1, 0, 2]).reshape(batch_size, -1)  # (batch_size, 2*n_layers*hidden_size)

    def forward(self, x):
        encoder_in = self.embeddings(x.squeeze_(1))  # (batch_size, embedding_dim)
        batch_size, seq_len = x.shape[0], x.shape[1]

        _, enc_states = self.encoder(encoder_in)
        # "concatenate the final hidden states of both forward and backward directions of all the layers"
        z_e = self._concat_hidden_states(enc_states, batch_size)

        # "the encoder-based feature vector ze is input into the decoder at each time step t", so we just repeat it
        decoder_in = z_e.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, encoder_hidden_dim)
        _, dec_states = self.decoder(decoder_in)
        z_d = self._concat_hidden_states(dec_states, batch_size)
        # compound feature vector
        z = torch.cat([z_e, z_d, z_e * z_d, torch.abs(z_e - z_d)], dim=1)
        z_c = self.compressor(z)
        return self.classifier(z_c)
