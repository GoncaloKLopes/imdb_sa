import torch
import torch.nn as nn


class BinarySARNN(nn.Module):

    def __init__(self, config):
        super(BinarySARNN, self).__init__()
        self.hidden_dim = config.d_hidden
        self.vocab_size = config.vocab_size
        self.embed_dim = config.d_embed
        self.batch_size = config.batch_size
        self.num_labels = 2
        self.num_layers = config.n_layers
        if config.nonlin not in ("tanh", "relu"):
            raise ValueError("Invalid activation function", config.nonlin,
                             ". Expected \"tanh\" or \"relu\".")
        else:
            self.nonlin = config.nonlin
        self.dropout = config.dropout
        self.bidir = config.bidir
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.rnn = nn.RNN(input_size=self.embed_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, dropout=self.dropout,
                          nonlinearity=self.nonlin, bidirectional=self.bidir)
        self.hidden_to_label = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, batch):
        # shape -> [batch_size, sentence_len] TODO CHECK THIS

        embeddings = self.embed(batch)
        # shape -> [batch_size, sentence_len, embed_dim] TODO CHECK THIS

        rnn_out, self.hidden = self.rnn(embeddings)
        # shape -> [batch_size, hidden_dim] TODO CHECK THIS

        logits = self.hidden_to_label(self.hidden[-1])
        return logits
