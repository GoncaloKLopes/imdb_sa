import torch.nn as nn
import torch.optim as O
from utils import ModelConfig, TrainConfig

RNN_CONFIG1 = ModelConfig(id="simple_rnn", d_hidden=64, vocab_size=None, d_embed=None,
                          batch_size=64, n_layers=1, nonlin="tanh",
                          dropout=0, bidir=False)

RNN_CONFIG2 = ModelConfig(id="simple_rnn_2layers", d_hidden=128, vocab_size=None, d_embed=None,
                          batch_size=128, n_layers=2, nonlin="tanh",
                          dropout=0, bidir=False)

RNN_CONFIG3 = ModelConfig(id="simples_rnn_2layers_bidir", d_hidden=128, vocab_size=None, d_embed=None,
                          batch_size=128, n_layers=2, nonlin="tanh",
                          dropout=0, bidir=True)

TRAIN_CONFIG1 = TrainConfig(id="sgd", criterion=nn.CrossEntropyLoss,
                            optimizer=O.SGD, epochs=5,
                            o_kwargs={"lr": 0.001})
TRAIN_CONFIG2 = TrainConfig(id="adadelta", criterion=nn.CrossEntropyLoss,
                            optimizer=O.Adadelta, epochs=5)
TRAIN_CONFIG3 = TrainConfig(id="adagrad", criterion=nn.CrossEntropyLoss,
                            optimizer=O.Adagrad, epochs=5)
TRAIN_CONFIG4 = TrainConfig(id="adam", criterion=nn.CrossEntropyLoss,
                            optimizer=O.Adam, epochs=5)
TRAIN_CONFIG5 = TrainConfig(id="rmsprop", criterion=nn.CrossEntropyLoss,
                            optimizer=O.RMSprop, epochs=5)
