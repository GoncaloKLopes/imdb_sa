import torch.nn as nn
import torch.optim as O
from utils import ModelConfig, TrainConfig

RNN_CONFIG1 = ModelConfig(d_hidden=64, vocab_size=None, d_embed=None,
                          batch_size=64, n_layers=1, nonlin="tanh",
                          dropout=0, bidir=False)

TRAIN_CONFIG1 = TrainConfig(criterion=nn.CrossEntropyLoss,
                            optimizer=O.SGD, epochs=5,
                            o_kwargs={"lr": 0.001})
TRAIN_CONFIG2 = TrainConfig(criterion=nn.CrossEntropyLoss,
                            optimizer=O.Adadelta, epochs=5)
TRAIN_CONFIG3 = TrainConfig(criterion=nn.CrossEntropyLoss,
                            optimizer=O.Adagrad, epochs=5)
TRAIN_CONFIG4 = TrainConfig(criterion=nn.CrossEntropyLoss,
                            optimizer=O.Adam, epochs=5)
TRAIN_CONFIG5 = TrainConfig(criterion=nn.CrossEntropyLoss,
                            optimizer=O.RMSprop, epochs=5)
