import csv
import os
import settings as s
import spacy
import torch

from torchtext import data


class ModelConfig:
    """Encapsulates model configuration."""
    def __init__(self, id, d_hidden, vocab_size, d_embed,
                 batch_size, n_layers, nonlin, dropout, bidir):
        self.id = id
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.dropout = dropout
        self.bidir = bidir

    def __str__(self):
        return "Model Config\n" + 12 * ("=") + "\n" +\
               "d_hidden = " + str(self.d_hidden) + "\n" +\
               "vocab_size = " + str(self.vocab_size) + "\n" +\
               "d_embed = " + str(self.d_embed) + "\n" +\
               "batch_size = " + str(self.batch_size) + "\n" +\
               "n_layers = " + str(self.n_layers) + "\n" +\
               "nonlin = " + str(self.nonlin) + "\n" +\
               "dropout = " + str(self.dropout) + "\n" +\
               "bidir = " + str(self.bidir)


class TrainConfig:
    """Encapsulates train configuration."""
    def __init__(self, id, criterion, optimizer, epochs, o_kwargs={}):
        self.id = id
        self.criterion = criterion
        self.optimizer = optimizer
        self.o_kwargs = o_kwargs
        self.epochs = epochs

    def __str__(self):
        return "Train Config\n" + 13*("=") + "\n" +\
               "criterion = " + str(self.criterion) + "\n" +\
               "optimizer = " + str(self.optimizer) + "\n" +\
               "optimizer args = " + str(self.o_kwargs) + "\n" +\
               "epochs = " + str(self.epochs)


def tokenize(text):
    """Tokenizes a piece of text."""
    return [tok.text for tok in s.SPACY_EN.tokenizer(text)]


def dir_to_csv(csv_fname, dir_paths):
    """Creates a .csv from directories of records."""
    pos_dirs = [os.path.join(_dir, s.POS_DIR) for _dir in dir_paths]
    neg_dirs = [os.path.join(_dir, s.NEG_DIR) for _dir in dir_paths]

    files = [(os.path.join(_dir, _file), 1) for _dir in pos_dirs for _file in os.listdir(_dir)]
    files.extend([(os.path.join(_dir, _file), 0) for _dir in neg_dirs for _file in os.listdir(_dir)])

    data_dir = os.path.join(os.getcwd(), s.DATA_DIR)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        csv_fname = os.path.join(data_dir, csv_fname)

        print("Creating ", csv_fname, "...")
        i = 0
        with open(csv_fname, "w+") as file_csv:
            writer = csv.writer(file_csv,
                                delimiter=" ",
                                quoting=csv.QUOTE_NONNUMERIC)
            for _file in files:
                with open(_file[0], "r") as f:
                    writer.writerow([f.read(), _file[1]])
                    i += 1
    else:
        print("Directory already exists, skipping creation...")


def binary_accuracy(preds, y):
    """Returns accuracy for a batch of predictions."""

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def epoch_time(start_time, end_time):
    """Calculates the elapsed minutes and seconds given a start and end time."""

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
