import csv
import os
import settings as s
import spacy

from torchtext import data


class ModelConfig:
    """Encapsulates model configuration."""
    def __init__(self, d_hidden, vocab_size, d_embed,
                 batch_size, n_layers, nonlin, dropout, bidir):
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
    def __init__(self, criterion, optimizer, epochs, o_kwargs={}):
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


def dir_to_csv(csv_fname, dir_path):
    """Creates a .csv from a directory of records."""

    pos_dir = os.path.join(dir_path, s.POS_DIR)
    neg_dir = os.path.join(dir_path, s.NEG_DIR)
    data_dir = os.path.join(os.getcwd(), s.DATA_DIR)

    files = [(os.path.join(pos_dir, file), 1)
             for file in os.listdir(pos_dir)]
    # add negative examples to our files list
    files.extend([(os.path.join(neg_dir, file), 0)
                  for file in os.listdir(neg_dir)])
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        csv_fname = os.path.join(data_dir, csv_fname)

        print("Creating ", csv_fname, "...")
        i = 0
        with open(csv_fname, "w+") as file_csv:
            writer = csv.writer(file_csv,
                                delimiter=" ",
                                quoting=csv.QUOTE_NONNUMERIC)
            for file in files:
                with open(os.path.join(dir_path, file[0]), "r") as f:
                    writer.writerow([f.read(), file[1]])
                    i += 1
    else:
        print("Directory already exists, skipping creation...")


def binary_accuracy(preds, y):
    """Returns accuracy for a batch of predictions."""

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
