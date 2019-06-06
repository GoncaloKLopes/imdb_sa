import os
import spacy

from configs import *

SPACY_EN = spacy.load("en")

DATASET_PATH = "/home/goncalo/datasets/aclImdb"
TRAIN_DIR = "train"
TEST_DIR = "test"
POS_DIR = "pos"
NEG_DIR = "neg"

TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
VAL_CSV = "val.csv"

DATA_DIR = "data"

EMBEDDINGS_FILE = "vectors_aclImdb.txt"
VOCAB_FILE = "imdb_vocab.pickle"

# Which RNN configuration to use
MODEL_CONFIG = RNN_CONFIG1
# Which training configuration to use
TRAIN_CONFIG = TRAIN_CONFIG1
