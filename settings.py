import os
import spacy

from configs import *

SPACY_EN = spacy.load("en")

DATASET_PATH = "/home/goncalo/data/aclImdb"
MODELS_PATH = "/home/jupyter/models/aclImdb"
TRAIN_DIR = "train"
TEST_DIR = "test"
POS_DIR = "pos"
NEG_DIR = "neg"

CSV = "reviews.csv"

DATA_DIR = "data"

EMBEDDINGS_FILE = "vectors_aclImdb.txt"
VOCAB_FILE = "imdb_vocab.pickle"

D_EMBEDDING = 300

# Which RNN configuration to use
MODEL_CONFIG = RNN_CONFIG3
# Which training configuration to use
TRAIN_CONFIG = TRAIN_CONFIG3
