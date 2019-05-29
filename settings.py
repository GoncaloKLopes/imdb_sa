import os
import spacy

SPACY_EN = spacy.load("en")

DATESET_PATH = "/home/goncalo/datasets/aclImdb"
TRAIN_DIR = "train"
TEST_DIR = "test"
POS_DIR = "pos"
NEG_DIR = "neg"

TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
VAL_CSV = "val.csv"
