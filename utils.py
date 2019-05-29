import csv
import os
import progressbar
import spacy

from progressbar import ProgressBar, Bar, Percentage
from settings import DATESET_PATH, TRAIN_DIR, TEST_DIR, POS_DIR, NEG_DIR,\
                     TRAIN_CSV, TEST_CSV, VAL_CSV
from torchtext import data


def tokenize(text):
    """Tokenizes a piece of text."""
    return [tok.text for tok in SPACY_EN.tokenizer(text)]


def to_csv():
    """Creates .csv files for the training, test and validation sets."""

    train_dir = os.path.join(DATESET_PATH, TRAIN_DIR)
    test_dir = os.path.join(DATESET_PATH, TEST_DIR)
    train_pos_dir = os.path.join(DATESET_PATH, TRAIN_DIR, POS_DIR)
    train_neg_dir = os.path.join(DATESET_PATH, TRAIN_DIR, NEG_DIR)
    test_pos_dir = os.path.join(DATESET_PATH, TEST_DIR, POS_DIR)
    test_neg_dir = os.path.join(DATESET_PATH, TEST_DIR, NEG_DIR)

    # training set
    dir_to_csv(TRAIN_CSV, train_dir)
    # test set
    dir_to_csv(TEST_CSV, test_pos_dir, 0)


def dir_to_csv(csv_fname, dir_path):
    """Creates a .csv from a directory of records."""

    pos_dir = os.path.join(dir_path, POS_DIR)
    neg_dir = os.path.join(dir_path, NEG_DIR)

    files = [os.path.join(pos_dir, file) for file in os.listdir(pos_dir)]
    files.extend([os.path.join(neg_dir, file) for file in os.listdir(neg_dir)])

    print("Creating ", csv_fname, "...")
    bar = ProgressBar(maxval=len(files),
                      widgets=[Bar('=', '[', ']'), ' ', Percentage()])
    bar.start()
    i = 0
    with open(csv_fname, "w+") as file_csv:
        writer = csv.writer(file_csv,
                            delimiter=" ",
                            quoting=csv.QUOTE_NONNUMERIC)
        for file in files:
            with open(os.path.join(dir_path, file), "r") as f:
                writer.writerow([f.read(), label])
                i += 1
                bar.update(i)
    bar.finish()
