import os
import torch
import torchtext


from settings import *
from utils import dir_to_csv, to_csv

# create train, test and eval sets
if not os.path.isfile(TRAIN_CSV):
    to_csv()


# train

