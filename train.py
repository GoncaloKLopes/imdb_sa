import pickle
import os
import settings as s
import time

import torch
import torchtext

import torch.optim as O
import torch.nn as nn

from gensim.models import KeyedVectors

from model import BinarySARNN
from utils import dir_to_csv, tokenize, binary_accuracy


train_files_path = os.path.join(s.DATASET_PATH, s.TRAIN_DIR)
train_dataset_path = os.path.join(s.DATA_DIR, s.TRAIN_CSV)
embeddings_path = os.path.join(s.DATASET_PATH, s.EMBEDDINGS_FILE)
vocab_path = os.path.join(s.DATA_DIR, s.VOCAB_FILE)

model_config = s.MODEL_CONFIG
train_config = s.TRAIN_CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create train set
dir_to_csv(s.TRAIN_CSV, train_files_path)

REVIEW = torchtext.data.Field(tokenize=tokenize, lower=True)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

print("Loading training set...")
train = torchtext.data.TabularDataset(path=train_dataset_path, format="CSV",
                                      fields=[("review", REVIEW),
                                              ("label", LABEL)])
train_iter = torchtext.data.Iterator(train, model_config.batch_size, device)

REVIEW.build_vocab(train)
print("Training set successfully loaded.\n")

# load word embeddings
print("Loading word embeddings...")
embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=False,
                                               unicode_errors='ignore')
print("Word embeddings successfully loaded.\n")

vocab = embeddings.vocab
if not os.path.isfile(vocab_path):
    print("Vocabulary file not present, creating...")
    with open(vocab_path, "wb+") as vf:
        pickle.dump(vocab, vf)
    print("Done.\n")
else:
    print("vocabulary file already exists, skipping creation...\n")

# set model configurations
model_config.vocab_size = len(embeddings.vocab)
model_config.d_embed = embeddings.vector_size

model = BinarySARNN(model_config)
model.embed.weight.data.copy_(torch.from_numpy(embeddings.vectors))
model.to(device)


# set training configuration
criterion = train_config.criterion()
criterion.to(device)

opt = train_config.optimizer(model.parameters(), **train_config.o_kwargs)

print(str(model_config) + "\n")
print(str(train_config) + "\n")

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss     Accuracy'
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f}'.split(','))
print(header)

for epoch in range(train_config.epochs):
    train_iter.init_epoch()

    for batch_idx, batch in enumerate(train_iter):
        model.train()
        opt.zero_grad()

        iterations += 1

        answer = model(batch)

        train_acc = binary_accuracy(answer, batch.label)
        print(train_acc)
        break
    break
