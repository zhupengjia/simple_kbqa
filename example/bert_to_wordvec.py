#!/usr/bin/env python
import sys, os, torch, h5py
from pytorch_transformers import BertModel, BertTokenizer
from nlptools.utils import zdump

model_path = sys.argv[1] if len(sys.argv) > 1 else "."
model_name = "bert-base-uncased"
vocab_name = os.path.join(model_path, 'vocab')
weight_path = os.path.join(model_path, '{}.h5py'.format(model_name))
word2idx_path = os.path.join(model_path, '{}.lookup'.format(model_name))

model = BertModel.from_pretrained(model_name)

weights = model.embeddings.word_embeddings.weight.detach().numpy()

tokenizer = BertTokenizer.from_pretrained(model_name)
word2idx = tokenizer.vocab

print(weights.shape)
print(len(tokenizer.vocab))

if os.path.exists(weight_path):
    os.remove(weight_path)

with h5py.File(weight_path, 'w') as h5file:
    h5file.create_dataset("word2vec", data=weights)

zdump(word2idx, word2idx_path)


