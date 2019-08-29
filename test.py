#!/usr/bin/env python
import ipdb
from simple_kbqa.qa_server import QAServer

g = QAServer(checkpoint="../chatbot_data/kbqa/elsa/checkpoint",
             w2v_word2idx="/home/pzhu/data/word2vec/en/bert-base-uncased.lookup",
             w2v_idx2vec="/home/pzhu/data/word2vec/en/bert-base-uncased.h5py")

print(g("who is the father of elsa's brother?"))

#print(g(node1="elsa", relation="father"))
#print(g(node1="elsa", node2="alice"))
