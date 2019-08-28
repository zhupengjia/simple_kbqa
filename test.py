#!/usr/bin/env python
import ipdb
from simple_kbqa.graph_search import GraphSearch

g = GraphSearch("../chatbot_data/kbqa/elsa/checkpoint", "/home/pzhu/data/word2vec/en/bert-base-uncased.lookup", "/home/pzhu/data/word2vec/en/bert-base-uncased.h5py")

print(g(node1="elsa", relation="father"))
print(g(node1="elsa", node2="alice"))
ipdb.set_trace()
