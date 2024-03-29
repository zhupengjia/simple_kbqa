#!/usr/bin/env python
import os, re, spacy, numpy, torch
from bidict import bidict
from sklearn.metrics.pairwise import cosine_distances
from nlptools.text.tokenizer import Tokenizer_BERT
from nlptools.text.embedding import Embedding_File
from nlptools.text import TFIDF
from nlptools.text.docsim import WMDSim

class GraphSearch:
    """
        Graph search via graph embedding
    """
    def __init__(self, checkpoint, w2v_word2idx, w2v_idx2vec, score_tolerate=0.05, min_score=0.4, min_sim=0.7, **args):
        """
            Input:
                - checkpoint: 
                - w2v_word2idx: pickle dump for word-idx mapping
                - w2v_idx2vec: hdf5 dump for idx-vector mapping
                - score_tolerate: float, will return all results in score tolerate range, default is 0.05
                - min_score: minimum score for related query
                - min_sim: minimum similarity to get synonym from word embedding
        """
        pi = 3.14159265358979323846
        checkpoint = torch.load(checkpoint, map_location={"cuda:0":"cpu"})
        embedding_range = checkpoint["model_state_dict"]["embedding_range"].item()
        self.graph_embedding = {
            "node": checkpoint["model_state_dict"]["entity_embedding"].numpy(),
            "relation": checkpoint["model_state_dict"]["relation_embedding"].numpy()/(embedding_range/pi)
        }
        self.id_map = {
            "node": bidict(checkpoint["entity2id"]),
            "relation": bidict(checkpoint["relation2id"])
        }
        self.embedding_dim = self.graph_embedding["node"].shape[1]//2
        self.tokenizer = Tokenizer_BERT(bert_model_name="bert-base-uncased", do_lower_case=True)
        self.vocab = self.tokenizer.vocab
        self.vocab.embedding = Embedding_File(w2v_word2idx, w2v_idx2vec)
        self.score_tolerate = score_tolerate
        self.min_score = min_score
        self.min_sim = min_sim
        self._build_index()

    def _build_index(self):
        self.index = {}
        self.word_ids = {}
        for n in ["node", "relation"]:
            self.index[n] = TFIDF(self.vocab.vocab_size)
            names = [""]*(max(self.id_map[n].inv.keys())+1)
            for k in self.id_map[n].inv:
                names[k] = self.tokenizer(self.id_map[n].inv[k])
            self.word_ids[n] = [self.vocab(n) for n in names]
            self.index[n].load_index(self.word_ids[n])
        self.similarity = WMDSim(vocab=self.vocab)

    def get_id(self, name, idtype="node"):
        """
            get node/relation id from name

            Input:
                - name: string
                - idtype: "node" or "relation"
        """
        name_ids = self.vocab(self.tokenizer(name))
        prefilter = 100
        if len(self.word_ids[idtype]) > prefilter:
            ids = [x[0] for x in self.index[idtype].search(name_ids, topN=prefilter)]

        else:
            ids = list(range(len(self.word_ids[idtype])))
        distances = [self.similarity.rwmd_distance(name_ids, self.word_ids[idtype][i]) for i in ids]
        min_id = numpy.argmin(distances)
        min_dis = distances[min_id]
        max_sim = 1/(1+min_dis)
        if max_sim <= self.min_sim :
            return None
        return ids[min_id]

    def _get_close(self, embedding, idtype="node"):
        """
            get most closed id
        """
        distances = cosine_distances([embedding], self.graph_embedding[idtype])[0]
        similarities = 1/(1+distances)
        ids = numpy.arange(len(similarities))
        max_sim = similarities.max()
        if max_sim < self.min_score:
            return None
        sim_filter = similarities >= max(self.min_score, max_sim-self.score_tolerate)
        ids = ids[sim_filter]
        similarities = similarities[sim_filter]
        ids2 = numpy.argsort(ids)[::-1]
        ids = ids[ids2]
        similarities = similarities[ids2]
        names = [self.id_map[idtype].inv[x] for x in ids]
        return list(zip(names, similarities))

    def __call__(self, node1=None, node2=None, relation=None):
        """
            Get node or relation from other node or relation

            Input:
                - node1: string
                - node2: string
                - relation: string
        """

        if node1 and relation:
            # get node2 from node1 and relationship
            node1_id = self.get_id(node1, "node")
            relation_id = self.get_id(relation, "relation")
            if node1_id is None or relation_id is None:
                return None
            node1_embedding = self.graph_embedding['node'][node1_id]
            phase_relation = self.graph_embedding['relation'][relation_id]

            re_node1 = node1_embedding[:self.embedding_dim]
            im_node1 = node1_embedding[self.embedding_dim:]
            re_relation = numpy.cos(phase_relation)
            im_relation = numpy.sin(phase_relation)

            re_node2 = re_node1 * re_relation - im_node1 * im_relation
            im_node2 = re_node1 * im_relation + im_node1 * re_relation
            node2_embedding = numpy.concatenate((re_node2, im_node2))
            return self._get_close(node2_embedding, "node")

        elif node2 and relation:
            # get node1 from node2 and relationship
            node2_id = self.get_id(node2, "node")
            relation_id = self.get_id(relation, "relation")
            if node2_id is None or relation_id is None:
                return None
            node2_embedding = self.graph_embedding['node'][node2_id]
            phase_relation = self.graph_embedding['relation'][relation_id]

            re_node2 = node2_embedding[:self.embedding_dim]
            im_node2 = node2_embedding[self.embedding_dim:]
            re_relation = numpy.cos(phase_relation)
            im_relation = numpy.sin(phase_relation)
           
            re_node1 = re_relation * re_node2 + im_relation * im_node2
            im_node1 = re_relation * im_node2 - im_relation * re_node2
            node1_embedding = numpy.concatenate((re_node1, im_node1))
            return self._get_close(node1_embedding, "node")

        elif node1 and node2:
            # get relationship between two nodes
            node1_id = self.get_id(node1, "node")
            node2_id = self.get_id(node2, "node")
            if node1_id is None or node2_id is None:
                return None
            node1_embedding = self.graph_embedding['node'][node1_id]
            node2_embedding = self.graph_embedding['node'][node2_id]

            re_node1 = node1_embedding[:self.embedding_dim]
            im_node1 = node1_embedding[self.embedding_dim:]
            re_node2 = node2_embedding[:self.embedding_dim]
            im_node2 = node2_embedding[self.embedding_dim:]
            
            re_relation = re_node1 * re_node2 + im_node1 * im_node2
            im_relation = im_node2 * re_node1 - re_node2 * im_node1

            relation_embedding = numpy.arctan2(im_relation, re_relation)
            return self._get_close(relation_embedding, "relation")

        else:
            raise("Must at least have two of node1, node2 and relation")
            

