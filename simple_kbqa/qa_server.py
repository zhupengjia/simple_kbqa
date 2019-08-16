#!/usr/bin/env python
import os, re, spacy, numpy
from spacy.symbols import nsubj, dobj, pobj, VERB
from nlptools.utils import zload
from nlptools.text.embedding import Embedding_File
from annoy import AnnoyIndex

class AnnoyMatcher:
    name = "annoy_matcher"
    def __init__(self, nlp, embedding, mapdict):
        """
            Annoy matcher for spacy pipeline

            Input:
                - nlp: spacy instance
                - embedding: instance of nlp.text.embedding
                - mapdict: dictionary line {"key", word_id_map_dict, ...}
        """
        self.embedding = embedding
        self.annoy_index = {}
        for k in mapdict.keys():
            self.annoy_index[k] = AnnoyIndex(self.embedding.dim)
            for w, i in mapdict[k].items():
                self.annoy_index[k].add_item(i, self._word_embedding(w))
            
    def _word_embedding(self, string):
        words = [x.strip() for x in re.split("\s", string) if x.strip()]
        embeddings = [self.embedding[w] for w in words]
        return numpy.mean(embeddings, axis=0)
    
    def __call__(self, doc):
        for token in doc:
            lemma = token.lemma_
            for k in self.annoy_index:
                result = self.annoy_index[k].get_nns_by_vector(self.embedding[lemma], 1, include_distances=True)
                print(k, result, lemma)
                #keyword, similarity = result[0][0], 1/(1+result[1][0])
                #print(k, token, keyword, similarity)
        return doc


class QAServer:
    """
        QA restful server
    """
    def __init__(self, embedding_file, w2v_word2idx, w2v_idx2vec, **args):
        """
            Input:
                - embedding_file: file path of trained knowledge graph embedding
                - w2v_word2idx: pickle filepath for word-idx mapping
                - w2v_idx2vec: numpy dump for idx-vector mapping
        """
        self.graph_embedding = zload(embedding_file)
        self.tokenizer = spacy.load("en")
        self.word_embedding = Embedding_File(w2v_word2idx, w2v_idx2vec)
        entity_matcher = AnnoyMatcher(self.tokenizer,
                                      self.word_embedding, 
                                      {"entity": self.graph_embedding["entity2id"],
                                       "relation": self.graph_embedding["relation2id"]}
                                     )
        self.tokenizer.add_pipe(entity_matcher, last=True)

    def __call__(self, question, session_id=None):
        tokens = self.tokenizer(question)
        for ent in tokens.ents:
            print(ent.text, ent.label_)


        verbs = set()
        subject = set()
        for possible_subject in tokens:

            print("===", possible_subject, possible_subject.pos_, possible_subject.dep_)
            if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
                verbs.add(possible_subject.head)
                print(possible_subject, possible_subject.head)
            #if possible_subject.dep == dobj and possible_subject.head.pos == verb:
            #    #verbs.add(possible_subject.head)
            #    print(possible_subject, possible_subject.head)
            #if possible_subject.dep == pobj and possible_subject.head.pos == verb:
            #    #verbs.add(possible_subject.head)
            #    print(possible_subject, possible_subject.head)
        #print(verbs)

