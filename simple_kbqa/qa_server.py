#!/usr/bin/env python
import spacy, numpy
from itertools import chain
from .graph_search import GraphSearch
from nlptools.text.ner import KeywordsMatcher
from nltk.corpus import wordnet

class QAServer:
    """
        QA server
    """
    def __init__(self, fallback_reply=None, **args):
        self.fallback_reply = fallback_reply
        self.graph_search = GraphSearch(**args)
        self._build_ner()
    
    def _find_synonym(self, sentence):
        replacements = []
        words = sentence.split()
        if len(words) > 1:
            return [] # TODO for multiple synonym replacement
        if len(words) < 1:
            return []
        synonyms = wordnet.synsets(words[0])
        lemmas = list(set(chain.from_iterable([[x.lower() for x in word.lemma_names()] for word in synonyms])))
        return lemmas

    def _build_ner(self):
        nodes = list([x.lower() for x in self.graph_search.id_map["node"].keys()])
        relations = list([x.lower() for x in self.graph_search.id_map["relation"].keys()])
        add_relations = []
        for r in relations:
            add_relations += self._find_synonym(r)
        self.tokenizer = spacy.load("en", disable=["ner", "textcat"])
        keywords = {"node": nodes, "relation": relations + add_relations}
        keywords_matcher = KeywordsMatcher(self.tokenizer, keywords)
        self.tokenizer.add_pipe(keywords_matcher, last=True)

    def __call__(self, question, session_id=None):
        doc = self.tokenizer(question)
        root = [token for token in doc if token.head == token]
        if len(root) < 1:
            return self.fallback_reply, 0
        root = root[0]
        entities = []
        def treeloop(node):
            for child in node.children:
                if child.ent_type_:
                    entities.append(child)
            for child in node.children:
                treeloop(child)
        treeloop(root)
        if len(entities) < 2:
            return self.fallback_reply, 0

        entities = entities[::-1]
        a, b = None, None
        for i in range(len(entities)-1):
            if not a:
                a = (entities[i].text, entities[i].ent_type_, 1)
            else:
                a = a[0]
            b = (entities[i+1].text, entities[i+1].ent_type_)
            if a[1] == "node" and b[1] == "relation":
                result = self.graph_search(node1=a[0], relation=b[0])
                if result is None:
                    return self.fallback_reply, 0
                a = [(x[0], "node", x[1]) for x in result]
            elif a[1] == "relation"  and b[1] == "node":
                result = self.graph_search(node2=b[0], relation=a[0])
                if result is None:
                    return self.fallback_reply, 0
                a = [(x[0], "node", x[1]) for x in result]
            elif a[1] == "node" and b[1] == "node":
                result = self.graph_search(node1=a[0], node2=b[0])
                if result is None:
                    return self.fallback_reply, 0
                a = [(x[0], "relation", x[1]) for x in result]
            else:
                return self.fallback_reply, 0
        return ", ".join([x[0] for x in a]), numpy.mean([x[2] for x in a])

