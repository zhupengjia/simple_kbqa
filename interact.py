#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='Simple KBQA')
parser.add_argument('-p', '--port', dest='port', default=5003, help="listen port, default is 5002")
parser.add_argument('--backend', dest='backend', default='shell', help="choose for backend from: shell, restful, default is shell")
parser.add_argument('--checkpoint', dest='checkpoint', required=True, help="checkpoint file of pretrained knowledge graph embedding")
parser.add_argument('--w2v_word2idx', dest='w2v_word2idx', required=True, help="word embedding idx mapping file")
parser.add_argument('--w2v_idx2vec', dest='w2v_idx2vec', required=True, help="word embedding weight hdf5 file")
parser.add_argument('--score_tolerate', dest='score_tolerate', default=0.05, help="score tolerate used for graph searching")
parser.add_argument('--min_score', dest='min_score', default=0.4, help="minimum score for graph searching")
parser.add_argument('--min_sim', dest='min_sim', default=0.7, help="minimum similarity for name matching")
parser.add_argument('--fallback', dest='fallback_reply', default=None, help="fallback reply if no answer")

args = parser.parse_args()

from simple_kbqa.backend import Backend

s = Backend(backend_type=args.backend,
            checkpoint=args.checkpoint,
            w2v_word2idx=args.w2v_word2idx,
            w2v_idx2vec=args.w2v_idx2vec,
            score_tolerate=args.score_tolerate,
            min_score=args.min_score,
            min_sim=args.min_sim,
            fallback_reply=args.fallback_reply,
            port=args.port
            )
s.run()

