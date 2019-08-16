#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='Simple KBQA')
parser.add_argument('-p', '--port', dest='port', default=5003, help="listen port, default is 5002")
parser.add_argument('--backend', dest='backend', default='shell', help="choose for backend from: shell, restful, default is shell")
parser.add_argument('--scorelimit', dest='scorelimit', default=0.4, help="Limitation of score, if below this number will return None")
parser.add_argument('embedding_file', help="Pretrained KB Embedding file")
parser.add_argument('--w2v_word2idx', required=True, help="Word embedding word-idx mapping file")
parser.add_argument('--w2v_idx2vec', required=True, help="Word embedding idx-vec mapping file")
args = parser.parse_args()

#from simple_kbqa.backend import Backend
from simple_kbqa.qa_server import QAServer

#s = Backend(backend_type=args.backend,
#            file_path=args.input,
#            model_path=args.model,
#            device=args.device,
#            recreate=args.recreate,
#            port=args.port,
#            score_limit=float(args.scorelimit),
#            return_relate=args.returnrelate)
#s.run()
q = QAServer(args.embedding_file, args.w2v_word2idx, args.w2v_idx2vec)
#q("Who is your father?")
q("where is your father working?")


