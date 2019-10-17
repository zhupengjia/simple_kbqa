FROM debian:buster-slim
MAINTAINER Pengjia Zhu <zhupengjia@gmail.com>

EXPOSE 5000/tcp

ENV WORKDIR /opt/chatbot
ENV BUILD_VERSION b1.0

RUN apt-get update &&\
    apt-get install -y --no-install-recommends python3-numpy python3-scipy python3-pandas python3-wheel python3-pip python3-sklearn python3-xlrd python3-tqdm python3-h5py python3-yaml python3-nltk python3-flask python3-setuptools python3-regex python3-requests python3-boto3 cython3 git&&\
    ln -s /usr/bin/python3 /usr/local/bin/python &&\
    ln -s /usr/bin/pip3 /usr/local/bin/pip &&\
    rm -rf /var/lib/apt/lists/*

# python packages
RUN pip3 install --no-cache bidict spacy&&\
    python3 -m spacy download en &&\
    python3 -c "import nltk;nltk.download('wordnet')" &&\
    pip3 install --no-cache torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install --no-cache transformers &&\
    pip3 install --no-cache git+git://github.com/zhupengjia/nlptools@master 

WORKDIR $WORKDIR

ENV BOT_VERSION a0.2

ADD simple_kbqa simple_kbqa

ADD interact.py interact.py

CMD python interact.py --checkpoint ${BOT_CHECKPOINT} --w2v_word2idx ${W2V_WORD2IDX} --w2v_idx2vec ${W2V_IDX2VEC} --backend restful --port 5000
#CMD tail -f /dev/null
