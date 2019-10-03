FROM python:3-slim
MAINTAINER Pengjia Zhu <zhupengjia@gmail.com>

EXPOSE 5000/tcp

ENV WORKDIR /opt/chatbot
ENV BUILD_VERSION b1.0

RUN apt-get update &&\
    apt-get install -y --no-install-recommends libhdf5-dev git&&\
    rm -rf /var/lib/apt/lists/*

# python packages
RUN pip3 install --no-cache --upgrade pip setuptools wheel numpy scipy pandas&&\
    pip3 install --no-cache pyyaml bidict tqdm h5py sklearn spacy nltk xlrd librosa unidecode lws&&\
    python3 -m spacy download en &&\
    python3 -c "import nltk;nltk.download('vader_lexicon')" &&\
    pip3 install --no-cache torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install flask slixmpp python-telegram-bot &&\
    pip3 install --no-cache transformers Cython fuzzywuzzy python-Levenshtein symspellpy&&\
    pip3 install --no-cache git+git://github.com/zhupengjia/nlptools@master &&\

#RUN pip3 install --no-cache https://f002.backblazeb2.com/file/pzhu-public/phonemizer-1.0.1-py3-none-any.whl

WORKDIR $WORKDIR

ENV BOT_VERSION a0.2

ADD simple_kbqa simple_kbqa

ADD interact.py interact.py

CMD python interact.py --checkpoint ${BOT_CHECKPOINT} --w2v_word2idx ${W2V_WORD2IDX} --w2v_idx2vec ${W2V_IDX2VEC}
#CMD tail -f /dev/null
