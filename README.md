# simple_kbqa

A simple kbqa based on knowledge graph embedding

node and relation extracted via keyword based entity recognition
multi relations extracted via part of speech with spacy
graph search use pretrained graph embedding via rotatE model

* Usage:

    - Train knowledge graph embedding

        ```shell
        git clone git@github.com:zhupengjia/KnowledgeGraphEmbedding.git
        cd KnowledgeGraphEmbedding/codes
        ./run2.py --data_path=$DATA_PATH --model=RotatE --save_path=$SAVE_PATH --double_entity_embedding --cuda --hidden_dim=50
        ```

    - Run
        
        ```shell
        ./interact.py --checkpoint=$CHECKPOINTFILE --w2v_word2idx=$WORDEMBEDDINGLOOKUPFILE --w2v_idx2vec=$WORDEMBEDDINGWEIGHTFILE --backend restful
        ```

