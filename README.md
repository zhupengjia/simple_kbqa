# simple_kbqa

A simple kbqa based on knowledge graph embedding

* Usage:

    - Train knowledge graph embedding

        ```shell
        git clone git@github.com:zhupengjia/KnowledgeGraphEmbedding.git
        cd KnowledgeGraphEmbedding/codes
        ./run2.py --data_path=$DATA_PATH --model=RotatE --save_path=$SAVE_PATH --double_entity_embedding --cuda --hidden_dim=50
        ```

