#!/bin/bash

DATA_DIR=~/conversational-journal/data/reranking_BERT_input/$1
DEV_FILE=$1"_bert4msmarco.txt"

QREL=~/conversational-journal/data/datasets/eval_topics.qrel

python3 ~/dl4marco-bert/convert_CAST_to_tfrecord.py \
                --output_folder=${DATA_DIR}/tfrecord \
                 --vocab_file=~/dl4marco-bert/data/msmarco/bertbase/vocab.txt \
                 --train_dataset_path=dummy1 \
                 --dev_dataset_path=${DATA_DIR}/${DEV_FILE} \
                 --eval_dataset_path=dummy2 \
                 --dev_qrels_path=${QREL} \
                 --max_query_length=64\
                 --max_seq_length=512 \
                 --num_eval_docs=1000