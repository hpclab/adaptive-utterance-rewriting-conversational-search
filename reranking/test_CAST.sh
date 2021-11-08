#!/bin/bash

DATA_DIR=~/conversational-journal/data/reranking_BERT_input/$1
MODEL_DIR=~/dl4marco-bert/data/msmarco/bertbase

python3 run_msmarco_CAST.py \
  --data_dir=${DATA_DIR}/tfrecord \
  --bert_config_file=${MODEL_DIR}/bert_config.json \
  --init_checkpoint=${MODEL_DIR}/model.ckpt-100000 \
  --output_dir=${DATA_DIR}/output \
  --msmarco_output=True \
  --do_train=False \
  --do_eval=True \
  --num_train_steps=400000 \
  --num_warmup_steps=40000 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=1e-6

exit 0