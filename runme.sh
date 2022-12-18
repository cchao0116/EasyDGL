#!/usr/bin/env bash
source activate py37_tf115

export PYTHONPATH=./src  # the path of the src folder
DATA_HOME=./Koubei  # the path of the data folder

# train -> the TFRECORD files
# test  -> the TFRECORD files
# mark  -> the pickly file which dump scipy.sparse.csr_matrix object
# num_items -> need to specify the number of items
# seqslen -> the maximum length of the sequence
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --train="${DATA_HOME}/train???.tfrec" \
    --test=${DATA_HOME}/test.tfrec \
    --mark=${DATA_HOME}/mark.pkl \
    --learning_rate=5e-4 --num_units=512 \
    --num_train_steps=100000 --num_warmup_steps=100 \
    --num_items=10214 --seqslen=30 --model=SMAREC --batch_size=512 --num_epochs=30

conda deactivate