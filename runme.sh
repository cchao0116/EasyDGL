#!/usr/bin/env bash
source activate py37_tf115

export PYTHONPATH=./src  # the path of the src folder
DATA_HOME=$1  # the path of the data folder


# train -> the TFRECORD files for training
# valid -> the TFRECORD files for validation
# test  -> the TFRECORD files for testing
# mark  -> the pickly file which dump scipy.sparse.csr_matrix object
# num_items -> need to specify the number of items
# seqslen -> the maximum length of the sequence

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --mark="${DATA_HOME}/mark.pkl" \
     --num_units=512 --hidden_dropout_rate=0.1 --attention_probs_dropout_rate=0.1 \
     --learning_rate=5e-4 --batch_size=512 --l2_reg=1e-4  --ct_reg=1e-7  \
     --num_items=17771 --model=EasyDGL \
     --num_blocks=1 --num_heads=8 --mask_seen --time_scale=86400

#
#   Time-independent Baselines,
#     GRU4REC, SASREC, GREC, S2PNM, BERT4REC
#

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --num_units=512 --hidden_dropout_rate=0.2 --attention_probs_dropout_rate=0.2 \
     --learning_rate=5e-5 --batch_size=512 --l2_reg=1e-4 \
     --num_items=17771 --model=BERT4REC \
     --num_blocks=3 --num_heads=8 --mask_seen

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --num_units=512 --hidden_dropout_rate=0. --attention_probs_dropout_rate=0. \
     --learning_rate=5e-4 --batch_size=512 --l2_reg=0. \
     --num_items=17771 --model=SASREC \
     --num_blocks=2 --num_heads=8 --mask_seen

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --num_units=512 --hidden_dropout_rate=0.2 --attention_probs_dropout_rate=0.2 \
     --learning_rate=1e-4 --batch_size=512 --l2_reg=1e-4 \
     --num_items=17771 --model=S2PNM \
     --num_blocks=1 --num_heads=1 --mask_seen

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --num_units=512 --hidden_dropout_rate=0.1 \
     --learning_rate=5e-4 --batch_size=512 --l2_reg=1e-4 \
     --num_items=17771 --model=GRU4REC \
     --num_blocks=1 --mask_seen

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --num_units=512 --hidden_dropout_rate=0.1 \
     --learning_rate=5e-4 --batch_size=512 --l2_reg=1e-4 \
     --num_items=17771 --model=GREC  \
     --dilations=1,4,1,4 --mask_seen

#
#   Time-conditioned Baselines,
#     TGAT, TiSASREC, TimelyREC, CTSMA
#

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --num_units=512 --hidden_dropout_rate=0.1 --attention_probs_dropout_rate=0.1 \
     --learning_rate=5e-5 --batch_size=512 --l2_reg=1e-4 \
     --num_items=17771 --model=TGAT \
     --num_blocks=3 --num_heads=1 --mask_seen --time_scale=86400

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --num_units=512 --hidden_dropout_rate=0.1 --attention_probs_dropout_rate=0.1 \
     --learning_rate=5e-4 --batch_size=512 --l2_reg=1e-4 \
     --num_items=17771 --model=TiSASREC --timelen=256\
     --num_blocks=2 --num_heads=8 --mask_seen --time_scale=86400

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --num_units=512 --hidden_dropout_rate=0.1 --attention_probs_dropout_rate=0.1 \
     --learning_rate=1e-3 --batch_size=512 --l2_reg=1e-4  \
     --num_items=17771 --model=TimelyREC \
     --num_blocks=2 --num_heads=4 --mask_seen --time_scale=86400

CUDA_VISIBLE_DEVICES=0 python src/main.py \
     --train="${DATA_HOME}/train???.tfrec" \
     --valid="${DATA_HOME}/validation.tfrec" \
     --test="${DATA_HOME}/test.tfrec" \
     --mark="${DATA_HOME}/mark.pkl" \
     --num_units=512 --hidden_dropout_rate=0.1 --attention_probs_dropout_rate=0.2 \
     --learning_rate=5e-4 --batch_size=512 --l2_reg=1e-4  --ct_reg=1e-7  \
     --num_items=17771 --model=CTSMA \
     --num_blocks=2 --num_heads=4 --mask_seen --time_scale=86400

conda deactivate