"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow.compat.v1 as tf

import module.coding as C
import module.sequential as S
from model.Base import Sequential, FeedForward, layernorm


class SASRec(Sequential):
    """The implementation of ---
        Kang WC, McAuley J.
        Self-attentive sequential recommendation.
        ICDM 2018.
    """

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)

        with tf.variable_scope("SASREC"):
            self.item_embs = C.Embedding(num_items, self.num_units, self.l2_reg,
                                         zero_pad=True, scale=True, scope="item_embs")
            self.pcoding = C.PositionCoding(self.seqslen, self.num_units, self.l2_reg, scope="spatial_embs")

            self.output_bias = self.output_bias(inf_pad=True)

            self.list_attention = list()
            self.list_dense = list()
            for i in range(FLAGS.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    attention = S.MultiHeadAttention(self.num_units, self.num_heads, self.attention_probs_dropout_rate)
                    fforward = FeedForward([self.num_units, self.num_units], self.hidden_dropout_rate)
                    self.list_attention.append(attention)
                    self.list_dense.append(fforward)

    def __call__(self, features, is_training):
        seqs_id = features['seqs_i']

        # positional encoding
        seqs_units = self.item_embs(seqs_id)
        seqs_units = self.pcoding(seqs_units)

        # Dropout
        with tf.variable_scope("input_transform"):
            seqs_units = tf.layers.dropout(seqs_units, rate=self.hidden_dropout_rate,
                                           training=tf.convert_to_tensor(is_training))
            seqs_masks = tf.expand_dims(tf.to_float(tf.not_equal(seqs_id, 0)), -1)

        # multi-head attention
        seqs_outs = seqs_units * seqs_masks
        for i, (attention, dense) in enumerate(zip(self.list_attention, self.list_dense)):
            with tf.variable_scope("num_blocks_%d" % i):
                with tf.variable_scope("attention"):
                    seqs_outs = attention(layernorm(seqs_outs), seqs_outs, is_training, causality=True)

                with tf.variable_scope("feedforward"):
                    seqs_outs = dense(layernorm(seqs_outs), is_training)
                    seqs_outs *= seqs_masks

        with tf.variable_scope("output_ln"):
            seqs_outs = layernorm(seqs_outs)

        if is_training:
            seqs_outs = tf.reshape(seqs_outs, [tf.shape(seqs_id)[0] * self.seqslen, self.num_units])
        else:
            # only using the latest representations for making predictions
            seqs_outs = tf.reshape(seqs_outs[:, -1], [tf.shape(seqs_id)[0], self.num_units])

        # compute logits
        logits = tf.matmul(seqs_outs, self.item_embs.lookup_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        return logits
