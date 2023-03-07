"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow.compat.v1 as tf

import module.coding as C
import module.temporal as T
from model.Base import Sequential, FeedForward, layernorm


class TiSASRec(Sequential):
    """ Implementation of the paper ---
        Li J, Wang Y, McAuley J.
        Time interval aware self-attention for sequential recommendation.
        WSDM 2020.
    """

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)
        self.timelen = FLAGS.timelen
        self.time_scale = FLAGS.time_scale

        with tf.variable_scope("TiSASRec"):
            self.item_embs = C.Embedding(num_items, self.num_units, self.l2_reg,
                                         zero_pad=True, scale=True, scope="item_embs")

            self.pcoding_K = C.PositionCoding(self.timelen, self.num_units, self.l2_reg, scope="pcoding_K")
            self.pcoding_V = C.PositionCoding(self.timelen, self.num_units, self.l2_reg, scope="pcoding_V")
            self.tcoding_K = C.TimeIntervalCoding(self.timelen, self.num_units, self.l2_reg, scope="tcoding_K")
            self.tcoding_V = C.TimeIntervalCoding(self.timelen, self.num_units, self.l2_reg, scope="tcoding_V")

            self.output_bias = self.output_bias(inf_pad=True)

            self.list_attention = list()
            self.list_dense = list()
            for i in range(FLAGS.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    attention = T.TiMultiHeadAttention(self.num_units, self.num_heads,
                                                       self.attention_probs_dropout_rate, self.l2_reg,
                                                       self.pcoding_K, self.pcoding_V, self.tcoding_K, self.tcoding_V)
                    fforward = FeedForward([self.num_units, self.num_units], self.hidden_dropout_rate)
                    self.list_attention.append(attention)
                    self.list_dense.append(fforward)

    def __call__(self, features, is_training):
        seqs_id = features['seqs_i']
        seqs_ts = features['seqs_t'] / self.time_scale

        # Embedding and Transform
        seqs_units = self.item_embs(seqs_id)
        # time delta is different (but delivers better results)
        # from Line 29 - 39 (Relation function) of util.py
        # in the sourcecodes provided by the authors
        seqs_spans = tf.tile(
            tf.expand_dims(seqs_ts[:, 1:], 2), [1, 1, self.seqslen]) - tf.tile(
            tf.expand_dims(seqs_ts[:, :-1], 1), [1, self.seqslen, 1])
        seqs_spans = tf.to_int64(tf.clip_by_value(seqs_spans, clip_value_min=0, clip_value_max=self.timelen))

        # Dropout
        seqs_units = tf.layers.dropout(seqs_units, rate=self.hidden_dropout_rate,
                                       training=tf.convert_to_tensor(is_training))
        seqs_masks = tf.expand_dims(tf.to_float(tf.not_equal(seqs_id, 0)), -1)

        # multi-head attention
        seqs_outs = seqs_units * seqs_masks
        for i, (attention, dense) in enumerate(zip(self.list_attention, self.list_dense)):
            with tf.variable_scope("num_blocks_%d" % i):
                with tf.variable_scope("attention"):
                    seqs_outs = attention(layernorm(seqs_outs), seqs_outs,
                                          seqs_spans, is_training, causality=True)

                with tf.variable_scope("feedforward"):
                    seqs_outs = dense(layernorm(seqs_outs), is_training)
                    seqs_outs *= seqs_masks
        with tf.variable_scope("out_ln"):
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
