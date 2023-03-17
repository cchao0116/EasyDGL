"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow.compat.v1 as tf

import module.coding as C
import module.temporal as T
from model.Base import Sequential


class TGREC(Sequential):
    """The implementation of ---
    Fan, Ziwei and Liu, Zhiwei and Zhang, Jiawei and Xiong, Yun and Zheng, Lei and Yu, Philip S.
    Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer.
    CIKM 2021.
    """

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)
        self.time_scale = FLAGS.time_scale

        with tf.variable_scope("TGREC"):
            self.item_embs = C.Embedding(num_items, self.num_units, self.l2_reg,
                                         zero_pad=True, scale=False, scope="item_embs")
            self.output_bias = self.output_bias(inf_pad=True)

            self.pcoding = C.PositionCoding(self.seqslen, self.num_units, self.l2_reg, scope="pcoding")
            self.tcoding = C.TimeFunctionCoding(self.num_units, scope="tcoding")

            self.list_attention = list()
            for i in range(FLAGS.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    attention = T.TgMultiHeadAttention(self.num_units, self.num_heads,
                                                       self.attention_probs_dropout_rate, self.l2_reg,
                                                       self.tcoding)
                    self.list_attention.append(attention)

    def __call__(self, features, is_training):
        seqs_id = features['seqs_i']
        seqs_ts = features['seqs_t'] / self.time_scale

        # Embedding and Transform
        seqs_units = self.item_embs(seqs_id)
        seqs_units += self.pcoding.code(seqs_units)
        seqs_spans = tf.tile(
            tf.expand_dims(seqs_ts, 2), [1, 1, self.seqslen]) - tf.tile(
            tf.expand_dims(seqs_ts, 1), [1, self.seqslen, 1])
        seqs_spans = tf.to_float(tf.maximum(seqs_spans, 0.))

        # Dropout
        seqs_units = tf.layers.dropout(seqs_units, rate=self.hidden_dropout_rate, training=is_training)
        seqs_masks = tf.expand_dims(tf.to_float(tf.not_equal(seqs_id, 0)), 1)
        seqs_masks = tf.tile(seqs_masks, [1, tf.shape(seqs_id)[1], 1])

        # Message-Passing
        seqs_outs = seqs_units
        for i, attention in enumerate(self.list_attention):
            with tf.variable_scope("num_blocks_%d" % i):
                # Dot-Product Multi-Head Attentions
                attention_outs = attention(seqs_outs, seqs_outs, seqs_masks, seqs_spans, is_training, causality=True)
                attention_outs = tf.layers.dropout(attention_outs, rate=self.hidden_dropout_rate, training=is_training)

                # Merge Layer
                intermediate_inputs = tf.concat([attention_outs, seqs_outs], axis=-1)
                intermediate_outs = tf.layers.dense(intermediate_inputs, self.num_units, activation=tf.nn.relu)
                seqs_outs = tf.layers.dense(intermediate_outs, self.num_units)

        if is_training:
            seqs_outs = tf.reshape(seqs_outs, [tf.shape(seqs_id)[0] * self.seqslen, self.num_units])
        else:
            # only using the latest representations for making predictions
            seqs_outs = tf.reshape(seqs_outs[:, -1], [tf.shape(seqs_id)[0], self.num_units])

        # compute logits
        logits = tf.matmul(seqs_outs, self.item_embs.lookup_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        return logits

    def trainOp(self, loss):
        global_step = tf.get_variable("global_step", shape=(), dtype=tf.int64, trainable=False)
        add_gstep = global_step.assign_add(1)
        # lrate = tf.train.exponential_decay(
        #     learning_rate=self.learning_rate, global_step=global_step,
        #     decay_steps=1000, decay_rate=0.9, staircase=True)
        # optimizer = tf.train.AdamOptimizer(learning_rate=lrate, beta2=0.98, epsilon=1e-9)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.)
        with tf.control_dependencies([add_gstep]):
            train_op = optimizer.minimize(loss)
        return train_op
