"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import pickle

import tensorflow.compat.v1 as tf

import module.coding as C
import module.temporal as T
from model.Base import Sequential, FeedForward, layernorm


class CTSMA(Sequential):
    """ Implementation of the paper ---
    C Chen, H Geng, N Yang, J Yan, D Xue, J Yu, X Yang.
    Learning Self-Modulating Attention in Continuous Time Space with Applications to Sequential Recommendation.
    ICML 2021.
    """

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)
        self.mark_lookup_table = pickle.load(open(FLAGS.mark, 'rb')).toarray()
        self.num_events = self.mark_lookup_table.shape[-1]
        self.ct_reg = FLAGS.ct_reg
        self.time_scale = FLAGS.time_scale

        with tf.variable_scope("CSTMA"):
            self.item_embs = C.Embedding(num_items, self.num_units, self.l2_reg,
                                         zero_pad=True, scale=True, scope="item_embs")
            self.pcoding = C.PositionCoding(self.seqslen, self.num_units, self.l2_reg, scope="spatial_embs")

            self.output_bias = self.output_bias(inf_pad=True)

            self.list_attention = list()
            self.list_dense = list()
            for i in range(FLAGS.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    attention = T.MAU(self.num_units, self.num_heads, self.num_events,
                                      self.attention_probs_dropout_rate)
                    fforward = FeedForward([self.num_units, self.num_units], self.hidden_dropout_rate)
                    self.list_attention.append(attention)
                    self.list_dense.append(fforward)

    def __call__(self, features, is_training):
        seqs_id = features['seqs_i']
        seqs_ts = features['seqs_t'] / self.time_scale
        seqs_spans = seqs_ts[:, 1:] - seqs_ts[:, :-1]

        # positional encoding
        seqs_marks = tf.nn.embedding_lookup(self.mark_lookup_table, seqs_id)
        seqs_units = self.item_embs(seqs_id)
        seqs_units = self.pcoding(seqs_units)

        # Dropout
        seqs_units = tf.layers.dropout(seqs_units, rate=self.hidden_dropout_rate,
                                       training=tf.convert_to_tensor(is_training))
        seqs_masks = tf.expand_dims(tf.to_float(tf.not_equal(seqs_id, 0)), 1)
        seqs_masks = tf.tile(seqs_masks, [self.num_heads, tf.shape(seqs_id)[1], 1])

        # multi-head self-modulating attention
        seqs_outs = seqs_units
        for i, (attention, dense) in enumerate(zip(self.list_attention, self.list_dense)):
            with tf.variable_scope("num_blocks_%d" % i):
                # sequential-temporal representations
                with tf.variable_scope("attention"):
                    seqs_outs, seqs_intny = attention(layernorm(seqs_outs), seqs_outs, seqs_masks,
                                                      seqs_spans, seqs_marks, is_training, causality=True)

                # feed-forward
                with tf.variable_scope("feed-forward"):
                    seqs_outs = dense(layernorm(seqs_outs), is_training)

                # likelihood for point process
                if is_training:
                    tf.add_to_collection("LLE_PP", seqs_intny)

        with tf.variable_scope("outln"):
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

    def train(self, features, labels):
        logits = self.__call__(features, is_training=True)
        log_probs = tf.log(tf.nn.softmax(logits, -1) + 1e-5)  # (bs*seqsLen, num_items)

        # continuous-time regularization
        regularizer = tf.losses.get_regularization_loss()
        if self.ct_reg != 0.:
            seqs_spans = features['seqs_t'][:, 1:] - features['seqs_t'][:, :-1]
            next_mark_onehot = tf.to_float(tf.nn.embedding_lookup(self.mark_lookup_table, labels))

            if self.num_heads != 1:
                seqs_spans = tf.tile(seqs_spans, [self.num_heads, 1])
                next_mark_onehot = tf.tile(next_mark_onehot, [self.num_heads, 1, 1])

            for seqs_intny in tf.get_collection("LLE_PP"):
                ct_regularizer = T.MAU.biased_likelihood(
                    seqs_intny, next_mark_onehot, seqs_spans)
                regularizer += self.ct_reg * ct_regularizer

        # softmax based loss without negative sampling
        labels = tf.reshape(labels, [-1])
        label_ids = tf.one_hot(labels, depth=self.num_items, dtype=tf.float32)
        label_weights = tf.to_float(tf.not_equal(labels, 0))

        per_example_loss = -tf.reduce_sum(log_probs * label_ids, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

        # perform gradient-based optimization
        loss = loss + regularizer
        train_op = self.trainOp(loss)

        with tf.variable_scope("Sequential/TRAIN"):
            _, loss_op = tf.metrics.mean(loss, name='loss')

        loss_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="main/Sequential/TRAIN")
        assert len(loss_vars) > 0, "(train)metric local variables should not be None."
        loss_init_op = tf.variables_initializer(loss_vars)

        return train_op, loss_op, loss_init_op
