"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow.compat.v1 as tf

import module.coding as C
import module.sequential as S
from model.Base import Sequential, FeedForward, layernorm


class TimelyREC(Sequential):
    """The implementation of ---
    Cho J, Hyun D, Kang S, Yu H.
    Learning heterogeneous temporal patterns of user preference for timely recommendation.
    WWW 2021.
    """

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)
        self.window_ratio = FLAGS.window_ratio
        self.time_scale = FLAGS.time_scale

        with tf.variable_scope("TimelyREC"):
            self.item_embs = C.Embedding(num_items, self.num_units, self.l2_reg,
                                         zero_pad=True, scale=False, scope="item_embs")
            self.output_bias = self.output_bias(inf_pad=True)

            # inductive user embeddings
            self.pcoding = C.PositionCoding(self.seqslen, self.num_units, self.l2_reg, scope="spatial_embs")
            self.tcoding = C.TimeSinusoidCoding(self.num_units)
            self.te_weight = tf.get_variable("te_weight", shape=(), initializer=tf.ones_initializer())

            self.attention = S.MultiHeadAttention(self.num_units, self.num_heads, self.attention_probs_dropout_rate)
            self.fforward = FeedForward([self.num_units, self.num_units], self.hidden_dropout_rate)

            # periodic patterns
            self.month_embs = C.Embedding(12, self.num_units, self.l2_reg,
                                          zero_pad=False, scale=False, scope="month_embs")
            self.day_embs = C.Embedding(31, self.num_units, self.l2_reg,
                                        zero_pad=False, scale=False, scope="day_embs")
            self.weekday_embs = C.Embedding(7, self.num_units, self.l2_reg,
                                            zero_pad=False, scale=False, scope="weekday_embs")
            self.hour_embs = C.Embedding(24, self.num_units, self.l2_reg,
                                         zero_pad=False, scale=False, scope="hour_embs")

            # Multi-Aspect Time Encoders
            self.month_mate = S.MATEncoder("month_mate")
            self.day_mate = S.MATEncoder("daymate")
            self.weekday_mate = S.MATEncoder("weekday_mate")
            self.hour_mate = S.MATEncoder("hour_mate")

            # Time-Aware History Encoders
            self.tahe = S.TAHEncoder()

    def timeslot(self, feature, maxrange, embedding: C.Embedding):
        # the irregularity of periodic patterns
        window_size = max(int(maxrange * self.window_ratio + 0.5), 1) + 1
        delta_3d = tf.reshape(tf.range(1, window_size + 1, dtype=tf.int64), [1, 1, window_size])
        delta_3d = tf.concat([delta_3d, -delta_3d], axis=2)  # 1, 1, window_size * 2

        # period information of different granulaity
        feature_3d = tf.expand_dims(embedding(feature), axis=2)  # batch_size, seqslen, 1, num_units

        timeslot_3d = tf.expand_dims(feature, axis=-1)  # batch_size, seqslen, 1
        timeslot_3d = tf.mod(timeslot_3d + delta_3d, maxrange)  # batch_size, seqslen, window_size * 2
        timeslot_3d = embedding(tf.reshape(timeslot_3d, [-1, self.seqslen, 2, window_size]))
        timeslot_3d = tf.reduce_sum(timeslot_3d, axis=2)  # batch_size, seqslen, window_size, num_units
        timeslot_3d = tf.math.cumsum(timeslot_3d, axis=1)  # batch_size, seqslen, window_size, num_units

        numerator = feature_3d + timeslot_3d  # batch_size, seqslen, window_size, num_units
        denominator = tf.range(1, window_size + 1, dtype=tf.float32) * 2. + 1.

        outs = numerator / tf.reshape(denominator, [1, window_size, 1])  # batch_size, seqslen, window_size, num_units
        outs = tf.concat([feature_3d, outs], axis=2)  # batch_size, seqslen, window_size + 1, num_units
        return feature_3d, outs

    def inductive_user_embs(self, features, is_training):
        seqs_id = features['seqs_i']

        # positional encoding
        seqs_units = self.item_embs(seqs_id)
        seqs_units = self.pcoding(seqs_units)

        # Dropout
        seqs_units = tf.layers.dropout(seqs_units, rate=self.hidden_dropout_rate,
                                       training=tf.convert_to_tensor(is_training))
        seqs_masks = tf.expand_dims(tf.to_float(tf.not_equal(seqs_id, 0)), -1)

        # Attention
        seqs_outs = seqs_units * seqs_masks
        with tf.variable_scope("TimelyREC/Atttention"):
            seqs_outs = self.attention(layernorm(seqs_outs), seqs_outs, is_training, causality=True)
        with tf.variable_scope("TimelyREC/fforward"):
            seqs_outs = self.fforward(layernorm(seqs_outs), is_training)
        seqs_outs *= seqs_masks

        with tf.variable_scope("TimelyREC/out"):
            seqs_outs = layernorm(seqs_outs)
        return seqs_outs

    def __call__(self, features, is_training):
        # N: batch_size, S: sequence_length, H: num_units, W: window_sizes
        with tf.variable_scope("TimelyREC/attention"):
            user_outs = self.inductive_user_embs(features, is_training)  # N, S, H

        with tf.variable_scope("TimelyREC/mate"):
            # Run MATE for different granularities
            # pd.datetime, month ranging from 1 to 12
            queries_month, keys_month = self.timeslot(features['seqs_month'] - 1, 12, self.month_embs)
            period_month = self.month_mate(queries_month, keys_month, user_outs)

            # pd.datetime, day ranging from 1 to 31
            queries_day, keys_day = self.timeslot(features['seqs_day'] - 1, 31, self.day_embs)
            period_day = self.day_mate(queries_day, keys_day, user_outs)

            # pd.datetime, weekday ranging from 0 to 6
            queries_weekday, keys_weekday = self.timeslot(features['seqs_weekday'], 7, self.weekday_embs)
            period_weekday = self.weekday_mate(queries_weekday, keys_weekday, user_outs)

            # pd.datetime, hour ranging from 0 to 23
            queries_hour, keys_hour = self.timeslot(features['seqs_hour'], 24, self.hour_embs)
            period_hour = self.hour_mate(queries_hour, keys_hour, user_outs)

            # Combine period information, Eq. (3)
            period_queries = tf.layers.dense(user_outs, self.num_units, use_bias=False)
            period_queries = tf.reshape(period_queries, [-1, self.seqslen, 1, self.num_units])

            period_keys = tf.concat([period_month, period_day, period_weekday, period_hour], axis=-1)
            period_keys = tf.reshape(period_keys, [-1, self.seqslen, 4, self.num_units])

            period_outs = tf.matmul(period_queries, period_keys, transpose_b=True)
            period_outs = tf.nn.sigmoid(period_outs)
            period_outs = tf.squeeze(tf.matmul(period_outs, period_keys), axis=2)  # N, S, H

        with tf.variable_scope("TimelyREC/tahe"):
            # Run TAHE
            seqs_inputs = self.item_embs(features['seqs_i'])
            # time encoding, linearly combined with historical item embeddings
            seqs_tcodes = self.tcoding.code(features['seqs_t'][:, :-1] / self.time_scale )
            seqs_inputs = seqs_inputs + self.te_weight * seqs_tcodes

            # apply key masks to sequence embeddings, so that key masking is omit in TAHE
            seqs_mask = tf.expand_dims(tf.sign(features['seqs_i']), axis=2)
            seqs_inputs *= tf.to_float(seqs_mask)
            history_outs = self.tahe(period_outs, period_outs, seqs_inputs)  # N, S, H

        with tf.variable_scope("TimelyREC/prediction"):
            seqs_outs = tf.concat([user_outs, history_outs, period_outs], axis=-1)  # N, S, 4H

            # using dictionary learning, in S2PNM
            # Here, the dicitonary size is quite small, larger size offers better result
            seqs_outs = tf.layers.dense(seqs_outs, 2 * self.num_units, activation=tf.nn.sigmoid)  # N, S, 2H
            seqs_outs = tf.layers.dense(seqs_outs, self.num_units)  # N, S, H

        batch_size = tf.shape(features['seqs_i'])[0]
        if is_training:
            # distributive laws:
            # logits = user_embedding * (item_embedding + te_weight * tcodes)
            #        = user_embedding * item_embedding + te_weight * user_embedding * tcodes
            bias_tcodes = tf.reduce_sum(seqs_outs * seqs_tcodes, axis=-1, keepdims=True)
            bias_tcodes = tf.reshape(bias_tcodes, [-1, 1])
            seqs_outs = tf.reshape(seqs_outs, [batch_size * self.seqslen, self.num_units])
        else:
            # only using the latest representations for making predictions
            bias_tcodes = 0.
            seqs_outs = tf.reshape(seqs_outs[:, -1], [batch_size, self.num_units])

        # compute logits
        logits = tf.matmul(seqs_outs, self.item_embs.lookup_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias) + bias_tcodes * self.te_weight
        return logits
