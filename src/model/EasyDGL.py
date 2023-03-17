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


def clip_by_value(input_tensor):
    return tf.clip_by_value(input_tensor, clip_value_min=0., clip_value_max=100.)


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


class EasyDGL(Sequential):

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)
        self.mask = num_items
        self.seqslen += 1
        self.num_items += 1
        self.masklen = FLAGS.masklen
        self.time_scale = FLAGS.time_scale

        self.mark_lookup_table = pickle.load(open(FLAGS.mark, 'rb')).toarray()
        self.num_events = self.mark_lookup_table.shape[-1]
        self.ct_reg = FLAGS.ct_reg

        with tf.variable_scope("CSTMA"):
            self.item_embs = C.Embedding(self.num_items, self.num_units, self.l2_reg,
                                         zero_pad=True, scale=True, scope="item_embs")
            self.mark_embs = C.Embedding(self.num_events, self.num_units, self.l2_reg,
                                         zero_pad=True, scale=False, scope="mark_embs")
            self.pcoding = C.PositionCoding(self.seqslen, self.num_units, self.l2_reg, scope="spatial_embs")
            self.tcoding = C.TimeSinusoidCoding(self.num_units)

            self.output_bias = self.output_bias(inf_pad=True)

            self.list_attention = list()
            self.list_dense = list()
            for i in range(FLAGS.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    attention = T.BiMAU(self.num_units, self.num_heads,
                                        self.num_events, self.attention_probs_dropout_rate)
                    fforward = FeedForward([self.num_units, self.num_units], self.hidden_dropout_rate)
                    self.list_attention.append(attention)
                    self.list_dense.append(fforward)

    def __call__(self, features, is_training):
        seqs_ids = features['seqs_i']
        seqs_ts = features['seqs_t'] / self.time_scale

        seqs_spans = clip_by_value(seqs_ts[:, 1:] - seqs_ts[:, :-1])
        seqs_spans = tf.concat([seqs_spans[:, :1], seqs_spans], axis=-1)

        seqs_marks = tf.where(tf.equal(seqs_ids, self.mask), tf.zeros_like(seqs_ids), seqs_ids)
        seqs_marks = tf.nn.embedding_lookup(self.mark_lookup_table, seqs_marks)

        # temporal encoding
        seqs_tcodes = self.tcoding.code(seqs_ts)

        # positional encoding
        seqs_units = self.item_embs(seqs_ids) + seqs_tcodes
        posn_codes = self.pcoding.code(seqs_units)

        # event mark encoding
        marks_codes = tf.nn.embedding_lookup(self.mark_embs.lookup_table, seqs_marks)
        marks_codes = tf.reduce_sum(marks_codes, axis=2)
        seqs_units = tf.concat([seqs_units, posn_codes, marks_codes], axis=-1)

        # Dropout
        seqs_units = tf.layers.dropout(seqs_units, rate=self.hidden_dropout_rate,
                                       training=tf.convert_to_tensor(is_training))
        seqs_masks = tf.expand_dims(tf.to_float(tf.not_equal(seqs_ids, 0)), 1)
        seqs_masks = tf.tile(seqs_masks, [self.num_heads, tf.shape(seqs_ids)[1], 1])

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        prev_outputs = seqs_units
        for i, attention in enumerate(self.list_attention):
            with tf.variable_scope("layer_%d" % i):
                layer_inputs = prev_outputs

                with tf.variable_scope("attention"):
                    with tf.variable_scope("self"):
                        # sequential-temporal representations
                        attention_outs, seqs_intny = attention(layer_inputs, layer_inputs, seqs_masks,
                                                               seqs_spans, seqs_marks, is_training)

                    # Run a linear projection of `hidden_size` then add a residual
                    # with `layer_input`.
                    with tf.variable_scope("output"):
                        attention_outs = tf.layers.dense(attention_outs, self.num_units)
                        attention_outs = tf.layers.dropout(attention_outs, rate=self.hidden_dropout_rate,
                                                           training=is_training)
                        attention_outs = layernorm(attention_outs + layer_inputs[:, :, :self.num_units])

                # The activation is only applied to the "intermediate" hidden layer.
                with tf.variable_scope("intermediate"):
                    intermediate_outputs = tf.layers.dense(attention_outs, 2 * self.num_units,
                                                           activation=gelu)

                # Down-project back to `hidden_size` then add the residual.
                with tf.variable_scope("output"):
                    layer_outputs = tf.layers.dense(intermediate_outputs, self.num_units)
                    layer_outputs = tf.layers.dropout(layer_outputs, rate=self.hidden_dropout_rate,
                                                      training=is_training)
                    layer_outputs = layernorm(layer_outputs + attention_outs)
                    prev_outputs = layer_outputs

                # likelihood for point process
                if is_training:
                    tf.add_to_collection("LLE_PP", seqs_intny)

        seqs_outs = prev_outputs
        with tf.variable_scope("cls/predictions"):
            with tf.variable_scope("transform"):
                seqs_outs = tf.layers.dense(seqs_outs, self.num_units, activation=gelu)
                seqs_outs = layernorm(seqs_outs)

        if is_training:
            seqs_outs = tf.batch_gather(seqs_outs, features['masked_positions'])
            seqs_outs = tf.reshape(seqs_outs, [tf.shape(seqs_ids)[0] * self.masklen, self.num_units])
        else:
            # only using the latest representations for making predictions
            seqs_outs = tf.reshape(seqs_outs[:, -1], [tf.shape(seqs_ids)[0], self.num_units])

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
            masked_positions = features['masked_positions']
            seqs_spans = clip_by_value(features['seqs_t'][:, 1:] - features['seqs_t'][:, :-1])
            seqs_spans = tf.concat([seqs_spans[:, :1], seqs_spans], axis=-1)
            seqs_spans = tf.batch_gather(seqs_spans, masked_positions)
            next_mark_onehot = tf.to_float(tf.nn.embedding_lookup(self.mark_lookup_table, labels))

            if self.num_heads != 1:
                seqs_spans = tf.tile(seqs_spans, [self.num_heads, 1])
                next_mark_onehot = tf.tile(next_mark_onehot, [self.num_heads, 1, 1])
                masked_positions = tf.tile(masked_positions, [self.num_heads, 1])

            for seqs_intny in tf.get_collection("LLE_PP"):
                seqs_intny = tf.batch_gather(seqs_intny, masked_positions)
                ct_regularizer = T.MAU.biased_likelihood(
                    seqs_intny, next_mark_onehot, seqs_spans)
                regularizer += self.ct_reg * ct_regularizer / self.num_heads

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
