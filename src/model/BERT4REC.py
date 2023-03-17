"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow.compat.v1 as tf

import module.coding as C
import module.sequential as S
from model.Base import Sequential, layernorm


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


def initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


class BERT4REC(Sequential):
    """ Implementation of the paper ---
        Sun F, Liu J, Wu J, Pei C, Lin X, Ou W, Jiang P.
        BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer.
        CIKM 2019.
    """

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)
        self.seqslen += 1
        self.num_items += 2
        self.masklen = FLAGS.masklen

        with tf.variable_scope("BERT4REC"):
            self.item_embs = C.Embedding(self.num_items, self.num_units, self.l2_reg,
                                         zero_pad=True, scope="item_embs")
            self.pcoding = C.PositionCoding(self.seqslen, self.num_units, self.l2_reg, scope="spatial_embs")
            self.output_bias = self.output_bias(inf_pad=True)

            self.list_attention = list()
            for i in range(FLAGS.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    attention = S.BERTAttention(self.num_units, self.num_heads, self.attention_probs_dropout_rate)
                    self.list_attention.append(attention)

    def __call__(self, features, is_training):
        seqs_ids = features['seqs_i']  #

        # Perform embedding lookup on the word ids.
        seqs_units = self.item_embs(seqs_ids)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        with tf.variable_scope("embeddings"):
            seqs_units += self.pcoding.code(seqs_units)
            seqs_units = layernorm(seqs_units)
            seqs_units = tf.layers.dropout(seqs_units, rate=self.hidden_dropout_rate, training=is_training)

        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        broadcast_ones = tf.expand_dims(tf.ones_like(seqs_ids, dtype=tf.float32), 2)
        key_masks = tf.expand_dims(tf.to_float(tf.not_equal(seqs_ids, 0)), 1)
        attention_masks = broadcast_ones * key_masks
        if self.num_heads != 1:
            attention_masks = tf.tile(attention_masks, [self.num_heads, 1, 1])

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        prev_outputs = seqs_units
        for i, attention in enumerate(self.list_attention):
            with tf.variable_scope("layer_%d" % i):
                layer_inputs = prev_outputs

                with tf.variable_scope("attention"):
                    with tf.variable_scope("self"):
                        attention_outs = attention(layer_inputs, layer_inputs, attention_masks, is_training)

                    # Run a linear projection of `hidden_size` then add a residual
                    # with `layer_input`.
                    with tf.variable_scope("output"):
                        attention_outs = tf.layers.dense(attention_outs, self.num_units)
                        attention_outs = tf.layers.dropout(attention_outs, rate=self.hidden_dropout_rate,
                                                           training=is_training)
                        attention_outs = layernorm(attention_outs + layer_inputs)

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
        logits = self.__call__(features, is_training=True)  # batch_size, mask_len
        labels = tf.reshape(labels, [-1])  # batch_size, mask_len
        label_weights = tf.to_float(tf.not_equal(labels, 0))

        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator + tf.losses.get_regularization_loss()
        train_op = self.trainOp(loss)

        with tf.variable_scope("Sequential/TRAIN"):
            _, loss_op = tf.metrics.mean(loss, name='loss')

        loss_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="main/Sequential/TRAIN")
        assert len(loss_vars) > 0, "(train)metric local variables should not be None."
        loss_init_op = tf.variables_initializer(loss_vars)

        return train_op, loss_op, loss_init_op
