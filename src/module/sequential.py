"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow.compat.v1 as tf

from module.normalize import layernorm


def initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


class MultiHeadAttention(object):
    """ Implementation of the paper ---
    Vaswani, Ashish, et al.
    Attention is all you need.
    NeurIPS 2017.
    """

    def __init__(self, num_units, num_heads, dropout_rate, scope="multihead_attention"):
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.scope = scope

    def __call__(self, queries, keys, is_training, causality):
        num_units, num_heads, dropout_rate = self.num_units, self.num_heads, self.dropout_rate

        with tf.variable_scope(self.scope):
            # Set the fall back option for num_units
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries[:, :, :num_units]

            # Normalize
            # outputs = normalize(outputs) # (N, T_q, C)

            return outputs


class BERTAttention(object):
    """Implementation followed by
        https://github.com/google-research/bert/blob/master/modeling.py
    """

    def __init__(self, num_units, num_heads, dropout_rate,
                 initializer_range=0.02, scope="BERTAttention"):
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.scope = scope
        self.initializer_range = initializer_range

    def __call__(self, queries, keys, attention_masks, is_training):
        num_units, num_heads, dropout_rate = self.num_units, self.num_heads, self.dropout_rate

        with tf.variable_scope(self.scope):
            # Set the fall back option for num_units
            Q = tf.layers.dense(queries, num_units, activation=None, name="Q",
                                kernel_initializer=initializer(self.initializer_range))  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None, name="K",
                                kernel_initializer=initializer(self.initializer_range))  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None, name="V",
                                kernel_initializer=initializer(self.initializer_range))  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(attention_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
            return outputs


class Conv1D(object):
    """Implementation followed by
        https://github.com/fajieyuan/WWW2020-grec/blob/master/ops.py
    """

    def __init__(self, filter_width, out_channels, dilation, causality, scope):
        self.filter_width = filter_width
        self.out_channels = out_channels
        self.dilation = dilation
        self.causality = causality
        self.scope = scope

    def __call__(self, features):
        filter_width = self.filter_width
        in_channels = features.get_shape()[-1]
        out_channels = self.out_channels
        dilation = self.dilation

        with tf.variable_scope(self.scope):
            weight = tf.get_variable('weight', [1, filter_width, in_channels, out_channels],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
            bias = tf.get_variable('bias', [out_channels],
                                   initializer=tf.zeros_initializer())

            if self.causality:
                padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
                padded = tf.pad(features, padding)
                input_expanded = tf.expand_dims(padded, dim=1)  # [batch, in_height, in_width, in_channels]
                out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID')
                out = tf.nn.bias_add(out, bias)
            else:
                input_expanded = tf.expand_dims(features, dim=1)
                out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='SAME')
                out = tf.nn.bias_add(out, bias)
            return tf.squeeze(out, [1])


class MaskedCNN(object):
    """ Implementation of the paper ---
        Yuan F, He X, Jiang H, Guo G, Xiong J, Xu Z, Xiong Y.
        Future data helps training: Modeling future contexts for session-based recommendation.
        WWW 2020.
    """

    def __init__(self, filter_width, out_channels, dilation, causality, scope="maskedCNN"):
        self.dilation = dilation
        self.scope = scope

        with tf.variable_scope(scope):
            self.dconv0 = Conv1D(filter_width, out_channels, self.dilation, causality, scope="dconv0")
            self.dconv1 = Conv1D(filter_width, out_channels, 2 * self.dilation, causality, scope="dconv1")

    def __call__(self, features, is_training):
        with tf.variable_scope(self.scope):
            # Dilated Convolution
            outs = self.dconv0(features)
            outs = layernorm(outs, "ln0")
            outs = tf.nn.relu(outs)

            # Dilated Convolution
            outs = self.dconv1(outs)
            outs = layernorm(outs, "ln1")
            outs = tf.nn.relu(outs)

            # Residual Connection
            outs += features

            return outs


class MATEncoder(object):
    """The implementation of --- Multi-Aspect Time Encoder
    Cho J, Hyun D, Kang S, Yu H.
    Learning heterogeneous temporal patterns of user preference for timely recommendation.
    WWW 2021.
    """

    def __init__(self, scope="MATE"):
        self.scope = scope

    def __call__(self, queries, keys, users):
        num_units = queries.get_shape().as_list()[-1]

        with tf.variable_scope(self.scope):
            # N: batch_size, S: sequence_length, H: num_units, W: window_sizes
            users = tf.layers.dense(users, num_units, use_bias=False)  # N, S, H
            users = tf.expand_dims(users, axis=2)  # N, S, 1, H

            Q = queries * users  # N, S, 1, H
            K = keys * users  # N, S, W, H
            V = K  # N, S, W, H

            outputs = tf.matmul(Q, K, transpose_b=True)  # N, S, 1, W
            outputs = outputs / tf.sqrt(tf.to_float(num_units))
            outputs = tf.nn.softmax(outputs)  # N, S, 1, W

            outputs = tf.matmul(outputs, V)  # N, S, 1, H
            return tf.squeeze(outputs, axis=2)  # N, S, H


class TAHEncoder(object):
    """The implementation of --- Time-Aware History Encoder
    Cho J, Hyun D, Kang S, Yu H.
    Learning heterogeneous temporal patterns of user preference for timely recommendation.
    WWW 2021.
    """

    def __call__(self, queries, keys, histories):
        # in fact, queries and keys are identical
        queries = tf.nn.l2_normalize(queries, axis=-1)
        keys = tf.nn.l2_normalize(keys, axis=-1)

        # N: batch_size, Q: length of queries, K: length of keys
        # H: num_units
        outputs = tf.matmul(queries, keys, transpose_b=True)  # N, Q, K
        # [ 1. + cos(x, y)] / 2.
        outputs = (1. + outputs) / 2.

        # causality = Future blinding
        diag_vals = tf.ones_like(outputs[0, :, :])  # (Q, K)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (Q, K)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (N, Q, K)
        outputs *= masks

        outputs = tf.matmul(outputs, histories)  # N, Q, H
        return outputs
