"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
from tensorflow.python.platform import tf_logging as logging

import numbers
import numpy as np
import tensorflow.compat.v1 as tf


def l2_regularizer(scale, scope=None):
    """Returns a function that can be used to apply L2 regularization to weights.
  Small values of L2 can help prevent overfitting the training data.
  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.
  Returns:
    A function with signature `l2(weights)` that applies L2 regularization.
  Raises:
    ValueError: If scale is negative or if scale is not a float.
    """
    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % (scale,))
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                             scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _: None

    def l2(weights):
        """Applies l2 regularization to weights."""
        with tf.name_scope(scope, 'l2_regularizer', [weights]) as name:
            my_scale = tf.convert_to_tensor(scale,
                                             dtype=weights.dtype.base_dtype,
                                             name='scale')
            return tf.multiply(my_scale, tf.nn.l2_loss(weights), name=name)

    return l2


class Embedding(object):
    def __init__(self, vocab_size, num_units, l2_reg=0., zero_pad=True,
                 scale=True, initializer=None, scope="embedding"):
        regularizer = None if l2_reg == 0. else l2_regularizer(l2_reg)
        self._num_units = num_units
        self._scale = scale

        with tf.variable_scope(scope):
            lookup_table = tf.get_variable(
                'lookup_table', dtype=tf.float32, initializer=initializer,
                shape=[vocab_size, num_units], regularizer=regularizer)
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        self.lookup_table = lookup_table

    def __call__(self, inputs):
        outputs = tf.nn.embedding_lookup(self.lookup_table, inputs)
        if self._scale:
            outputs = outputs * (self._num_units ** 0.5)
        return outputs


class PositionCoding(object):
    def __init__(self, vocab_size, num_units, l2_reg=0., initializer=None, scope="coding/pos"):
        with tf.variable_scope(scope):
            self.pembs = Embedding(vocab_size, num_units, l2_reg, zero_pad=False, initializer=initializer, scale=False)

    def __call__(self, inputs, **kwargs):
        pcoding = self.code(inputs)
        return tf.concat([inputs, pcoding], axis=-1)

    def code(self, inputs):
        batch_size, seqs_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        pcoding = self.pembs(tf.tile(tf.expand_dims(tf.range(seqs_len), 0), [batch_size, 1]))
        return pcoding


class TimeIntervalCoding(object):
    """ Implementation of the paper ---
            Li J, Wang Y, McAuley J.
            Time interval aware self-attention for sequential recommendation.
            WSDM 2020.
    """

    def __init__(self, vocab_size, num_units, l2_reg=0., scope="coding/tim"):
        with tf.variable_scope(scope):
            self.pembs = Embedding(vocab_size, num_units, l2_reg, zero_pad=False, scale=False)

    def code(self, inputs):
        return self.pembs(inputs)


class TimeFunctionCoding(object):
    """Implementation of the paper ---
        Xu D, Ruan C, Korpeoglu E, Kumar S, Achan K.
        Inductive representation learning on temporal graphs.
        ICLR 2020.
    """

    def __init__(self, num_units, scope="coding/tif"):
        self._num_units = num_units
        with tf.variable_scope(scope):
            self.basis_freq = tf.get_variable(
                "basis_freq", shape=[num_units], dtype=tf.float32,
                initializer=tf.constant_initializer(np.linspace(0, 9, num_units).astype(np.float32)))
            self.phase = tf.get_variable("phase", shape=[num_units],
                                         dtype=tf.float32, initializer=tf.zeros_initializer())

    def code(self, inputs):
        batch_size, seqslen = tf.shape(inputs)[0], tf.shape(inputs)[1]
        inputs = tf.to_float(tf.reshape(inputs, [batch_size, seqslen, -1]))
        inputs = tf.tile(tf.expand_dims(inputs, -1), [1, 1, 1, self._num_units])

        # Harmonic
        outs = inputs * self.basis_freq
        outs = tf.nn.bias_add(outs, self.phase)
        outs = tf.cos(outs)
        return outs


class TimeSinusoidCoding(object):
    """The implementation of ---
    Cho J, Hyun D, Kang S, Yu H.
    Learning heterogeneous temporal patterns of user preference for timely recommendation.
    WWW 2021.
    """

    def __init__(self, num_units):
        self.num_units = num_units
        scale = np.power(10000, np.arange(0, num_units, 2) * 1. / num_units)
        self.scale = tf.reshape(tf.convert_to_tensor(scale, dtype=tf.float32), [1, 1, num_units // 2])

    def code(self, inputs):
        shape_list = inputs.shape.as_list()
        assert len(shape_list) == 2, "the tensor rank should be 2."

        x = tf.expand_dims(tf.to_float(inputs), axis=-1)
        x = tf.tile(x, [1, 1, self.num_units // 2]) / self.scale

        code_even = tf.sin(x)
        code_odd = tf.cos(x)

        code = tf.stack([code_even, code_odd], axis=-1)
        code = tf.reshape(code, [-1, shape_list[1], self.num_units])
        return code
