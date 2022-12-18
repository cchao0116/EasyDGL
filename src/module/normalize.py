"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow.compat.v1 as tf


def layernorm(x, scope, epsilon=1e-8):
    with tf.variable_scope(scope):
        shape = x.get_shape()
        beta = tf.get_variable('beta', [int(shape[-1])], initializer=tf.constant_initializer(0))
        gamma = tf.get_variable('gamma', [int(shape[-1])], initializer=tf.constant_initializer(1))

        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)

        x = (x - mean) / tf.sqrt(variance + epsilon)

        return gamma * x + beta
