"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import abc

import numpy as np
import tensorflow.compat.v1 as tf


def layernorm(inputs, center=True, scale=True, activation_fn=None,
              reuse=None, trainable=True, begin_norm_axis=1, begin_params_axis=-1, scope=None):
    """Run layer normalization on the last dimension of the tensor."""
    with tf.variable_scope(
            scope, 'LayerNorm', [inputs], reuse=reuse):
        inputs = tf.convert_to_tensor(inputs)
        inputs_shape = inputs.shape
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        if begin_norm_axis < 0:
            begin_norm_axis = inputs_rank + begin_norm_axis
        if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
            raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                             'must be < rank(inputs) (%d)' %
                             (begin_params_axis, begin_norm_axis, inputs_rank))
        params_shape = inputs_shape[begin_params_axis:]
        if not params_shape.is_fully_defined():
            raise ValueError(
                'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
                (inputs.name, begin_params_axis, inputs_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = tf.get_variable(
                'beta',
                shape=params_shape,
                dtype=dtype,
                initializer=tf.zeros_initializer(),
                trainable=trainable)
        if scale:
            gamma = tf.get_variable(
                'gamma',
                shape=params_shape,
                dtype=dtype,
                initializer=tf.ones_initializer(),
                trainable=trainable)
        # By default, compute the moments across all the dimensions except the one with index 0.
        norm_axes = list(range(begin_norm_axis, inputs_rank))
        mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        # Note that epsilon must be increased for float16 due to the limited
        # representable range.
        variance_epsilon = 1e-12 if dtype != tf.float16 else 1e-3
        outputs = tf.nn.batch_normalization(
            inputs,
            mean,
            variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=variance_epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


class FeedForward(object):
    def __init__(self, num_units=(2048, 512), dropout_rate=0.2, scope="dense"):
        with tf.variable_scope(scope):
            self.conv1d0 = tf.layers.Conv1D(num_units[0], 1, activation=tf.nn.relu, name="Inner")
            self.conv1d1 = tf.layers.Conv1D(num_units[1], 1, activation=None, name="Readout")
            self.dropout_rate = dropout_rate

    def __call__(self, inputs, is_training):
        # Inner layer
        outputs = self.conv1d0(inputs)
        outputs = tf.layers.dropout(outputs, rate=self.dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        outputs = self.conv1d1(outputs)
        outputs = tf.layers.dropout(outputs, rate=self.dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs
        return outputs


class Sequential(object):

    def __init__(self, num_items, FLAGS):
        self.num_items = num_items
        self.num_units = FLAGS.num_units

        self.num_heads = FLAGS.num_heads
        self.hidden_dropout_rate = FLAGS.hidden_dropout_rate
        self.attention_probs_dropout_rate = FLAGS.attention_probs_dropout_rate
        self.seqslen = FLAGS.seqslen

        self.learning_rate = FLAGS.learning_rate
        self.l2_reg = FLAGS.l2_reg
        self.num_train_steps = FLAGS.num_train_steps
        self.num_warmup_steps = FLAGS.num_warmup_steps

    def output_bias(self, inf_pad=True):
        if inf_pad:
            output_bias = tf.get_variable(
                "output_bias", shape=[self.num_items - 1], initializer=tf.zeros_initializer())
            return tf.concat([[-1000.], output_bias], -1)
        else:
            return tf.get_variable(
                "output_bias", shape=[self.num_items], initializer=tf.zeros_initializer())

    @abc.abstractmethod
    def __call__(self, features, is_training):
        raise NotImplementedError("the model is not implemented")

    def train(self, features, labels):
        logits = self.__call__(features, is_training=True)
        log_probs = tf.log(tf.nn.softmax(logits, -1) + 1e-5)  # (bs*seqsLen, num_items)

        labels = tf.reshape(labels, [-1])
        label_ids = tf.one_hot(labels, depth=self.num_items, dtype=tf.float32)
        label_weights = tf.to_float(tf.not_equal(labels, 0))

        per_example_loss = -tf.reduce_sum(log_probs * label_ids, axis=[-1])
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

    def trainOp(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(loss)

        # from optimization import create_optimizer
        # return create_optimizer(loss, self.learning_rate, self.num_train_steps,
        #                         self.num_warmup_steps, False)

    def eval(self, features, labels, mask_seen=True):
        batch_size = tf.shape(labels)[0]
        seqslen = features['seqs_i'].get_shape().as_list()[1]
        logits = self.__call__(features, is_training=False)
        tf.add_to_collection("ANALYTICS", logits)

        if mask_seen:
            # mask the training observations when making predictions
            x_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, seqslen]), [-1])
            y_indices = tf.cast(tf.reshape(features['seqs_i'], [-1]), tf.int32)
            sp_indices = tf.cast(tf.stack([x_indices, y_indices], axis=1), tf.int64)
            sp_values = tf.ones_like(y_indices, dtype=tf.float32) * -np.inf
            logits_masks = tf.sparse.SparseTensor(sp_indices, sp_values, [batch_size, self.num_items])
            logits += tf.sparse.to_dense(logits_masks, validate_indices=False)
        probs = tf.nn.softmax(logits, -1)  # (bs*seqsLen, num_items)
        probs = tf.reshape(probs, [batch_size, self.num_items])

        # next-item predictions
        pred = probs  # batch_size, num_items
        real = labels[:, -1:]  # batch_size, 1

        with tf.variable_scope("Sequential/EVAL"):
            metrics = dict()

            x_indices = tf.cast(tf.range(batch_size), tf.int32)
            y_indices = tf.cast(tf.reshape(real, [-1]), tf.int32)
            sp_indices = tf.cast(tf.stack([x_indices, y_indices], axis=1), tf.int64)
            sp_values = tf.ones_like(x_indices, dtype=tf.float32)
            real = tf.sparse.SparseTensor(sp_indices, sp_values, [batch_size, self.num_items])
            real = tf.sparse.to_dense(real)

            _, tp_k_idx = tf.nn.top_k(pred, k=100)
            true_positive_100 = tf.batch_gather(real, tf.cast(tp_k_idx, dtype=tf.int64))  # [batch size, topN]
            true_positive_50 = tf.batch_gather(real, tf.cast(tp_k_idx[:, :50], dtype=tf.int64))  # [batch size, topN]
            true_positive_10 = tf.batch_gather(real, tf.cast(tp_k_idx[:, :10], dtype=tf.int64))  # [batch size, topN]

            # implementations of HitRate@k
            HR100 = tf.sign(tf.reduce_sum(true_positive_100, axis=-1))
            HR50 = tf.sign(tf.reduce_sum(true_positive_50, axis=-1))
            HR10 = tf.sign(tf.reduce_sum(true_positive_10, axis=-1))
            metrics['H100'] = tf.metrics.mean(HR100, name='H100')
            metrics['H50'] = tf.metrics.mean(HR50, name='H50')
            metrics['H10'] = tf.metrics.mean(HR10, name='H10')

            # implementations of NDCG@k
            gain = 1. / np.log2(np.arange(2, 100 + 2))
            NDCG100 = tf.reduce_sum(true_positive_100 * gain, axis=-1)
            NDCG50 = tf.reduce_sum(true_positive_50 * gain[:50], axis=-1)
            NDCG10 = tf.reduce_sum(true_positive_10 * gain[:10], axis=-1)
            metrics['N100'] = tf.metrics.mean(NDCG100, name='N100')
            metrics['N50'] = tf.metrics.mean(NDCG50, name='N50')
            metrics['N10'] = tf.metrics.mean(NDCG10, name='N10')
            metrics_op = {key: val[1] for key, val in metrics.items()}

        metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="main/Sequential/EVAL")
        assert len(metric_vars) > 0, "(eval)metric local variables should not be None."
        metric_init_op = tf.variables_initializer(metric_vars)
        return metrics_op, metric_init_op
