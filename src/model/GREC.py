"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import numpy as np
import tensorflow.compat.v1 as tf

import module.coding as C
import module.sequential as S
from model.Base import Sequential


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class GREC(Sequential):
    """ Implementation of the paper ---
        Yuan F, He X, Jiang H, Guo G, Xiong J, Xu Z, Xiong Y.
        Future data helps training: Modeling future contexts for session-based recommendation.
        WWW 2020.
    """

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)
        self.num_items += 2
        self.masklen = FLAGS.masklen
        self.filter_width = FLAGS.filter_width
        self.dilations = [int(dilation) for dilation in FLAGS.dilations.split(',')]

        with tf.variable_scope("GREC"):
            self.embs_enc = C.Embedding(self.num_items, self.num_units, self.l2_reg, scale=False, zero_pad=False,
                                        initializer=tf.truncated_normal_initializer(stddev=0.02), scope="enc_embs")
            self.embs_dec = C.Embedding(self.num_items, self.num_units, self.l2_reg, scale=False, zero_pad=False,
                                        initializer=tf.truncated_normal_initializer(stddev=0.02), scope="decc_embs")

            self.reguCNNs, self.maskCNNs = list(), list()
            for i, dilation in enumerate(self.dilations):
                out_channels = self.num_units
                self.reguCNNs.append(S.MaskedCNN(self.filter_width, out_channels, dilation,
                                                 causality=False, scope="regudCNN_%d" % i))
                self.maskCNNs.append(S.MaskedCNN(self.filter_width, out_channels, dilation,
                                                 causality=True, scope="maskedCNN_%d" % i))

    def __call__(self, features, is_training):
        # Perform embedding lookup on the word ids.
        if is_training:
            seqs_ids_enc, seqs_ids_dec = features['seqs_m'], features['seqs_i']
        else:
            seqs_ids_enc, seqs_ids_dec = features['seqs_i'], features['seqs_i']

        seqs_inputs_enc = self.embs_enc(seqs_ids_enc)
        seqs_inputs_dec = self.embs_dec(seqs_ids_dec)

        # Run the regular CNN layers
        seqs_outs = seqs_inputs_enc
        for i, reguCNN in enumerate(self.reguCNNs):
            with tf.variable_scope("block_%d" % i):
                seqs_outs = reguCNN(seqs_outs, is_training=is_training)

        # Run the projector
        with tf.variable_scope("projector"):
            layer_inps = tf.add(seqs_outs, seqs_inputs_dec)
            layer_outs = tf.layers.dense(layer_inps, 2 * self.num_units, activation=gelu)
            layer_outs = tf.layers.dense(layer_outs, self.num_units)
            layer_outs += layer_inps

        # Run the masked CNN layers
        seqs_outs = layer_outs
        for i, maskCNN in enumerate(self.maskCNNs):
            with tf.variable_scope("block_%d" % i):
                seqs_outs = maskCNN(seqs_outs, is_training=is_training)

        # Run batch gather
        if is_training:
            seqs_outs = tf.batch_gather(seqs_outs, features['masked_positions'])
            seqs_outs = tf.reshape(seqs_outs, [tf.shape(seqs_ids_enc)[0] * self.masklen, self.num_units])
        else:
            # only using the latest representations for making predictions
            seqs_outs = tf.reshape(seqs_outs[:, -1], [tf.shape(seqs_ids_enc)[0], self.num_units])

        # Compute logits
        with tf.variable_scope("cls/predictions"):
            with tf.variable_scope("transform"):
                seqs_outs = tf.nn.relu(seqs_outs)
                logits = tf.layers.dense(seqs_outs, self.num_items)

        if not is_training:
            masks = tf.ones_like(logits[:, :1]) * -1000.
            logits = tf.concat([masks, logits[:, 1:-1], masks], axis=-1)
        return logits

    def train(self, features, labels):
        logits = self.__call__(features, is_training=True)  # batch_size, mask_len
        labels = tf.reshape(labels, [-1])  # batch_size, mask_len
        # label_weights = tf.to_float(tf.not_equal(labels, 0))

        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # numerator = tf.reduce_sum(label_weights * per_example_loss)
        # denominator = tf.reduce_sum(label_weights) + 1e-5
        # loss = numerator / denominator
        loss = tf.reduce_mean(per_example_loss)
        regularization = 0.001 * tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss = loss + regularization

        train_op = self.trainOp(loss)
        with tf.variable_scope("Sequential/TRAIN"):
            _, loss_op = tf.metrics.mean(loss, name='loss')

        loss_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="main/Sequential/TRAIN")
        assert len(loss_vars) > 0, "(train)metric local variables should not be None."
        loss_init_op = tf.variables_initializer(loss_vars)

        return train_op, loss_op, loss_init_op

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
