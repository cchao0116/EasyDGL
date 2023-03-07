"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow.compat.v1 as tf

import module.coding as C
from model.Base import Sequential
from model.compat import cudnn_rnn, extender


class GRU4REC(Sequential):
    """ Implementation of the paper ---
        Hidasi B, Karatzoglou A, Baltrunas L, Tikk D.
        Session-based recommendations with recurrent neural networks.
        ICLR 2016.
    """

    def __init__(self, num_items, FLAGS):
        super().__init__(num_items, FLAGS)

        with tf.variable_scope("GRU4REC"):
            self.item_embs = C.Embedding(self.num_items, self.num_units, self.l2_reg,
                                         zero_pad=True, scale=True, scope="item_embs")

            self._cudnn_rnn = cudnn_rnn.CudnnGRU(
                num_layers=FLAGS.num_blocks, num_units=self.num_units, direction='unidirectional',
                kernel_initializer=tf.orthogonal_initializer(), name='GRU4REC/GRU')

            self.output_bias = self.output_bias(inf_pad=True)

    def __call__(self, features, is_training):
        seqs_id = features['seqs_i']
        seqs_units = self.item_embs(seqs_id)

        # Dropout
        seqs_units = tf.layers.dropout(seqs_units, rate=self.hidden_dropout_rate,
                                       training=tf.convert_to_tensor(is_training))
        seqs_masks = tf.expand_dims(tf.to_float(tf.not_equal(seqs_id, 0)), -1)

        # Recurrency
        with tf.variable_scope("S2PNM/Reccurency"):
            h, _ = self._cudnn_rnn(tf.transpose(seqs_units, [1, 0, 2]))
            h = tf.transpose(h, [1, 0, 2])  # mask the hidden states as dynamic_rnn did

        seqs_outs = h * seqs_masks

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
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta2=0.98, epsilon=1e-9)
        optimizer = extender.clip_gradients_by_norm(optimizer, clip_norm=5.)
        with tf.control_dependencies([add_gstep]):
            train_op = optimizer.minimize(loss)
        return train_op
