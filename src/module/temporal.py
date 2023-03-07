"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import tensorflow.compat.v1 as tf

from module.normalize import layernorm


def silu(x, beta=1.0):
    return x * tf.nn.sigmoid(beta * x)


class TiMultiHeadAttention(object):
    """ Implementation of the paper ---
    Li J, Wang Y, McAuley J.
    Time interval aware self-attention for sequential recommendation.
    WSDM 2020.
    """

    def __init__(self, num_units, num_heads, dropout_rate, l2_reg,
                 pcoding_K, pcoding_V, tcoding_K, tcoding_V,
                 scope="attention/timeinterval"):
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.scope = scope
        self.l2_reg = l2_reg

        self.pcoding_K = pcoding_K
        self.pcoding_V = pcoding_V
        self.tcoding_K = tcoding_K
        self.tcoding_V = tcoding_V

    def __call__(self, queries, keys, intervals, is_training, causality):
        is_training = tf.convert_to_tensor(is_training)
        num_units, num_heads, dropout_rate = self.num_units, self.num_heads, self.dropout_rate

        with tf.variable_scope(self.scope):
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            K_embs_p = tf.concat(tf.split(self.pcoding_K.code(queries), num_heads, axis=2), axis=0)
            V_embs_p = tf.concat(tf.split(self.pcoding_V.code(queries), num_heads, axis=2), axis=0)
            K_embs_t = tf.concat(tf.split(self.tcoding_K.code(intervals), num_heads, axis=3), axis=0)
            V_embs_t = tf.concat(tf.split(self.tcoding_V.code(intervals), num_heads, axis=3), axis=0)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs_p = tf.matmul(Q_, tf.transpose(K_embs_p, [0, 2, 1]))
            outputs_t = tf.squeeze(tf.matmul(K_embs_t, tf.expand_dims(Q_, axis=3)))
            outputs = outputs + outputs_p + outputs_t

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
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
            outputs = tf.nn.softmax(outputs)

            # Query Masking
            query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

            # Weighted sum
            outputs_value = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs_value_p = tf.matmul(outputs, V_embs_p)
            outputs_value_t = tf.squeeze(tf.matmul(tf.expand_dims(outputs, axis=2), V_embs_t), axis=2)
            outputs = outputs_value + outputs_value_p + outputs_value_t

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            return outputs


class TfMultiHeadAttention(object):
    """Implementation of the paper ---
    Xu D, Ruan C, Korpeoglu E, Kumar S, Achan K.
    Inductive representation learning on temporal graphs.
    ICLR 2020.
    """

    def __init__(self, num_units, num_heads, dropout_rate, l2_reg,
                 pcoding_K, tcoding_K, scope="attention/timeinterval"):
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.scope = scope
        self.l2_reg = l2_reg

        self.pcoding_K = pcoding_K
        self.tcoding_K = tcoding_K

    def __call__(self, queries, keys, intervals, is_training, causality):
        is_training = tf.convert_to_tensor(is_training)
        num_units, num_heads, dropout_rate = self.num_units, self.num_heads, self.dropout_rate

        with tf.variable_scope(self.scope):
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            K_embs_p = tf.concat(tf.split(self.pcoding_K.code(queries), num_heads, axis=2), axis=0)
            K_embs_t = tf.concat(tf.split(self.tcoding_K.code(intervals), num_heads, axis=3), axis=0)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs_p = tf.matmul(Q_, tf.transpose(K_embs_p, [0, 2, 1]))
            outputs_t = tf.squeeze(tf.matmul(K_embs_t, tf.expand_dims(Q_, axis=3)))
            outputs = outputs + outputs_p + outputs_t

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
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
            outputs = tf.nn.softmax(outputs)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            return outputs


class TgMultiHeadAttention(object):
    """The implementation of ---
    Fan, Ziwei and Liu, Zhiwei and Zhang, Jiawei and Xiong, Yun and Zheng, Lei and Yu, Philip S.
    Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer.
    CIKM 2021.
    """

    def __init__(self, num_units, num_heads, dropout_rate, l2_reg,
                 tcoding, scope="attention/TgMultiHeadAttention"):
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.scope = scope
        self.l2_reg = l2_reg

        self.tcoding = tcoding

    def __call__(self, queries, keys, masks, intervals, is_training, causality):
        is_training = tf.convert_to_tensor(is_training)
        num_units, num_heads, dropout_rate = self.num_units, self.num_heads, self.dropout_rate

        with tf.variable_scope(self.scope):
            queries_t = self.tcoding.code(tf.zeros_like(intervals[:, :, :1], dtype=tf.float32))  # (N, T_q, 1, C)
            queries = tf.expand_dims(queries, axis=2)  # (N, T_q, 1, C)
            queries = tf.concat([queries, queries_t], axis=-1)  # (N, T_q, 1, 2C)

            keys_t = self.tcoding.code(intervals)  # (N, T_q, T_k, C)
            keys = tf.tile(tf.expand_dims(keys, axis=1), [1, tf.shape(queries)[1], 1, 1])  # (N, T_q, T_k, C)
            keys = tf.concat([keys, keys_t], axis=-1)  # (N, T_q, T_k, 2C)

            # Set the fall back option for num_units
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, 1, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_q, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_q, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=3), axis=0)  # (h*N, T_q, 1, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=3), axis=0)  # (h*N, T_q, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=3), axis=0)  # (h*N, T_q, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, 1, T_k)
            outputs = tf.squeeze(outputs, axis=2)  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

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
            outputs = tf.expand_dims(outputs, axis=2)  # (h*N, T_q, 1, T_k)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, Tk, C/h)
            outputs = tf.squeeze(outputs, axis=2)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual Connections
            outputs = tf.layers.dense(outputs, 2 * self.num_units)
            outputs += tf.squeeze(queries, axis=2)
            outputs = layernorm(outputs, "ln")
            return outputs


class MAU(object):
    """ Implementation of the paper ---
    C Chen, H Geng, N Yang, J Yan, D Xue, J Yu, X Yang.
    Learning Self-Modulating Attention in Continuous Time Space with Applications to Sequential Recommendation.
    ICML 2021.
    """

    def __init__(self, num_units, num_heads, num_events, dropout_rate, scope="modulating_attention"):
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_events = num_events
        self.dropout_rate = dropout_rate
        self.scope = scope

    def intensity(self, H, intervals, mark_onehot):
        num_units, num_heads, num_events = self.num_units, self.num_heads, self.num_events
        intervals = tf.tile(tf.expand_dims(intervals, axis=-1), [num_heads, 1, 1])

        # E: number of events, C: number of channels, h: number of hidden units
        # N: batch size
        layer_inputs = tf.concat([H, intervals], axis=-1)
        with tf.variable_scope("sequential_temporal_combined"):
            layers_outputs = tf.layers.dense(layer_inputs, num_units // num_heads * num_events,
                                             activation=tf.nn.sigmoid)  # (h*N, T_q,  h/C*E)
            layers_outputs = tf.concat(tf.split(layers_outputs, num_events, axis=2), axis=0)  # (h*N*E, T_q, C/h)

            weight = tf.get_variable("weight", trainable=True, shape=[num_events, num_units // num_heads],
                                     initializer=tf.glorot_uniform_initializer())
            weight = tf.reshape(weight, shape=[num_events, 1, num_units // num_heads, 1])
            weight = tf.tile(weight, [1, tf.shape(H)[0], 1, 1])
            weight = tf.reshape(weight, [num_events * tf.shape(H)[0], num_units // num_heads, 1])  # (h*N*E, h/C, 1)

            scaling = tf.get_variable("scaling", trainable=True, shape=[num_events],
                                      initializer=tf.zeros_initializer())
            scaling = tf.reshape(tf.exp(scaling), shape=[num_events, 1, 1, 1])
            scaling = tf.tile(scaling, [1, tf.shape(H)[0], 1, 1])
            scaling = tf.reshape(scaling, [num_events * tf.shape(H)[0], 1, 1])  # (h*N*E, 1, 1)

            mark_intensity = tf.matmul(layers_outputs, weight) / scaling
            mark_intensity = scaling * tf.log(1. + tf.exp(mark_intensity))  # (h*N*E, T_q, 1)
            mark_intensity = tf.concat(tf.split(mark_intensity, num_events, axis=0), axis=2)  # (h*N, T_q, E)

            mark_intensity_4d = tf.tile(tf.expand_dims(mark_intensity, 2),
                                        [1, 1, tf.shape(H)[1], 1])  # (h*N, T_q, T_k, E)
            mark_onehot_4d = tf.tile(tf.expand_dims(tf.to_float(mark_onehot), 1),
                                     [num_heads, tf.shape(H)[1], 1, 1])  # (h*N, T_q, T_k, E)
            mark_intensity_4d = tf.reduce_sum(mark_intensity_4d * mark_onehot_4d, axis=-1)  # (h*N, T_q, T_k)

        return mark_intensity_4d, mark_intensity

    @classmethod
    def biased_likelihood(cls, mark_intensity, next_mark_onehot, intervals):
        # mark_intensity: marked intensity for every kind of events (h*N, T_q, E)
        # next_mark_onehot: the mark for the next item (h*N, T_q, E)
        mark_intensity *= tf.sign(tf.reduce_sum(next_mark_onehot, axis=2, keepdims=True))
        event_intensity = tf.reduce_sum(mark_intensity * next_mark_onehot, axis=2)

        event_ll = tf.log(tf.where(tf.equal(event_intensity, 0), tf.ones_like(event_intensity), event_intensity))
        event_ll = tf.reduce_sum(event_ll)

        entire_intensity = tf.reduce_sum(mark_intensity, axis=2)
        nu_integral = entire_intensity * intervals * .5
        non_event_ll = tf.reduce_sum(nu_integral)

        num_events = tf.reduce_sum(next_mark_onehot)
        biased_mle = -tf.reduce_sum(event_ll - non_event_ll) / num_events
        return biased_mle

    def __call__(self, queries, keys, masks, intervals, marks, is_training, causality):
        num_units, num_heads, dropout_rate = self.num_units, self.num_heads, self.dropout_rate

        with tf.variable_scope(self.scope):
            # Set the fall back option for num_units
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            T = tf.layers.dense(keys, num_units, activation=None)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            T_ = tf.concat(tf.split(T, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Weighted sum
            # marked_intensity_4d: marked intensity for every past events (h*N, T_q, T_k)
            # mark_intensity: marked intensity for every kind of events (h*N, T_q, E)
            sequential_units = tf.matmul(outputs, T_)  # ( h*N, T_q, C/h)
            marked_intensity_4d, mark_intensity = self.intensity(sequential_units, intervals, marks)

            # Dropouts
            outputs = marked_intensity_4d * outputs
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries[:, :, :num_units]

            # Normalize
            # outputs = normalize(outputs) # (N, T_q, C)

            return outputs, mark_intensity


WEIGHT_INITIALIZER = tf.random_normal_initializer(stddev=0.02)


class BiMAU(MAU):
    """ Implementation of the paper ---
    Bi-lelve Modulating Attention Unit (BiMAU)
    """

    def __init__(self, num_units, num_heads, num_events, dropout_rate, scope="TMAU"):
        super().__init__(num_units, num_heads, num_events, dropout_rate, scope)

    def __call__(self, queries, keys, masks, intervals, marks, is_training, causality=None):
        num_units, num_heads, dropout_rate = self.num_units, self.num_heads, self.dropout_rate

        with tf.variable_scope(self.scope):
            # Set the fall back option for num_units
            QKVT = tf.layers.dense(queries, 4 * num_units, activation=None, kernel_initializer=WEIGHT_INITIALIZER)
            Q, K, V, T = tf.split(QKVT, [num_units, num_units, num_units, num_units], axis=-1)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            T_ = tf.concat(tf.split(T, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Weighted sum
            # marked_intensity_4d: marked intensity for every past events (h*N, T_q, T_k)
            # mark_intensity: marked intensity for every kind of events (h*N, T_q, E)
            sequential_units = tf.matmul(outputs, T_)  # ( h*N, T_q, C/h)
            marked_intensity_4d, mark_intensity = self.intensity(sequential_units, intervals, marks)

            # Dropouts
            marked_intensity_diag_ones = tf.ones_like(marked_intensity_4d[:, :, 0])
            marked_intensity_4d = tf.linalg.set_diag(marked_intensity_4d, marked_intensity_diag_ones)

            outputs = marked_intensity_4d * outputs
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries[:, :, :num_units]

            # Normalize
            # outputs = normalize(outputs) # (N, T_q, C)

            return outputs, mark_intensity


class MGAU(MAU):
    """ Implementation of the paper ---
    Modulating Gated Attention Unit (MGAU)
    """

    def __init__(self, num_units, num_heads, num_events, dropout_rate, scope="GAU_SMA"):
        super().__init__(num_units, num_heads, num_events, dropout_rate, scope)

    def __call__(self, queries, keys, masks, intervals, marks, is_training, causality=None):
        num_units, num_heads, dropout_rate = self.num_units, self.num_heads, self.dropout_rate

        with tf.variable_scope(self.scope):
            # Set the fall back option for num_units
            QKVT = tf.layers.dense(queries, 4 * num_units, activation=None, kernel_initializer=WEIGHT_INITIALIZER)
            Q, K, V, T = tf.split(QKVT, [num_units, num_units, num_units, num_units], axis=-1)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            T_ = tf.concat(tf.split(T, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Weighted sum
            # marked_intensity_4d: marked intensity for every past events (h*N, T_q, T_k)
            # mark_intensity: marked intensity for every kind of events (h*N, T_q, E)
            sequential_units = tf.matmul(outputs, T_)  # ( h*N, T_q, C/h)
            marked_intensity_4d, mark_intensity = self.intensity(sequential_units, intervals, marks)

            # Dropouts
            outputs = marked_intensity_4d * outputs
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries[:, :, :num_units]

            # Normalize
            # outputs = normalize(outputs) # (N, T_q, C)

            return outputs, mark_intensity
