"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""

import numpy as np
import tensorflow.compat.v1 as tf


class TfExampleDecoder(object):
    """Tensorflow Example proto decoder."""

    def __init__(self, seqslen, has_labels=False, has_datetime=False):
        self._keys_to_features = {
            "seqs_i": tf.FixedLenFeature([seqslen], tf.int64),
            "seqs_t": tf.FixedLenFeature([seqslen], tf.float32)}

        if has_labels:
            self._keys_to_features['labels'] = tf.FixedLenFeature([seqslen], tf.int64)

        if has_datetime:
            self._keys_to_features['seqs_month'] = tf.FixedLenFeature([seqslen], tf.int64)
            self._keys_to_features['seqs_day'] = tf.FixedLenFeature([seqslen], tf.int64)
            self._keys_to_features['seqs_weekday'] = tf.FixedLenFeature([seqslen], tf.int64)
            self._keys_to_features['seqs_hour'] = tf.FixedLenFeature([seqslen], tf.int64)

    def decode(self, serialized_example):
        parsed_tensors = tf.io.parse_single_example(
            serialized_example, self._keys_to_features)
        return parsed_tensors


def choice(seqslen, ignore_head, maskslen):
    masked_positions = np.random.choice(seqslen - ignore_head, maskslen, replace=False)
    return masked_positions + ignore_head


class MaskedPostProcessor(object):
    def __init__(self, seqslen: int, maskslen: int, mask: int, is_training):
        self.seqslen = seqslen
        self.maskslen = maskslen
        self.mask = mask
        self.is_training = is_training

    def mask_last(self, decoded_tensors: dict):
        tokens = decoded_tensors['seqs_i']
        masked_positions_indicator = tf.one_hot(self.seqslen - 1, self.seqslen, dtype=tf.int64)
        masked_tokens = masked_positions_indicator * (self.mask - tokens) + tokens

        timestamps = decoded_tensors['seqs_t']

        features = {
            # the masked sequence tokens
            'seqs_i': masked_tokens,
            # the sequence timestamps
            'seqs_t': timestamps}
        labels = tokens
        return features, labels

    def mask_random(self, decoded_tensors: dict):
        masked_positions = tf.py_func(choice, [self.seqslen, 0, self.maskslen], [tf.int64])
        masked_positions = tf.reshape(masked_positions, [self.maskslen])

        tokens = decoded_tensors['seqs_i']
        masked_positions_indicator = tf.one_hot(masked_positions, self.seqslen, dtype=tf.int64)
        masked_positions_indicator = tf.reduce_sum(masked_positions_indicator, axis=0)
        masked_tokens = masked_positions_indicator * (self.mask - tokens) + tokens

        timestamps = decoded_tensors['seqs_t']

        features = {
            # the masked sequence tokens
            'seqs_i': masked_tokens,
            # the positions for masked tokens
            'masked_positions': masked_positions,
            # the sequence timestamps
            'seqs_t': timestamps}
        labels = tf.gather(tokens, masked_positions)
        return features, labels

    def __call__(self, decoded_tensors: dict):
        if not self.is_training:
            return self.mask_last(decoded_tensors)
        return self.mask_random(decoded_tensors)


class RegressivePostProcessor(object):

    def __init__(self, is_training, has_datetime=False, keep_entire=False):
        self.has_datetime = has_datetime
        self.keep_entire = keep_entire
        self.is_training = is_training

    def __call__(self, decoded_tensors: dict):
        tokens = decoded_tensors['seqs_i']
        timestamps = decoded_tensors['seqs_t']

        features = {'seqs_i': tokens[:-1], 'seqs_t': timestamps}
        labels = tokens[1:] if self.is_training else tokens

        if self.has_datetime:
            features['seqs_month'] = decoded_tensors['seqs_month'][:-1]
            features['seqs_day'] = decoded_tensors['seqs_day'][:-1]
            features['seqs_weekday'] = decoded_tensors['seqs_weekday'][:-1]
            features['seqs_hour'] = decoded_tensors['seqs_hour'][:-1]

        return features, labels


class GRECPostProcessor(object):
    def __init__(self, seqslen: int, maskslen: int, mask: int, is_training: bool):
        self.seqslen = seqslen
        self.maskslen = maskslen
        self.mask = mask
        self.is_training = is_training

    def mask_random(self, decoded_tensors: dict):
        ignore_head = 1  # True, ignore the head of the sequence

        # masked_positions: 1, 3
        masked_positions = tf.py_func(choice, [self.seqslen, ignore_head, self.maskslen], [tf.int64])
        masked_positions = tf.reshape(masked_positions, [self.maskslen])

        # tokens: 1, 2, 3, 4, 5
        tokens = decoded_tensors['seqs_i']

        # masked_tokens: 1, MASK, 3, MASK, 5
        masked_positions_indicator = tf.one_hot(masked_positions, self.seqslen, dtype=tf.int64)
        masked_positions_indicator = tf.reduce_sum(masked_positions_indicator, axis=0)
        masked_tokens = masked_positions_indicator * (self.mask - tokens) + tokens

        # prediction_positions: in auto-regressive fashion
        #   1st MASK,    1 -> MASK
        #   2nd MASK,    1, MASK, 3 -> MASK
        prediction_positions = masked_positions - 1

        features = {
            'seqs_i': tokens,  # the orignal sequence tokens
            'seqs_m': masked_tokens,  # the masked sequence tokens
            'masked_positions': prediction_positions  # the positions to yield predictions
        }
        labels = tf.gather(tokens, masked_positions)
        return features, labels

    def __call__(self, decoded_tensors: dict):
        if not self.is_training:
            tokens = decoded_tensors['seqs_i'][:-1]
            features = {
                'seqs_i': tokens,  # the orignal sequence tokens
                'seqs_m': tokens,  # the masked sequence tokens
            }
            labels = decoded_tensors['seqs_i'][-1:]
            return features, labels

        return self.mask_random(decoded_tensors)


class MAUPostProcessor(object):
    def __init__(self, seqslen: int, maskslen: int, mask: int, is_training):
        self.seqslen = seqslen
        self.maskslen = maskslen
        self.mask = mask
        self.is_training = is_training

    def mask_last(self, decoded_tensors: dict):
        tokens = decoded_tensors['seqs_i']
        masked_positions_indicator = tf.one_hot(self.seqslen - 1, self.seqslen, dtype=tf.int64)
        masked_tokens = masked_positions_indicator * (self.mask - tokens) + tokens

        timestamps = decoded_tensors['seqs_t']
        seqs_month = decoded_tensors['seqs_month']
        seqs_weekday = decoded_tensors['seqs_weekday']

        features = {
            # the masked sequence tokens
            'seqs_i': masked_tokens,
            # the sequence timestamps
            'seqs_t': timestamps,
            'seqs_month': seqs_month,
            'seqs_weekday': seqs_weekday}
        labels = tokens
        return features, labels

    def mask_random(self, decoded_tensors: dict):
        ignore_head = 1  # True, ignore the head of the sequence
        masked_positions = tf.py_func(choice, [self.seqslen, ignore_head, self.maskslen], [tf.int64])
        masked_positions = tf.reshape(masked_positions, [self.maskslen])

        tokens = decoded_tensors['seqs_i']
        masked_positions_indicator = tf.one_hot(masked_positions, self.seqslen, dtype=tf.int64)
        masked_positions_indicator = tf.reduce_sum(masked_positions_indicator, axis=0)
        masked_tokens = masked_positions_indicator * (self.mask - tokens) + tokens

        timestamps = decoded_tensors['seqs_t']
        seqs_month = decoded_tensors['seqs_month']
        seqs_weekday = decoded_tensors['seqs_weekday']

        features = {
            # the masked sequence tokens
            'seqs_i': masked_tokens,
            # the positions for masked tokens
            'masked_positions': masked_positions,
            # the sequence timestamps
            'seqs_t': timestamps,
            'seqs_month': seqs_month,
            'seqs_weekday': seqs_weekday}
        labels = tf.gather(tokens, masked_positions)
        return features, labels

    def __call__(self, decoded_tensors: dict):
        if not self.is_training:
            return self.mask_last(decoded_tensors)
        return self.mask_random(decoded_tensors)


class InputReader(object):
    def __init__(self, file_pattern, is_training, decoder, processor=None):
        self._file_pattern = file_pattern
        self._is_training = is_training

        assert decoder is not None, "decoder is not specified"
        self.decoder = decoder

        assert processor is not None, "postprocessor is not specified"
        self.processor = processor

    def __call__(self, batch_size) -> tf.data.Dataset:
        dataset = tf.data.Dataset.list_files(
            self._file_pattern, shuffle=self._is_training)

        # Prefetch data from files.
        def _prefetch_dataset(filename):
            dataset = tf.data.TFRecordDataset(filename).prefetch(1)
            return dataset

        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                _prefetch_dataset, cycle_length=32, sloppy=self._is_training))
        if self._is_training:
            dataset = dataset.shuffle(64)

        def _parse_example(value):
            """Processes one batch of data."""
            with tf.name_scope('parser'):
                decoded_tensors = self.decoder.decode(value)
                features, labels = self.processor(decoded_tensors)
                return features, labels

        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                _parse_example, batch_size, num_parallel_batches=64))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
