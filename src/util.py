"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging

import numpy as np
import tensorflow.compat.v1 as tf

import dataloader as D


class EarlyStopping(object):
    def __init__(self, FLAGS, patience=10):
        self.model = FLAGS.model
        self.patience = patience

        self.counter = 0
        self.res = None
        self.best_valid = None
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

        self.saver = tf.train.Saver(max_to_keep=1)

    def step(self, loss, acc, valid: dict, test: dict, sess=None):
        if np.isnan(loss):
            self.early_stop = True
        elif self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.best_valid = valid
            self.res = test
        elif acc < self.best_acc:  # count if better acc is not achieved
            self.counter += 1
            logging.info(f'EarlyStopping {self.model} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = min(loss, self.best_loss)
            self.best_acc = max(acc, self.best_acc)
            # refresh the best results on the evaluation and testing data
            for k, v in self.res.items():
                if self.best_valid[k] <= valid[k]:
                    self.res[k] = test[k]
            self.counter = 0
            self.save_ckpt(sess)

        return self.early_stop

    def save_ckpt(self, sess):
        if sess is not None:
            self.saver.save(sess, f"ckpt/{self.model}")

    def summary(self):
        logging.info("SUMMARY: %s" % ({k: "{0:.5f}".format(v) for k, v in self.res.items()}))


def ranking(FLAGS):
    if FLAGS.model == "SASREC":
        from model import SASRec
        return SASRec(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "TiSASREC":
        from model import TiSASRec
        return TiSASRec(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "TGAT":
        from model import TGAT
        return TGAT(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "S2PNM":
        from model import S2PNM
        return S2PNM(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "GRU4REC":
        from model import GRU4REC
        return GRU4REC(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "BERT4REC":
        from model import BERT4REC
        return BERT4REC(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "GREC":
        from model import GREC
        return GREC(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "TimelyREC":
        from model import TimelyREC
        return TimelyREC(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "TGREC":
        from model import TGREC
        return TGREC(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "CTSMA":
        from model import CTSMA
        return CTSMA(FLAGS.num_items, FLAGS)
    elif FLAGS.model == "EasyDGL":
        from model import EasyDGL
        return EasyDGL(FLAGS.num_items, FLAGS)
    else:
        raise NotImplementedError("The ranking model: {0} not implemented".format(FLAGS.model))


def reader(FLAGS, file_pattern, is_training: bool):
    if FLAGS.model == 'BERT4REC':
        seqslen = FLAGS.seqslen + 1
        mask = FLAGS.num_items
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(seqslen, has_datetime=False),
                             processor=D.MaskedPostProcessor(seqslen, FLAGS.masklen, mask, is_training))
    elif FLAGS.model == 'GREC':
        seqslen = FLAGS.seqslen + 1
        mask = FLAGS.num_items
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(seqslen),
                             processor=D.GRECPostProcessor(seqslen, FLAGS.masklen, mask, is_training))
    elif FLAGS.model == 'TimelyREC':
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(FLAGS.seqslen + 1, has_datetime=True),
                             processor=D.RegressivePostProcessor(is_training, has_datetime=True))
    elif FLAGS.model == 'CTSMA':
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(FLAGS.seqslen + 1, has_datetime=False),
                             processor=D.RegressivePostProcessor(is_training, has_datetime=False, keep_entire=True))
    elif FLAGS.model == 'EasyDGL':
        seqslen = FLAGS.seqslen + 1
        mask = FLAGS.num_items
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(seqslen, has_datetime=False),
                             processor=D.MAUPostProcessor(seqslen, FLAGS.masklen, mask, is_training))
    else:
        return D.InputReader(file_pattern, is_training=is_training,
                             decoder=D.TfExampleDecoder(FLAGS.seqslen + 1),
                             processor=D.RegressivePostProcessor(is_training))
