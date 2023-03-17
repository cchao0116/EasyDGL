"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import pickle
import argparse
import itertools
import logging.config
import os
import random

import numpy as np
import tensorflow.compat.v1 as tf

import tqdm

from util import ranking, reader

tf.disable_v2_behavior()


def args():
    parser = argparse.ArgumentParser(description='Continuous-Time Self-Modulating Attention (CTSMA)')
    parser.add_argument('--ckpt', action="store", required=True,
                        help="training data file patterns", )
    parser.add_argument('--test', action="store", required=True,
                        help="test data file patterns")
    parser.add_argument('--model', action="store", required=True,
                        help="algorithm names")
    parser.add_argument('--num_items', type=int, required=True)

    # ---- for definition
    parser.add_argument('--num_units', type=int, default=50)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--seqslen', type=int, default=30)
    parser.add_argument('--timelen', type=int, default=256)
    parser.add_argument('--time_scale', type=float, default=1)

    # ---- for GREC
    parser.add_argument('--masklen', type=int, default=6)
    parser.add_argument('--filter_width', type=int, default=3)
    parser.add_argument('--dilations', type=str, default="1,2,2,4")

    # ---- for TimelyREC
    parser.add_argument('--window_ratio', type=float, default=0.2)

    # ---- for CTSMA
    parser.add_argument('--mark', type=str, help="mark data file")
    parser.add_argument('--ct_reg', type=float, default=0.)

    # ---- for optimization
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--l2_reg', type=float)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.)
    parser.add_argument('--attention_probs_dropout_rate', type=float, default=0.)
    parser.add_argument('--num_train_steps', type=int)
    parser.add_argument('--num_warmup_steps', type=int)

    # ---- for evaluation
    #     parser.add_argument('--topN', type=int, default=50)
    parser.add_argument('--mask_seen', action="store_true", dest="mask_seen")
    parser.set_defaults(l2_reg=0., mask_seen=False)
    return parser.parse_args()


def main():
    logging.info("1. build data pipeline")
    with tf.device('/cpu:0'):
        te_reader = reader(FLAGS, FLAGS.test, is_training=False)
        te_data = te_reader(FLAGS.batch_size).make_initializable_iterator()

    logging.info("2. create neural model")
    with tf.variable_scope("main"):
        m = ranking(FLAGS)
        features, labels = te_data.get_next()
        metrics_op, metric_init_op = m.eval(features, labels, mask_seen=FLAGS.mask_seen)

    logging.info("3. train and evaluate model")
    saver = tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt)

        logits, analytics_op = list(), tf.get_collection("ANALYTICS")[0]

        sess.run([metric_init_op, te_data.initializer])
        with tqdm.tqdm(itertools.count(), ascii=True) as tq:
            try:
                for _ in tq:
                    metrics, analytics = sess.run([metrics_op, analytics_op])
                    logits.append(analytics)
            except tf.errors.OutOfRangeError:
                logging.info("%s" % ({k: "{0:.5f}".format(v) for k, v in metrics.items()}))
            pickle.dump(logits, open("res", 'wb'))


if __name__ == "__main__":
    logging.config.fileConfig('./conf/logging.conf')

    SEED = 9876
    np.random.seed(SEED)
    tf.random.set_random_seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    FLAGS = args()
    logging.info("================")
    logging.info(vars(FLAGS))
    logging.info("================")

    main()
