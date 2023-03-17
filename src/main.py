"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import argparse
import itertools
import logging.config

import os
import tqdm
import random

import numpy as np
import tensorflow.compat.v1 as tf

from util import ranking, reader, EarlyStopping

tf.disable_v2_behavior()


def args():
    parser = argparse.ArgumentParser(description='Continuous-Time Self-Modulating Attention (CTSMA)')
    parser.add_argument('--train', action="store", required=True,
                        help="training data file patterns")
    parser.add_argument('--valid', action="store", required=True,
                        help="validation data file patterns")
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

    # ---- for unit to scale down the timestamps
    parser.add_argument('--time_scale', type=float, default=1)

    # ---- for masking
    parser.add_argument('--masklen', type=int, default=6)

    # ---- for GREC
    parser.add_argument('--filter_width', type=int, default=3)
    parser.add_argument('--dilations', type=str, default="1,2,2,4")

    # ---- for TiSASREC, discrete time encoding
    parser.add_argument('--timelen', type=int, default=256)

    # ---- for TimelyREC
    parser.add_argument('--window_ratio', type=float, default=0.2)

    # ---- for CTSMA/EasyDGL
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
    parser.add_argument('--eval_per_steps', type=int, default=1)
    parser.add_argument('--mask_seen', action="store_true", dest="mask_seen")
    parser.set_defaults(l2_reg=0., mask_seen=False)
    return parser.parse_args()


def main():
    logging.info("1. build data pipeline")
    with tf.device('/cpu:0'):
        tr_reader = reader(FLAGS, FLAGS.train, is_training=True)
        tr_data = tr_reader(FLAGS.batch_size).make_initializable_iterator()

        vl_reader = reader(FLAGS, FLAGS.valid, is_training=False)
        vl_data = vl_reader(FLAGS.batch_size).make_initializable_iterator()

        te_reader = reader(FLAGS, FLAGS.test, is_training=False)
        te_data = te_reader(FLAGS.batch_size).make_initializable_iterator()

    logging.info("2. create neural model")
    with tf.variable_scope("main"):
        m = ranking(FLAGS)
        features, labels = tr_data.get_next()
        train_op, loss_op, loss_init_op = m.train(features, labels)

        # reuse variables for the next tower.
        tf.get_variable_scope().reuse_variables()

        # used for validation
        features, labels = vl_data.get_next()
        vl_metrics_op, vl_metric_init_op = m.eval(features, labels, mask_seen=FLAGS.mask_seen)

        # used for testing
        features, labels = te_data.get_next()
        te_metrics_op, te_metric_init_op = m.eval(features, labels, mask_seen=FLAGS.mask_seen)

    logging.info("3. train and evaluate model")
    stopper = EarlyStopping(FLAGS)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for epoch in range(FLAGS.num_epochs):
            sess.run([loss_init_op, tr_data.initializer])
            with tqdm.tqdm(itertools.count(), ascii=True) as tq:
                try:
                    for step in tq:
                        _, running_loss = sess.run([train_op, loss_op])
                        if step % 10 == 0:
                            tq.set_postfix(loss='{0:.5f}'.format(running_loss))
                except tf.errors.OutOfRangeError:
                    logging.info("%03d: Loss=%.4f" % (epoch, running_loss))

            if epoch % FLAGS.eval_per_steps:
                continue

            # Validation: calculate the ranking metrics
            sess.run([vl_metric_init_op, vl_data.initializer])
            with tqdm.tqdm(itertools.count(), ascii=True) as tq:
                try:
                    for _ in tq:
                        vl_metrics = sess.run(vl_metrics_op)
                except tf.errors.OutOfRangeError:
                    logging.info("%03d: %s" % (epoch, {k: "{0:.5f}".format(v) for k, v in vl_metrics.items()}))

            # Testing: calculate the ranking metrics
            sess.run([te_metric_init_op, te_data.initializer])
            with tqdm.tqdm(itertools.count(), ascii=True) as tq:
                try:
                    for _ in tq:
                        te_metrics = sess.run(te_metrics_op)
                except tf.errors.OutOfRangeError:
                    pass

            stopper.step(running_loss, vl_metrics['H100'], vl_metrics, te_metrics, sess)  # focused on HR[changable]
            # stopping when no performance gain is achieved
            if stopper.early_stop:
                break
    stopper.summary()


if __name__ == "__main__":
    logging.config.fileConfig('./conf/logging.conf')

    SEED = 9876
    np.random.seed(SEED)
    tf.random.set_random_seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    os.environ['TF_DETERMINISTIC_OPS'] = str(1)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(1)
    os.environ['HOROVOD_FUSION_THRESHOLD'] = str(0)

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    FLAGS = args()
    logging.info("================")
    logging.info(vars(FLAGS))
    logging.info("================")

    main()
