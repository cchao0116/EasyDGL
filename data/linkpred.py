"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import argparse
import logging.config
import os

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf


class TrainingInstance(object):
    def __init__(self, tokens, timestamps, seqs_month,
                 seqs_day, seqs_weekday, seqs_hour):
        self.tokens = tokens
        self.timestamps = timestamps

        self.seqs_month = seqs_month
        self.seqs_day = seqs_day
        self.seqs_weekday = seqs_weekday
        self.seqs_hour = seqs_hour

    def tfexample(self):
        def fint(f_vals):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=f_vals))

        def ffloat(f_vals):
            return tf.train.Feature(float_list=tf.train.FloatList(value=f_vals))

        example_tr = tf.train.Example(
            features=tf.train.Features(
                feature={'seqs_i': fint(self.tokens), 'seqs_t': ffloat(self.timestamps),
                         'seqs_month': fint(self.seqs_month), 'seqs_day': fint(self.seqs_day),
                         'seqs_weekday': fint(self.seqs_weekday), 'seqs_hour': fint(self.seqs_hour)}
            ))
        return example_tr.SerializeToString()


class TripletDataset:
    """transform pandas-like data e.g., uid,sid,cid,time

    - splitting training/validation/test set.

    - preprocess the data in TFRECORD format.
    """

    @classmethod
    def filter(cls, tp: pd.DataFrame, min_uc: int, min_sc: int) -> pd.DataFrame:
        # Only keep the quartets for items which were clicked on by at least min_sc users.
        if min_sc > 0:
            itemcount = tp['sid'].value_counts()
            tp = tp[tp['sid'].isin(itemcount.index[itemcount >= min_sc])]

        # Only keep the quartets for users who clicked on at least min_uc items
        # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
        if min_uc > 0:
            usercount = tp['uid'].value_counts()
            tp = tp[tp['uid'].isin(usercount.index[usercount >= min_uc])]

        return tp

    @classmethod
    def numerize(cls, tp: pd.DataFrame, profile2id, show2id) -> pd.DataFrame:
        uid = tp['uid'].map(lambda x: profile2id[x]).values
        sid = tp['iid'].map(lambda x: show2id[x]).values
        tid = tp['time'].values

        tp = pd.DataFrame(data={'uid': uid, 'sid': sid, 'time': tid}, columns=['uid', 'sid', 'time'])
        tp.sort_values(by=['uid', 'time'], ascending=True, inplace=True)
        return tp

    @classmethod
    def csv(cls, train_plays: pd.DataFrame, vad_plays: pd.DataFrame, test_plays: pd.DataFrame,
            fout: int, n_test_items=1):
        """

        Parameters
        ----------
        train_plays
        vad_plays
        test_plays
        fout
        n_test_items

        Returns
        -------

        """

        def split_train_test_timeseries(data):
            data_grouped_by_user = data.groupby('uid')
            tr_list, te_list = list(), list()

            for i, (_, group) in enumerate(data_grouped_by_user):
                n_items_u = len(group)

                idx = np.zeros(n_items_u, dtype='bool')
                idx[-n_test_items:] = True

                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])

                i += 1
                if i % 5000 == 0:
                    logging.info("%d users sampled" % i)

            data_tr = pd.concat(tr_list)
            data_te = pd.concat(te_list)
            return data_tr, data_te

        logging.info("yielding validation data.")
        vad_plays_tr, vad_plays_te = split_train_test_timeseries(vad_plays)
        logging.info("yielding test data.")
        test_plays_tr, test_plays_te = split_train_test_timeseries(test_plays)

        train_plays.to_csv(os.path.join(fout, 'train.csv'), index=False)
        vad_plays_tr.to_csv(os.path.join(fout, 'validation_tr.csv'), index=False)
        vad_plays_te.to_csv(os.path.join(fout, 'validation_te.csv'), index=False)
        test_plays_tr.to_csv(os.path.join(fout, 'test_tr.csv'), index=False)
        test_plays_te.to_csv(os.path.join(fout, 'test_te.csv'), index=False)

    @classmethod
    def tfrecord(cls, train_plays: pd.DataFrame, vad_plays: pd.DataFrame, test_plays: pd.DataFrame,
                 fout: str, seqslen: int, n_shards: int):
        seqslen += 1  # the former 50 as fold-in data, the latter 50 as labels

        def preproc(data):
            data_grouped_by_user = data.groupby('uid')

            outs = list()
            for i, (_, group) in enumerate(data_grouped_by_user):
                seqs_i = group['sid'].values.astype(np.int32)
                seqs_t = group['time'].values.astype(np.float32)
                seqs_month = group['month'].values.astype(np.int32)
                seqs_day = group['day'].values.astype(np.int32)
                seqs_weekday = group['weekday'].values.astype(np.int32)
                seqs_hour = group['hour'].values.astype(np.int32)

                # right-aligned
                currlen = seqs_i.size
                if currlen > seqslen:
                    seqs_i = seqs_i[-seqslen:]
                    seqs_t = seqs_t[-seqslen:]
                    seqs_month = seqs_month[-seqslen:]
                    seqs_day = seqs_day[-seqslen:]
                    seqs_weekday = seqs_weekday[-seqslen:]
                    seqs_hour = seqs_hour[-seqslen:]
                else:
                    seqs_i = np.pad(seqs_i, (seqslen - currlen, 0))
                    seqs_t = np.pad(seqs_t, (seqslen - currlen, 0))
                    seqs_month = np.pad(seqs_month, (seqslen - currlen, 0))
                    seqs_day = np.pad(seqs_day, (seqslen - currlen, 0))
                    seqs_weekday = np.pad(seqs_weekday, (seqslen - currlen, 0))
                    seqs_hour = np.pad(seqs_hour, (seqslen - currlen, 0))
                outs.append(TrainingInstance(seqs_i, seqs_t, seqs_month, seqs_day, seqs_weekday, seqs_hour))

                i += 1
                if i % 5000 == 0:
                    logging.info("%d users sampled" % i)
            return outs

        logging.info("TFRECORD...preproc the training data for next-item predictions")
        train_observations = preproc(train_plays)
        logging.info("TFRECORD...preproc the validation data for next-item predictions")
        valid_observations = preproc(vad_plays)
        logging.info("TFRECORD...preproc the test data for next-item predictions")
        test_observations = preproc(test_plays)

        logging.info("TFRECORD...write the training data for next-item predictions")
        cnt_tr = len(train_observations)
        shards = list(range(0, cnt_tr, cnt_tr // n_shards))
        if shards[-1] != cnt_tr:
            shards.append(cnt_tr)

        for shard_i, (shard_beg, shard_end) in enumerate(zip(shards[:-1], shards[1:])):
            with tf.io.TFRecordWriter(os.path.join(fout, 'train%03d.tfrec' % shard_i)) as writer:
                for instance in train_observations[shard_beg:shard_end]:
                    writer.write(instance.tfexample())

        logging.info("TFRECORD...write the validation data for next-item predictions")
        with tf.io.TFRecordWriter(os.path.join(fout, 'validation.tfrec')) as writer:
            for instance in valid_observations:
                writer.write(instance.tfexample())

        logging.info("TFRECORD...write the test data for next-item predictions")
        with tf.io.TFRecordWriter(os.path.join(fout, 'test.tfrec')) as writer:
            for instance in test_observations:
                writer.write(instance.tfexample())


def args():
    parser = argparse.ArgumentParser(description='Dataset Processor for Learning-to-Rank Task')
    parser.add_argument('--fin', required=True, help="the input triplet dataset")
    parser.add_argument('--fout', required=True, help="the folder of output files")
    parser.add_argument('--n_shards', type=int, help="the numbser of shards for training data")
    parser.add_argument('--n_heldout_users', type=int, help="the number of hold-out users")
    parser.add_argument('--min_uc', type=int, help="the minimum numbser of recorders for each user")
    parser.add_argument('--min_sc', type=int, help="the minimum numbser of recorders for each item")
    parser.add_argument('--seqslen', type=int, help="the maximum sequence length")

    # ---- for BERT, masked sequence
    parser.add_argument('--dupe_factor', type=int,
                        help="Number of times to duplicate the input data (with different masks)")
    parser.add_argument('--max_predictions_per_seq', type=int,
                        help="Maximum number of masked LM predictions per sequence.")
    parser.add_argument('--masked_seqs_prob', type=float,
                        help="Maximum number of masked LM predictions per sequence.")
    parser.set_defaults(min_uc=5, min_sc=100, n_shards=100, seqslen=30, n_heldout_users=20000,
                        dupe_factor=10, max_predictions_per_seq=15, masked_seqs_prob=0.5)
    return parser.parse_args()


def main():
    # df: pd.DataFrame = pd.read_csv(FLAGS.fin, usecols=['uid', 'sid', 'time'])
    #
    # # filtering the input data
    # df = TripletDataset.filter(df, FLAGS.min_uc, FLAGS.min_sc)
    # user_activity = df['uid'].value_counts()
    # deg_i = df['sid'].value_counts()
    # sparsity = 1. * df.shape[0] / (user_activity.shape[0] * deg_i.shape[0])
    # logging.info("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
    #              (df.shape[0], user_activity.shape[0], deg_i.shape[0], sparsity * 100))
    #
    # # spliting training/validation/test users
    # unique_uid = user_activity.index
    # idx_perm = np.random.permutation(unique_uid.size)
    # unique_uid = unique_uid[idx_perm]
    #
    # n_users = unique_uid.size
    # tr_users = unique_uid[:(n_users - FLAGS.n_heldout_users * 2)]
    # vd_users = unique_uid[(n_users - FLAGS.n_heldout_users * 2): (n_users - FLAGS.n_heldout_users)]
    # te_users = unique_uid[(n_users - FLAGS.n_heldout_users):]

    # logging.info("spliting training/validation/test users")
    # train_plays = df.loc[df['uid'].isin(tr_users)]
    # unique_sid = pd.unique(train_plays['sid'])
    #
    # vad_plays = df.loc[df['uid'].isin(vd_users)]
    # vad_plays = vad_plays.loc[vad_plays['sid'].isin(unique_sid)]
    #
    # test_plays = df.loc[df['uid'].isin(te_users)]
    # test_plays = test_plays.loc[test_plays['sid'].isin(unique_sid)]

    # format data in the pandas fashion
    # logging.info("numerizing training/validation/test data")
    # profile2id = dict((pid, i + 1) for (i, pid) in enumerate(unique_uid))
    # show2id = dict((sid, i + 1) for (i, sid) in enumerate(unique_sid))
    # train_plays = TripletDataset.numerize(train_plays, profile2id, show2id)
    # vad_plays = TripletDataset.numerize(vad_plays, profile2id, show2id)
    # test_plays = TripletDataset.numerize(test_plays, profile2id, show2id)

    def rename(tp: pd.DataFrame):
        tp.rename(columns={"use_ID": "uid", "ite_ID": "sid"}, inplace=True)
        # tp['sid'] = tp['sid'] + 1
        tp.sort_values(by=['uid', 'time'], ascending=True, inplace=True)

    train_plays = pd.read_csv(os.path.join(FLAGS.fin, "train.csv"))
    logging.info("TRAIN sid in range [{0}, {1}]".format(train_plays['sid'].min(), train_plays['sid'].max()))
    rename(train_plays)

    vad_tr = pd.read_csv(os.path.join(FLAGS.fin, "validation_tr.csv"))
    vad_te = pd.read_csv(os.path.join(FLAGS.fin, "validation_te.csv"))
    vad_plays = pd.concat([vad_tr, vad_te])
    logging.info("VALID sid in range [{0}, {1}]".format(vad_plays['sid'].min(), vad_plays['sid'].max()))
    rename(vad_plays)

    test_tr = pd.read_csv(os.path.join(FLAGS.fin, "test_tr.csv"))
    test_te = pd.read_csv(os.path.join(FLAGS.fin, "test_te.csv"))
    test_plays = pd.concat([test_tr, test_te])
    logging.info("TEST sid in range [{0}, {1}]".format(test_plays['sid'].min(), test_plays['sid'].max()))
    rename(test_plays)

    logging.info("transforming training/validation/test data")
    # TripletDataset.csv(train_plays, vad_plays, test_plays, FLAGS.fout)
    TripletDataset.tfrecord(train_plays, vad_plays, test_plays, FLAGS.fout, FLAGS.seqslen, FLAGS.n_shards)


if __name__ == "__main__":
    logging.config.fileConfig('./conf/logging.conf')
    np.random.seed(9876)
    tf.random.set_random_seed(9876)

    FLAGS = args()
    logging.info("================")
    logging.info(vars(FLAGS))
    logging.info("================")

    main()
