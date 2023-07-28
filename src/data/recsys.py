"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging
import os.path

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset


def collate_maskrandom(list_tensors, mask_rate, mask_len, mask_const):
    timestamps = np.vstack([item[1] for item in list_tensors]).astype(np.float)
    tokens = np.vstack([item[0] for item in list_tensors]).astype(np.int32)
    tokens_len = np.sum(np.sign(tokens), axis=1, keepdims=False).astype(np.int32)
    tokens_n_mask = np.minimum(tokens_len * mask_rate, mask_len).astype(np.int32)

    masked_tokens = tokens
    masked_positions = np.zeros_like(tokens)[:, :mask_len]
    masked_labels = np.zeros_like(masked_positions)
    for i, (tl, ml) in enumerate(zip(tokens_len, tokens_n_mask)):
        masked_indices = np.random.choice(tl, ml, replace=False)
        masked_positions[i, :ml] = masked_indices
        masked_labels[i, :ml] = tokens[i, masked_indices]
        masked_tokens[i, masked_indices] = mask_const

    timestamps = th.from_numpy(timestamps).float()
    masked_tokens = th.from_numpy(masked_tokens).long()
    masked_positions = th.from_numpy(masked_positions).long()
    masked_labels = th.from_numpy(masked_labels).long()
    decoded_tensors = {'t': timestamps, 'x': masked_tokens, 'p': masked_positions, 'y': masked_labels}
    return decoded_tensors, masked_labels


def collate_masklast(list_tensors, mask_const):
    timestamps = np.vstack([item[1] for item in list_tensors]).astype(np.float)
    tokens = np.vstack([item[0] for item in list_tensors]).astype(np.int32)

    masked_positions = np.sum(np.sign(tokens), axis=1, keepdims=True) - 1
    masked_tokens = tokens
    xaxis = np.arange(tokens.shape[0])[:, np.newaxis]
    masked_labels = masked_tokens[xaxis, masked_positions]
    masked_tokens[xaxis, masked_positions] = mask_const

    timestamps = th.from_numpy(timestamps).float()
    masked_tokens = th.from_numpy(masked_tokens).long()
    masked_positions = th.from_numpy(masked_positions).long()
    masked_labels = th.from_numpy(masked_labels).long()
    decoded_tensors = {'t': timestamps, 'x': masked_tokens, 'p': masked_positions, 'y': masked_labels}
    return decoded_tensors, masked_labels


class NetflixDataset(Dataset):

    def __init__(self, data: dict):
        self.user_sequences = data

    def __len__(self):
        return self.user_sequences['x'].shape[0]

    def __getitem__(self, index):
        if th.is_tensor(index):
            index = index.tolist()

        seqs_i = self.user_sequences['x'][index]
        seqs_t = self.user_sequences['t'][index]
        return seqs_i, seqs_t


class Reader:
    def __init__(self, fpath: str, ftrain: str,
                 fval_tr: str, fval_te: str,
                 ftest_tr: str, ftest_te: str,
                 seqs_len: int = 30):
        self.seqs_len = seqs_len

        fout_train: str = os.path.join(fpath, f"train.npz")
        fout_valid: str = os.path.join(fpath, f"valid.npz")
        fout_test: str = os.path.join(fpath, f"test.npz")
        if os.path.exists(fout_train):
            logging.info(f"load training file: {fout_train}")
            logging.info(f"load validation file: {fout_valid}")
            logging.info(f"load test file: {fout_test}")
            self.load(fout_train, fout_valid, fout_test)
        else:
            logging.info(f"load training file: {ftrain}")
            logging.info(f"load validation file: {fval_tr}, {fval_te}")
            logging.info(f"load test file: {ftest_tr}, {ftest_te}")
            self.train_data, self.valid_data, self.test_data, self.num_items = self.build(
                ftrain, fval_tr, fval_te, ftest_tr, ftest_te)
            logging.info(f"save training file: {fout_train}")
            logging.info(f"save validation file: {fout_valid}")
            logging.info(f"save test file: {fout_test}")
            self.save(fout_train, fout_valid, fout_test)

    def build(self, ftrain, fval_tr, fval_te, ftest_tr, ftest_te):
        train_df: pd.DataFrame = pd.read_csv(ftrain, usecols=['uid', 'sid', 'time'])
        valid_tr: pd.DataFrame = pd.read_csv(fval_tr, usecols=['uid', 'sid', 'time'])
        valid_te: pd.DataFrame = pd.read_csv(fval_te, usecols=['uid', 'sid', 'time'])
        test_tr: pd.DataFrame = pd.read_csv(ftest_tr, usecols=['uid', 'sid', 'time'])
        test_te: pd.DataFrame = pd.read_csv(ftest_te, usecols=['uid', 'sid', 'time'])
        valid_df = pd.concat([valid_tr, valid_te])
        test_df = pd.concat([test_tr, test_te])
        train_df.drop_duplicates(inplace=True)
        valid_df.drop_duplicates(inplace=True)
        test_df.drop_duplicates(inplace=True)

        train_df.sort_values(['uid', 'time'], inplace=True)
        valid_df.sort_values(['uid', 'time'], inplace=True)
        test_df.sort_values(['uid', 'time'], inplace=True)

        if train_df['sid'].min() == 0:
            raise RuntimeError('The item indices begin from zero, however zero is used for masking.')

        num_items = train_df['sid'].max()
        num_train_users = train_df['uid'].unique().size
        num_valid_users = valid_df['uid'].unique().size
        num_test_users = test_df['uid'].unique().size
        print("Netflix statistics: ")
        print(f"training users: {num_train_users}")
        print(f"validation users: {num_valid_users}")
        print(f"test users: {num_test_users}")

        def filter_by_seqslen(df: pd.DataFrame, out):
            data_grouped_by_user = df.groupby('uid')
            for i, (_, group) in enumerate(data_grouped_by_user):
                si = group['sid'].values.astype(np.int32)
                st = group['time'].values.astype(np.float32)
                sl = min(si.size, self.seqs_len)
                out['x'][i, :sl] = si[-sl:]
                out['t'][i, :sl] = st[-sl:]

        train_data = {
            'x': np.zeros([num_train_users, self.seqs_len], dtype=np.int32),
            't': np.zeros([num_train_users, self.seqs_len], dtype=np.float)}
        valid_data = {
            'x': np.zeros([num_valid_users, self.seqs_len], dtype=np.int32),
            't': np.zeros([num_valid_users, self.seqs_len], dtype=np.float)}
        test_data = {
            'x': np.zeros([num_test_users, self.seqs_len], dtype=np.int32),
            't': np.zeros([num_test_users, self.seqs_len], dtype=np.float)}
        filter_by_seqslen(train_df, train_data)
        filter_by_seqslen(valid_df, valid_data)
        filter_by_seqslen(test_df, test_data)

        return train_data, valid_data, test_data, num_items

    def save(self, fout_train, fout_valid, fout_test):
        np.savez(fout_train, x=self.train_data['x'], t=self.train_data['t'], num_items=self.num_items)
        np.savez(fout_valid, x=self.valid_data['x'], t=self.valid_data['t'], num_items=self.num_items)
        np.savez(fout_test, x=self.test_data['x'], t=self.test_data['t'], num_items=self.num_items)

    def load(self, fout_train, fout_valid, fout_test):
        def decompress(npz_files, return_n=False):
            if return_n:
                return {'x': npz_files['x'], 't': npz_files['t']}, npz_files['num_items']
            else:
                return {'x': npz_files['x'], 't': npz_files['t']}

        self.train_data, self.num_items = decompress(np.load(fout_train), True)
        self.valid_data = decompress(np.load(fout_valid))
        self.test_data = decompress(np.load(fout_test))
