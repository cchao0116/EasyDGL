"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging
import os.path
import pickle

import dgl
import numpy as np
import torch as th
from sklearn.cluster import SpectralClustering
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


def collate_mask(list_tensors, mask_fn):
    xs = [i[0] for i in list_tensors]
    ts = [i[1] for i in list_tensors]
    ys = [i[2] for i in list_tensors]

    xs = np.concatenate(xs, axis=0)
    ts = np.concatenate(ts, axis=0)
    ys = np.concatenate(ys, axis=0)

    xs = th.from_numpy(xs).float()
    ts = th.from_numpy(ts).float()
    labels = th.from_numpy(ys).float()
    ps = mask_fn(labels)

    decoded_tensors = {'x': xs, 't': ts, 'p': ps}
    return decoded_tensors, labels


def collate_fn(list_tensors):
    xs = [i[0] for i in list_tensors]
    ts = [i[1] for i in list_tensors]
    ys = [i[2] for i in list_tensors]

    xs = np.concatenate(xs, axis=0)
    ts = np.concatenate(ts, axis=0)
    ys = np.concatenate(ys, axis=0)

    decoded_tensors = {'x': th.from_numpy(xs).float(),
                       't': th.from_numpy(ts).float()}
    labels = th.from_numpy(ys).float()
    return decoded_tensors, labels


class METRDataset(Dataset):
    def __init__(self, data: dict):
        # test.npz, (6850, 12, 207, 2)
        self.x = data['x'].astype(np.float)
        self.t = data['t'].astype(np.float)
        self.y = data['y'].astype(np.float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index) -> T_co:
        x = self.x[index:index + 1]
        t = self.t[index:index + 1]
        y = self.y[index:index + 1]
        return x, t, y


class Reader(object):
    def __init__(self, fpath: str):
        fout_train: str = os.path.join(fpath, "train.npz")
        fout_valid: str = os.path.join(fpath, "valid.npz")
        fout_test: str = os.path.join(fpath, "test.npz")
        fout_adj: str = os.path.join(fpath, "adj_mx.pkl")
        fout_congest: str = os.path.join(fpath, "congest.npz")

        if not os.path.exists(fout_train):
            raise RuntimeError(f"{fout_train} not existed.")

        logging.info(f"load training file: {fout_train}")
        logging.info(f"load validation file: {fout_valid}")
        logging.info(f"load test file: {fout_test}")
        self.train_data = self.load_npz(fout_train)
        self.valid_data = self.load_npz(fout_valid)
        self.test_data = self.load_npz(fout_test)
        logging.info(f"load adjacent file: {fout_adj}")
        self.g, self.adj = self.load_graph(fout_adj)

        if not os.path.exists(fout_congest):
            mark_lookup = self.build()
            fout_mark: str = os.path.join(fpath, "mark.npy")
            self.save(fout_congest, fout_mark, mark_lookup)
        else:
            logging.info(f"load congestion file: {fout_congest}")
            self.load(fout_congest)

    def build(self, k=5):
        # for example, in test.npz, the shape is (6850, 12, 207)
        all_train_data = self.train_data['x'][::12]
        all_train_data = np.reshape(all_train_data, (-1, 207, 2))
        # mean: (207), std: (207)
        mean, std = np.mean(all_train_data, axis=0), np.std(all_train_data, axis=0)
        print(mean.shape, std.shape)

        def event(input_data: np.ndarray, cutoff_time=12):
            # for example, in test.npz, the shape is (6850, 12, 207, 2)
            # => (6850, 12, 207)
            xtrain_mean, xtrain_std = mean[:, 0], std[:, 0]
            input_data = input_data[:, :, :, 0]
            mask_events = np.logical_and(input_data < (xtrain_mean - xtrain_std), input_data != 0.)
            mask_events = mask_events.astype(np.float)[:, ::-1]

            # for example,
            #   input: [0, 1, 1, 0, 0, ...]
            #   -> [..., 0, 0, 1, 1, 0]
            #   -> [..., 0, 0, 1, 2, 2]
            #   -> [..., 0, 0, 1, 1, 1]
            #   -> 3
            mask_events = np.sign(np.cumsum(mask_events, axis=1))
            mask_events = np.sum(mask_events, axis=1, keepdims=True)

            shape = input_data.shape
            event_time = np.arange(shape[1]).reshape((1, shape[1], 1))
            event_time = np.tile(event_time.astype(np.float), (shape[0], 1, shape[2]))
            event_time = event_time + (shape[1] - mask_events)
            event_time = np.maximum(event_time, cutoff_time)
            return event_time

        logging.info("compute timestamps for congestion events")
        train_event = event(self.train_data['x'])
        valid_event = event(self.valid_data['x'])
        test_event = event(self.test_data['x'])

        self.train_data['t'] = train_event.astype(np.float)
        self.train_data['mean'] = mean
        self.train_data['std'] = std
        self.valid_data['t'] = valid_event.astype(np.float)
        self.test_data['t'] = test_event.astype(np.float)

        logging.info("compute marks for congestion events")
        clustering = SpectralClustering(n_clusters=k)
        clustering.fit(self.adj)
        mark_indices = clustering.labels_ + 1
        mark_lookup = np.zeros((207, k + 1), dtype=np.float)
        mark_lookup[np.arange(207), mark_indices] = 1.
        return mark_lookup

    @staticmethod
    def load_npz(fname):
        npzfiles = np.load(fname)
        return {'x': npzfiles['x'].astype(np.float),
                'y': npzfiles['y'].astype(np.float),
                'x_offsets': npzfiles['x_offsets'].astype(np.int),
                'y_offsets': npzfiles['y_offsets'].astype(np.int)}

    @staticmethod
    def load_graph(fname):
        with open(fname, 'rb') as f:
            _, _, adj = pickle.load(f)
            adj = adj.astype(np.float)
            num_nodes = adj.shape[0]

            src, dst = np.nonzero(adj)
            graph: dgl.DGLGraph = dgl.graph(
                (src, dst), num_nodes=num_nodes)
            # shape: num_edge, 1
            graph.edata['ef'] = th.FloatTensor(adj[(src, dst)]).unsqueeze(1)
        return graph, adj

    def save(self, fout_congest, fout_mark, mark_lookup):
        logging.info(f"save congestion events: {fout_congest}")
        np.savez(fout_congest, mean=self.train_data['mean'],
                 std=self.train_data['std'], train_event=self.train_data['t'],
                 valid_event=self.valid_data['t'], test_event=self.test_data['t'])
        logging.info(f"save event marks: {fout_mark}")

        with open(fout_mark, 'wb') as f:
            pickle.dump(mark_lookup, f)

    def load(self, fout_congest):
        npzfiles = np.load(fout_congest)
        self.train_data['t'] = npzfiles['train_event']
        self.train_data['mean'] = npzfiles['mean']
        self.train_data['std'] = npzfiles['std']
        self.valid_data['t'] = npzfiles['valid_event']
        self.test_data['t'] = npzfiles['test_event']
