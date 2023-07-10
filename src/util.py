"""
@version: 2.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging

import numpy as np


class EarlyStoppingV1(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
        self.best_valid_res = dict()
        self.best_test_res = dict()

    def step(self, loss, acc, val_res: dict, test_res: dict):
        self.update(val_res, test_res)

        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
        elif acc <= self.best_acc:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = min(loss, self.best_loss)
            self.best_acc = max(acc, self.best_acc)
            self.counter = 0
        return self.early_stop

    def update(self, valid_res: dict, test_res: dict):
        if not bool(self.best_valid_res):
            self.best_valid_res = valid_res
            self.best_test_res = test_res
            return

        for key, best_val in self.best_valid_res.items():
            if valid_res[key] >= best_val:
                self.best_valid_res[key] = valid_res[key]
                self.best_test_res[key] = test_res[key]

    def summary(self):
        logging.info(f"SUMMARY -- "
                     f"H@50: {self.best_test_res['HR@50']:.5f} "
                     f"N@50: {self.best_test_res['NDCG@50']:.5f} "
                     f"N@100: {self.best_test_res['NDCG@100']:.5f}")


class EarlyStoppingV2(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
        self.best_valid_res = dict()
        self.best_test_res = dict()

    def step(self, loss, acc, val_res: dict, test_res: dict):
        self.update(val_res, test_res)

        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
        elif acc <= self.best_acc:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = min(loss, self.best_loss)
            self.best_acc = max(acc, self.best_acc)
            self.counter = 0
        return self.early_stop

    def update(self, valid_res: dict, test_res: dict):
        if not bool(self.best_valid_res):
            self.best_valid_res = valid_res
            self.best_test_res = test_res
            return

        for key, best_val in self.best_valid_res.items():
            if valid_res[key][0] >= best_val[0]:
                self.best_valid_res[key] = valid_res[key]
                self.best_test_res[key] = test_res[key]

    def summary(self):
        res = list()
        for maF1, miF1, Ac in zip(
                self.best_test_res['maF1'], self.best_test_res['miF1'], self.best_test_res['Ac']):
            res.append(np.asarray([maF1, miF1, Ac]))
        with np.printoptions(formatter={'float_kind': "{:.5f}".format}):
            logging.info(
                "==REPO===T0:{0}, T1:{1}, T2:{2}=====".format(res[0], res[1], res[2]))


class EarlyStoppingV3(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
        self.best_valid_res = dict()
        self.best_test_res = dict()

    def step(self, loss, acc, val_res: dict, test_res: dict):
        self.update(val_res, test_res)

        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
        elif acc >= self.best_acc:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = min(loss, self.best_loss)
            self.best_acc = min(acc, self.best_acc)
            self.counter = 0
        return self.early_stop

    def update(self, valid_res: dict, test_res: dict):
        if not bool(self.best_valid_res):
            self.best_valid_res = valid_res
            self.best_test_res = test_res
            return

        for key, best_val in self.best_valid_res.items():
            entry_to_update = valid_res[key] <= best_val
            self.best_valid_res[key][entry_to_update] = valid_res[key][entry_to_update]
            self.best_test_res[key][entry_to_update] = test_res[key][entry_to_update]

    def summary(self):
        res = list()
        for mae, rmse, mape in zip(
                self.best_test_res['mae'], self.best_test_res['rmse'], self.best_test_res['mape']):
            res.append(np.asarray([mae, rmse, mape]))
        with np.printoptions(formatter={'float_kind': "{:.5f}".format}):
            logging.info(
                "==REPO===15m:{0}, 30m:{1}, 45m:{2}=====".format(res[0], res[1], res[2]))
