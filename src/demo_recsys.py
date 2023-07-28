"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import argparse
import functools
import logging.config
import os
import random

import numpy as np
import torch as th
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.recsys import Reader, NetflixDataset, collate_masklast, collate_maskrandom
from model.EasyDGL import Recommender, EasyDGLConfig
from util import EarlyStoppingV1


def args():
    parser = argparse.ArgumentParser(description='EasyDGL Benchmark')
    parser.add_argument('--config', default='conf/model/Netflix/EasyDGL.yaml', type=str,
                        help='Config file.')
    parser.add_argument('--device', type=int, default=0,
                        help='running device. E.g `--device 0`, if using cpu, set `--device -1`')
    return parser.parse_args()


def evaluate(net, dataloader):
    measures = {"HR@50": list(), "NDCG@50": list(), "NDCG@100": list()}

    net.eval()
    with th.no_grad():
        gain = 1. / th.log2(th.arange(2., 2. + 100).to(device))  # [N]
        idcg = np.cumsum(gain.cpu().numpy())
        for feat, label in dataloader:
            feat = {k: v.to(device) for k, v in feat.items()}
            label = label.to(device)
            probs = net(feat).softmax(dim=2)
            probs = probs.squeeze(1)

            # filter items in the training
            ix = th.arange(probs.shape[0])[:, None]
            training_items = feat['x']
            probs[ix, training_items] = 0.

            # compute the results
            _, pred = th.topk(probs, 100, dim=1)

            # zero-values used for masking
            truth_lookup = th.zeros_like(probs)
            truth_lookup[ix, label] = 1.
            Tp = truth_lookup[ix, pred] * th.sign(pred)
            hr50 = th.sum(Tp[:, :50], dim=1).cpu().numpy()

            P = th.sum(label.sign(), dim=1).cpu().numpy()
            dcg50 = th.sum(Tp[:, :50] * gain[:50], dim=1).cpu().numpy()
            idcg50 = idcg[np.minimum(P, 50) - 1]
            dcg100 = th.sum(Tp * gain, dim=1).cpu().numpy()
            idcg100 = idcg[np.minimum(P, 100) - 1]

            measures["HR@50"].append(hr50)
            measures["NDCG@50"].append(dcg50 / idcg50)
            measures["NDCG@100"].append(dcg100 / idcg100)
    return {k: np.concatenate(v).mean() for k, v in measures.items()}


def run():
    data_config = config['data']
    reader = Reader(data_config['fpath'], data_config['ftrain'],
                    data_config['fval_tr'], data_config['fval_te'],
                    data_config['ftest_tr'], data_config['ftest_te'],
                    config['seqs_len'])
    train_data = NetflixDataset(reader.train_data)
    valid_data = NetflixDataset(reader.valid_data)
    test_data = NetflixDataset(reader.test_data)

    batch_size = data_config.get('batch_size')
    mask_rate = data_config.get('mask_rate')
    mask_len = data_config.get('mask_len')
    mask_const = data_config.get('mask_const')
    collate_fn = functools.partial(collate_maskrandom, mask_rate=mask_rate,
                                   mask_len=mask_len, mask_const=mask_const)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=4,
                                  shuffle=True, collate_fn=collate_fn)
    collate_fn = functools.partial(collate_masklast, mask_const=mask_const)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_fn)
    logging.info("======data configure======")
    logging.info(f"seqs_len: {config['seqs_len']}")
    logging.info(f"batch_size: {batch_size}")
    logging.info(f"mask_rate: {mask_rate}")
    logging.info(f"mask_rate: {mask_rate}")
    logging.info(f"mask_len: {mask_len}")

    training_config = config['train']
    patience = training_config.get('patience')
    epochs = training_config.get('epochs')
    lr = training_config.get('lr')
    lr_decay_ratio = training_config.get('lr_decay_ratio')
    weight_decay = training_config.get('weight_decay')
    grad_clip = training_config.get('max_grad_norm')
    test_every_n_epochs = training_config.get('test_every_n_epochs')
    logging.info("======train configure======")
    logging.info(f"learning rate: {lr}")
    logging.info(f"learning decay ratio: {lr_decay_ratio}")
    logging.info(f"l2 weight decay: {weight_decay}")
    logging.info(f"epochs: {epochs}")
    logging.info(f"test_every_n_epochs: {test_every_n_epochs}")

    net = Recommender(EasyDGLConfig(config)).to(device)
    optim = th.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
    scheduler = th.optim.lr_scheduler.ExponentialLR(optim, gamma=lr_decay_ratio)

    stopper = EarlyStoppingV1(patience)
    for epoch in range(epochs + 1):
        running_loss = list()

        # training stage
        net.train()
        for feat, label in train_dataloader:
            feat = {k: v.to(device) for k, v in feat.items()}
            label = label.to(device)

            # Forward-inference
            logits = net.forward(feat)
            loss = net.loss(logits, label)
            running_loss.append(loss.item())

            # Back-propogation
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optim.step()

        scheduler.step()  # update the learning rate
        logging.info("{0:3d}, loss={1:5f}, lr={2:5f}".format(
            epoch, np.mean(running_loss), scheduler.get_last_lr()[0]))

        # evaluation stage
        if epoch % test_every_n_epochs == 0:
            valid_res = evaluate(net, valid_dataloader)
            logging.info(f"H@50:{valid_res['HR@50']:.5f}, "
                         f"N@50:{valid_res['NDCG@50']:.5f}, "
                         f"N@100:{valid_res['NDCG@100']:.5f}")
            test_res = evaluate(net, test_dataloader)
            logging.info(f"H@50:{test_res['HR@50']:.5f}, "
                         f"N@50:{test_res['NDCG@50']:.5f}, "
                         f"N@100:{test_res['NDCG@100']:.5f}")
            stopper.step(running_loss, valid_res['NDCG@100'].mean(), valid_res, test_res)
            if stopper.early_stop:
                break
    stopper.summary()


if __name__ == "__main__":
    logging.config.fileConfig('./conf/logging.conf')

    SEED = 9876
    np.random.seed(SEED)
    random.seed(SEED)
    th.manual_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    flags = args()
    config = yaml.load(open(flags.config, 'r'), yaml.Loader)
    device = th.device(flags.device) if flags.device >= 0 else th.device('cpu')

    run()
