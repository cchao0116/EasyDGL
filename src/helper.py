"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import torch as th


class TrafficForecasting:
    @classmethod
    def build(cls, config):
        model = config['m']
        if model in ['EasyDGL']:
            from model import mEasyDGL, EasyDGLConfig
            net = mEasyDGL(EasyDGLConfig(config))
        elif model in ['DCRNN']:
            from model import DCRNN, DCRNNConfig
            net = DCRNN(DCRNNConfig(config))
        else:
            raise RuntimeError('{model} is not matched.')

        training_config = config['train']
        lr = training_config.get('lr')
        eps = training_config.get('eps', 1e-8)
        lr_decay_ratio = training_config.get('lr_decay_ratio', 1.)
        weight_decay = training_config.get('weight_decay', 0.)
        optimizer = training_config['optimizer']
        if optimizer in ['adam']:
            optim = th.optim.Adam(net.parameters(), lr, weight_decay=weight_decay, eps=eps)
        else:
            raise RuntimeError('{optimizer} is not matched.')

        if model in ['DCRNN']:
            steps = training_config['steps']
            scheduler = th.optim.lr_scheduler.MultiStepLR(optim, milestones=steps, gamma=lr_decay_ratio)
        else:
            scheduler = th.optim.lr_scheduler.ExponentialLR(optim, gamma=lr_decay_ratio)

        return net, optim, scheduler
