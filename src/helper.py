"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import torch as th


class TrafficForecasting:
    @classmethod
    def build(cls, config, reader):
        model = config['m']
        if model in ['EasyDGL']:
            from model import mEasyDGL, EasyDGLConfig
            net = mEasyDGL(EasyDGLConfig(config))
        elif model in ['DCRNN']:
            from model import DCRNN, DCRNNConfig
            net = DCRNN(DCRNNConfig(config))
        elif model in ['AGCRN']:
            from model.AGCRN import AGCRN, AGCRNConfig
            net = AGCRN(AGCRNConfig(config))
        elif model in ['MTGNN']:
            from model.MTGNN import MTGNN, MTGNNConfig
            net = MTGNN(MTGNNConfig(config))
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

        if model in ['DCRNN', 'AGCRN']:
            steps = training_config['steps']
            scheduler = th.optim.lr_scheduler.MultiStepLR(optim, milestones=steps, gamma=lr_decay_ratio)
        else:
            scheduler = th.optim.lr_scheduler.ExponentialLR(optim, gamma=lr_decay_ratio)

        if model in ['MTGNN']:
            mean = reader.train_data['x'][..., 0].mean()
            std = reader.train_data['x'][..., 0].std()
            scaler = StandardScaler(mean, std)
        else:
            mean = reader.train_data['x'].mean([0, 1])
            mean = th.from_numpy(mean).float()
            std = reader.train_data['x'].std([0, 1])
            std = th.from_numpy(std).float()
            scaler = StandardScalerV1(mean, std)
        return net, optim, scheduler, scaler


class StandardScalerV1:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        # data: batch_size, num_timesteps_in, num_nodes, num_features
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        # data: batch_size, num_timesteps_out, num_nodes
        return (data * std[:, 0]) + mean[:, 0]


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        # data: batch_size, num_timesteps_in, num_nodes, num_features
        data[..., 0] = (data[..., 0] - self.mean) / self.std
        return data

    def inverse_transform(self, data):
        # data: batch_size, num_timesteps_out, num_nodes
        return (data * self.std) + self.mean
