"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import torch as th
import torch.nn as nn


class RandomMask(nn.Module):
    def __init__(self, rate: float):
        super(RandomMask, self).__init__()
        self.rate = rate

    def forward(self, tensor: th.Tensor):
        if not self.training:
            return None

        # computation in cpu device
        num_timesteps_out = tensor.shape[1]
        # tensor: batch_size, num_timesteps_out, num_nodes, num_features
        # masked_ix: batch_size, num_timesteps_out, num_nodes
        masked_ix = th.ones_like(tensor[..., 0]) * self.rate
        # masked_ix: num_nodes, num_timesteps_out * batch_size
        th.bernoulli(masked_ix, out=masked_ix)
        masked_ix = th.cat(masked_ix.chunk(num_timesteps_out, dim=1), dim=0)
        masked_ix = masked_ix.permute(2, 0, 1)
        return masked_ix


class CorrelationAdjustedMask(nn.Module):
    def __init__(self, rate: float, sep: int, n):
        super(CorrelationAdjustedMask, self).__init__()
        self.rate = rate
        self.sep = sep
        self.n = n

    def forward(self, tensor: th.Tensor):
        if not self.training:
            return None

        # computation in cpu device
        # tensor: batch_size, num_timesteps_out, num_nodes, num_features
        # => batch_size, num_timesteps_out, num_nodes
        tensor = tensor[..., 0]
        num_timesteps_out = tensor.shape[1]
        # tensor: batch_size, num_timesteps_out, num_nodes, num_features
        # masked_ix: batch_size, num_timesteps_out, num_nodes
        masked_ix = th.ones_like(tensor) * self.rate
        th.bernoulli(masked_ix, out=masked_ix)
        tensor_busket = th.clip(th.ceil(tensor / self.sep), min=0., max=self.n)
        th.multiply(masked_ix, tensor_busket, out=masked_ix)
        # mask_buskets: num_nodes, num_timesteps_out * batch_size, 1
        mask_buskets = th.cat(masked_ix.chunk(num_timesteps_out, dim=1), dim=0)
        mask_buskets = mask_buskets.permute(2, 0, 1)
        return mask_buskets
