"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import math

import numpy as np
import torch as th
import torch.nn as nn


def batch_gather(inputs, indices):
    shape = indices.shape
    assert len(shape) == 2, \
        "batch_gather, rank should be greater than two"

    axis = th.arange(shape[0])[:, None]
    return inputs[axis, indices]


def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device is None:
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A @ B


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'elu':
            return nn.ELU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        elif act == 'rrelu':
            return nn.RReLU()
        else:
            raise NotImplementedError
    else:
        return act


class PositionCoding(nn.Module):
    def __init__(self, vocab_size, num_units):
        super(PositionCoding, self).__init__()
        self.pembs = nn.Embedding(vocab_size, num_units)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.pembs.weight)

    def forward(self, inputs: th.Tensor):
        batch_size, seqs_len = inputs.shape[:2]

        indices = th.tile(th.arange(seqs_len).unsqueeze(0), [batch_size, 1])
        indices = indices.to(inputs.device)
        pcoding = self.pembs(indices)
        return pcoding


class TimeSinusoidCoding(nn.Module):
    def __init__(self, num_units):
        super(TimeSinusoidCoding, self).__init__()
        self.num_units = num_units
        scale = np.power(10000, np.arange(0, num_units, 2) * 1. / num_units)
        self.scale = th.from_numpy(scale).float().view(1, 1, num_units // 2)

    def forward(self, inputs: th.Tensor):
        shape_list = inputs.shape
        assert len(shape_list) == 2, "the tensor rank should be 2."

        scale = self.scale.to(inputs.device)
        x = inputs.unsqueeze(-1).float()
        x = x.tile([1, 1, self.num_units // 2]) / scale

        code_even = th.sin(x)
        code_odd = th.cos(x)

        tcoding = th.stack([code_even, code_odd], dim=-1)
        tcoding = tcoding.view([-1, shape_list[1], self.num_units])
        return tcoding


class GLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GLU, self).__init__()
        self.linear = nn.Linear(in_features, 2 * out_features)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    def forward(self, tensor: th.Tensor):
        h = self.linear(tensor)
        gate, gain = h.chunk(2, dim=-1)
        h = th.tanh(gain) * th.sigmoid(gate)
        return h
