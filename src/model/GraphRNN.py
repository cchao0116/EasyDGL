"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import torch as th
import torch.nn as nn


class DCGRUCell(nn.Module):
    def __init__(self, zrgate, cgate):
        super(DCGRUCell, self).__init__()
        self.zrgate = zrgate
        self.cgate = cgate

    def forward(self, graph, feat, state):
        # feat:  batch_size, num_nodes, num_units
        # state: batch_size, num_nodes, num_units
        feat_and_state = th.cat([feat, state], dim=-1)
        zr = th.sigmoid(self.zrgate(graph, feat_and_state))
        z, r = zr.chunk(2, dim=-1)

        feat_and_rstate = th.cat([feat, r * state], dim=-1)
        c = th.tanh(self.cgate(graph, feat_and_rstate))

        new_state = z * state + (1 - z) * c
        return new_state
