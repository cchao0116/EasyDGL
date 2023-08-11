"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging

import dgl
import dgl.function as dglfn
import dgl.ops as F
import numpy as np
import torch as th
import torch.nn as nn

from model.GraphRNN import DCGRUCell


class DCRNNConfig(object):
    def __init__(self, yaml_config):
        # for personalized recommendation
        data_config = yaml_config['data']
        self.num_nodes = data_config.get('num_nodes')
        self.num_features = data_config.get('num_features', 2)
        self.num_timesteps_in = data_config.get('num_timesteps_in', 12)
        self.num_timesteps_out = data_config.get('num_timesteps_out', 12)

        model_config = yaml_config['model']
        self.n_classes = model_config.get('n_classes', 1)
        self.diffusion_steps = model_config.get('diffution_steps', 2)
        self.num_blocks = model_config.get('num_blocks', 2)
        self.num_units = model_config.get('num_units', 64)
        self.cl_decay_steps = model_config.get('cl_decay_steps')

        logging.info(f"======DCRNN configure======")
        logging.info(f"num_timesteps_in: {self.num_timesteps_in}")
        logging.info(f"num_timesteps_out: {self.num_timesteps_out}")
        logging.info(f"num_features: {self.num_features}")
        logging.info(f"n_classes: {self.n_classes}")
        logging.info(f"num_blocks: {self.num_blocks}")
        logging.info(f"num_units: {self.num_units}")
        logging.info(f"diffusion_steps: {self.diffusion_steps}")
        logging.info(f"cl_decay_steps: {self.cl_decay_steps}")


def dual_random_walk_udf(graph):
    reverse = dgl.reverse(graph, copy_ndata=True, copy_edata=True)
    g = dgl.batch([graph, reverse])
    deg = F.copy_e_sum(g, g.edata['ef'])
    g.srcdata['D_invsqrt'] = 1 / th.clip(deg, min=1.)
    g.apply_edges(dglfn.u_mul_e('D_invsqrt', 'ef', 'nef'))
    g.edata.pop('ef')
    g.edata['ef'] = g.edata.pop('nef')
    return g


class DiffusionConv(nn.Module):
    def __init__(self, in_feats, out_feats, diffusion_steps):
        super(DiffusionConv, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.linear = nn.Linear((2 * diffusion_steps + 1) * in_feats, out_feats)

    def forward(self, graph: dgl.DGLGraph, feat):
        layer_medium = [feat]

        if 'ef' not in graph.edata:
            raise RuntimeError("Graph does not have ef edge-feature.")

        with graph.local_scope():
            # feat: num_nodes, batch_size, num_units
            x0 = th.cat([feat, feat], dim=0)
            graph.ndata['h'] = x0
            graph.update_all(dglfn.u_mul_e('h', 'ef', 'm'), dglfn.sum('m', 'neigh'))
            x1 = graph.ndata.pop('neigh')
            layer_medium.extend(x1.chunk(2, dim=0))

            for _ in range(2, self.diffusion_steps + 1):
                graph.ndata['h'] = x1
                graph.update_all(dglfn.u_mul_e('h', 'ef', 'm'), dglfn.sum('m', 'neigh'))
                x2 = 2 * graph.ndata.pop('neigh') - x0
                layer_medium.extend(x2.chunk(2, dim=0))
                x1, x0 = x2, x1

        layer_outs = th.cat(layer_medium, dim=2)
        layer_outs = self.linear(layer_outs)
        return layer_outs


class EncodingLayer(nn.Module):
    def __init__(self, in_feats, out_feats, diffusion_steps, num_layers):
        super(EncodingLayer, self).__init__()

        self.grucells = nn.ModuleList()
        zrgate = DiffusionConv(in_feats + out_feats, 2 * out_feats, diffusion_steps)
        hgate = DiffusionConv(in_feats + out_feats, out_feats, diffusion_steps)
        self.grucells.append(DCGRUCell(zrgate, hgate))
        for _ in range(1, num_layers):
            zrgate = DiffusionConv(2 * out_feats, 2 * out_feats, diffusion_steps)
            hgate = DiffusionConv(2 * out_feats, out_feats, diffusion_steps)
            self.grucells.append(DCGRUCell(zrgate, hgate))

    def forward(self, graph, feat, multilayer_states):
        # feat: batch_size, num_nodes, num_units
        # state [list]: num_layers x [ batch_size, num_nodes, num_units ]
        new_state = list()
        prev_feat = feat
        for i, cell in enumerate(self.grucells):
            prev_feat = cell(graph, prev_feat, multilayer_states[i])
            new_state.append(prev_feat)
        return prev_feat, new_state


class DecodingLayer(nn.Module):
    def __init__(self, in_feats, out_feats, diffusion_steps, num_layers):
        super(DecodingLayer, self).__init__()
        self.linear = nn.Linear(out_feats, in_feats)

        self.grucells = nn.ModuleList()
        zrgate = DiffusionConv(in_feats + out_feats, 2 * out_feats, diffusion_steps)
        hgate = DiffusionConv(in_feats + out_feats, out_feats, diffusion_steps)
        self.grucells.append(DCGRUCell(zrgate, hgate))
        for _ in range(1, num_layers):
            zrgate = DiffusionConv(2 * out_feats, 2 * out_feats, diffusion_steps)
            hgate = DiffusionConv(2 * out_feats, out_feats, diffusion_steps)
            self.grucells.append(DCGRUCell(zrgate, hgate))

    def forward(self, graph, feat, multilayer_states):
        # feat: batch_size, num_nodes, num_units
        # state [list]: num_layers x [ batch_size, num_nodes, num_units ]
        new_state = list()
        prev_feat = feat
        for i, cell in enumerate(self.grucells):
            # print(f"{i}: {prev_feat.shape}, {multilayer_states[i].shape}")
            prev_feat = cell(graph, prev_feat, multilayer_states[i])
            new_state.append(prev_feat)

        output = self.linear(prev_feat)
        return output, new_state


class DCRNN(nn.Module):
    def __init__(self, config: DCRNNConfig):
        super(DCRNN, self).__init__()
        num_features = config.num_features
        num_units = config.num_units
        diffusion_steps = config.diffusion_steps
        num_layers = config.num_blocks
        n_classes = config.n_classes

        self.n_classes = n_classes
        self.num_units = num_units
        self.num_nodes = config.num_nodes
        self.num_timesteps_in = config.num_timesteps_in
        self.num_timesteps_out = config.num_timesteps_out
        self.cl_decay_steps = config.cl_decay_steps

        self.cache = dict()
        self.batches_seen = 0
        self.encoder = EncodingLayer(num_features, num_units, diffusion_steps, num_layers)
        self.decoder = DecodingLayer(n_classes, num_units, diffusion_steps, num_layers)

    def curriculum_learning(self, timestep_out, prev_layer_out, feat):
        if self.training:
            sampling_threshold = self.cl_decay_steps / (
                    self.cl_decay_steps + np.exp(self.batches_seen / self.cl_decay_steps))
            self.batches_seen += 1

            if np.random.uniform(0, 1) < sampling_threshold:
                # nfeat: num_nodes, batch_size, num_features
                decoder_input = feat['y'][:, timestep_out].permute(1, 0, 2)
                # => num_nodes, batch_size, n_classes
                decoder_input = decoder_input[..., :self.n_classes]
                return decoder_input

        return prev_layer_out

    def check_before(self, graph: dgl.DGLGraph):
        if 'g' not in self.cache:
            self.cache['g'] = dual_random_walk_udf(graph)

    def forward(self, graph, feat):
        self.check_before(graph)
        graph = self.cache['g']

        # encoding procedure
        # nfeat: batch_size, num_timesteps_in, num_nodes, num_features
        nfeat: th.Tensor = feat['x']
        batch_size = nfeat.shape[0]

        # nfeat: num_timesteps_in, num_nodes, batch_size, num_features
        nfeat = nfeat.permute(1, 2, 0, 3)
        prev_state = th.zeros(self.num_nodes, batch_size, self.num_units).to(nfeat.device)
        prev_state = [prev_state] * self.num_timesteps_in
        for timestep_in in range(self.num_timesteps_in):
            _, prev_state = self.encoder(graph, nfeat[timestep_in], prev_state)

        # decoding procedure
        decoder_out = list()
        layer_in = th.zeros_like(nfeat[0])[..., :self.n_classes]
        for timestep_out in range(self.num_timesteps_out):
            # layer_out: batch_size, num_nodes
            layer_out, prev_state = self.decoder(graph, layer_in, prev_state)
            decoder_out.append(layer_out)
            layer_in = self.curriculum_learning(timestep_out, layer_out, feat)
        # out: num_nodes, num_timesteps_out, batch_size
        out = th.stack(decoder_out, dim=1)[..., 0]
        # => batch_size, num_timesteps_out, num_nodes
        out = out.permute(2, 1, 0)
        return out

    @staticmethod
    def loss(y_pred, y_true):
        mask = (y_true != 0).float()
        mask /= mask.mean()
        loss = th.abs(y_pred - y_true)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return loss.mean()
