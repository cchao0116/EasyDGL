"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import math
import logging
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MTGNNConfig(object):
    def __init__(self, yaml_config):
        # for personalized recommendation
        data_config = yaml_config['data']
        self.num_nodes = data_config.get('num_nodes')
        self.num_features = data_config.get('num_features', 2)
        self.num_timesteps_in = data_config.get('num_timesteps_in', 12)
        self.num_timesteps_out = data_config.get('num_timesteps_out', 12)

        model_config = yaml_config['model']
        self.n_classes = model_config.get('n_classes', 1)
        self.num_units = model_config.get('num_units', 40)
        self.diffusion_steps = model_config.get('diffution_steps', 2)
        self.num_blocks = model_config.get('num_blocks', 3)

        self.feat_drop = model_config.get('feat_drop', 0.)
        self.msg_drop = model_config.get('msg_drop', 0.)

        self.dilation_exponential = model_config.get('dilation_exponential', 1)
        self.residual_channels = model_config.get('residual_channels', 32)
        self.conv_channels = model_config.get('conv_channels', 32)
        self.skip_channels = model_config.get('skip_channels', 64)
        self.out_channels = model_config.get('out_channels', 128)

        self.neigh_topk = model_config.get('neigh_topk', None)
        self.propalpha = model_config.get('propalpha', 0.05)
        self.tanhalpha = model_config.get('tanhalpha', 3)

        logging.info(f"======MTGNN configure======")
        logging.info(f"num_timesteps_in: {self.num_timesteps_in}")
        logging.info(f"num_timesteps_out: {self.num_timesteps_out}")
        logging.info(f"num_features: {self.num_features}")
        logging.info(f"n_classes: {self.n_classes}")
        logging.info(f"num_blocks: {self.num_blocks}")
        logging.info(f"residual_channels: {self.residual_channels}")
        logging.info(f"conv_channels: {self.conv_channels}")
        logging.info(f"skip_channels: {self.skip_channels}")
        logging.info(f"out_channels: {self.out_channels}")
        logging.info(f"diffusion_steps: {self.diffusion_steps}")


class MTGraph(nn.Module):
    def __init__(self, num_nodes, num_units, alpha):
        super(MTGraph, self).__init__()
        self.num_nodes = num_nodes
        self.alpha = alpha

        self.emb0 = nn.Parameter(th.empty(num_nodes, num_units))
        self.emb1 = nn.Parameter(th.empty(num_nodes, num_units))
        self.linear0 = nn.Linear(num_units, num_units)
        self.linear1 = nn.Linear(num_units, num_units)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # normal_ is must-be, impact the performance
        nn.init.normal_(self.emb0)
        nn.init.normal_(self.emb1)
        nn.init.kaiming_uniform_(self.linear0.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear1.weight, a=math.sqrt(5))

    def forward(self, k=None):
        nodevec1 = th.tanh(self.alpha * self.linear0(self.emb0))
        nodevec2 = th.tanh(self.alpha * self.linear1(self.emb1))
        adj = th.mm(nodevec1, nodevec2.transpose(1, 0)) - th.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(th.tanh(self.alpha * adj))

        if k is not None:
            mask = th.zeros(self.num_nodes, self.num_nodes).to(adj.device)
            s1, t1 = adj.topk(k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask
        return adj


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()

        kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(kernel_set))

        for kern in kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, feat):
        # feat: batch_size, cin, num_nodes, num_timesteps_in
        xs = [conv(feat) for conv in self.tconv]

        out_shape = xs[-1].size(3)
        xs = [x[..., -out_shape:] for x in xs]
        xs = th.cat(xs, dim=1)
        return xs


class GatedCNN(nn.Module):
    def __init__(self, cin, cout, dilation_factor):
        super(GatedCNN, self).__init__()
        self.gainCNN = DilatedInception(cin, cout, dilation_factor)
        self.gateCNN = DilatedInception(cin, cout, dilation_factor)

    def forward(self, feat):
        gain = th.tanh(self.gainCNN(feat))
        gate = th.sigmoid(self.gateCNN(feat))
        out = gain * gate
        return out


class MhPConv(nn.Module):
    def __init__(self, in_feats, out_feats, diffusion_steps, alpha):
        super(MhPConv, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.alpha = alpha
        self.fc = nn.Conv2d(in_feats * (diffusion_steps + 1), out_feats, kernel_size=(1, 1))

    def forward(self, adj, feat):
        adj = adj + th.eye(adj.size(0)).to(feat.device)
        adj = adj / adj.sum(1).view(-1, 1)

        h = feat
        out = [h]
        for i in range(self.diffusion_steps):
            # h: batch_size, cin, num_nodes, adjusted_num_timesteps_in (after tconv)
            # adj: num_nodes, num_nodes
            h_msg = th.einsum('ncvl,vw->ncwl', h, adj)
            h = self.alpha * feat + (1 - self.alpha) * h_msg
            out.append(h)
        ho = th.cat(out, dim=1)
        ho = self.fc(ho)
        return ho


class CurriculumLearner(nn.Module):
    def __init__(self, step, max_task_level):
        super(CurriculumLearner, self).__init__()
        self.step = step
        self.max_task_level = max_task_level
        self.step_trained = self.task_level = 1

    def forward(self, y_pred, y_true):
        if self.step_trained % self.step == 0 and self.task_level < self.max_task_level:
            self.task_level += 1

        self.step_trained += 1
        return y_pred[:, :self.task_level], y_true[:, :self.task_level]


class MTGNN(nn.Module):

    def __init__(self, config: MTGNNConfig):
        super(MTGNN, self).__init__()
        num_nodes = config.num_nodes
        num_features = config.num_features
        num_units = config.num_units
        diffusion_steps = config.diffusion_steps
        num_layers = config.num_blocks
        n_classes = config.n_classes
        num_timesteps_in = config.num_timesteps_in

        cl_steps = 2500
        self.cl = CurriculumLearner(cl_steps, num_timesteps_in)

        residual_channels = config.residual_channels
        conv_channels = config.conv_channels
        skip_channels = config.skip_channels

        # input convolution layer
        self.feat_drop = nn.Dropout(config.feat_drop)
        self.msg_drop = nn.Dropout(config.msg_drop)
        self.feat_conv = nn.Conv2d(num_features, residual_channels, (1, 1))

        # Encoding Module
        self.num_layers = num_layers
        kernel_size = 7  # 7 days in one week
        dilation_exponential = config.dilation_exponential
        receptive_field = num_layers * (kernel_size - 1) + 1
        if dilation_exponential > 1:
            receptive_field = 1 + (kernel_size - 1) * (
                    dilation_exponential ** num_layers - 1) / (
                                      dilation_exponential - 1)
        self.padding_size = max(0, receptive_field - num_timesteps_in)

        # temporal convolution neural networks
        self.temproal_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        for j in range(1, num_layers + 1):
            dilation_factor = dilation_exponential ** (j - 1)
            self.temproal_convs.append(
                GatedCNN(residual_channels, conv_channels, dilation_factor))
            self.residual_convs.append(
                nn.Conv2d(in_channels=conv_channels,
                          out_channels=residual_channels,
                          kernel_size=(1, 1)))

        # skip connections
        self.layernorms = nn.ModuleList()
        max_kernel_size = max(receptive_field, num_timesteps_in)
        self.skip_beg = nn.Conv2d(num_features, skip_channels, (1, max_kernel_size))
        self.skip_convs = nn.ModuleList()
        receptive_field_base = 1
        for j in range(1, num_layers + 1):
            receptive_field_j = receptive_field_base + j * (kernel_size - 1)
            if dilation_exponential > 1:
                receptive_field_j = receptive_field_base + (kernel_size - 1) * (
                        dilation_exponential ** j - 1) / (
                                            dilation_exponential - 1)
            self.skip_convs.append(
                nn.Conv2d(conv_channels, skip_channels,
                          (1, max_kernel_size - receptive_field_j + 1)))

            normalized_shape = [residual_channels, num_nodes, max_kernel_size - receptive_field_j + 1]
            self.layernorms.append(nn.LayerNorm(normalized_shape))
        out_kernel_size = max(0, num_timesteps_in - receptive_field) + 1
        self.skip_end = nn.Conv2d(residual_channels, skip_channels, (1, out_kernel_size))

        # Graph convolution neural networks
        tanhalpha = config.tanhalpha
        self.neigh_topk = config.neigh_topk
        self.graph = MTGraph(num_nodes, num_units, tanhalpha)

        propalpha = config.propalpha
        self.origin_gcns = nn.ModuleList()
        self.reversed_gcns = nn.ModuleList()
        for j in range(1, num_layers + 1):
            self.origin_gcns.append(
                MhPConv(conv_channels, residual_channels,
                        diffusion_steps, propalpha))
            self.reversed_gcns.append(
                MhPConv(conv_channels, residual_channels,
                        diffusion_steps, propalpha))

        # Decoding Module
        out_channels = config.out_channels
        self.out_convs = nn.ModuleList()
        self.out_convs.append(nn.Conv2d(skip_channels, out_channels, (1, 1)))
        self.out_convs.append(nn.Conv2d(out_channels, n_classes, (1, 1)))

    def forward(self, _, feat):
        # feat: batch_size, num_timesteps_in, num_nodes, num_features
        x = feat['x']
        # => batch_size, num_features, num_nodes, num_timesteps_in
        x = x.permute(0, 3, 2, 1)
        if self.padding_size > 0:
            x = nn.functional.pad(x, (self.padding_size, 0, 0, 0))

        # graph: num_nodes, num_nodes
        graph: th.Tensor = self.graph(self.neigh_topk)

        skip = self.skip_beg(self.feat_drop(x))
        x = self.feat_conv(x)
        for tconv, layernorm, ogcn, rgcn, rconv, sconv in zip(
                self.temproal_convs, self.layernorms, self.origin_gcns,
                self.reversed_gcns, self.residual_convs, self.skip_convs):
            temporal_h = self.msg_drop(tconv(x))
            spatial_h = ogcn(graph, temporal_h) + rgcn(graph.transpose(1, 0), temporal_h)
            out = spatial_h + x[:, :, :, -spatial_h.size(3):]
            x = layernorm(out)

            skip = sconv(temporal_h) + skip
        skip = self.skip_end(x) + skip

        out = F.relu(skip)
        for out_conv in self.out_convs[:-1]:
            out = out_conv(out)
            out = th.relu(out)
        # out: batch_size, num_timesteps_out, num_nodes, 1
        out = self.out_convs[-1](out)
        # => batch_size, num_timesteps_out, num_nodes
        out = out.squeeze(3)
        return out

    def loss(self, y_pred, y_true):
        y_pred, y_true = self.cl(y_pred, y_true)
        mask = (y_true != 0).float()
        mask /= mask.mean()
        loss = th.abs(y_pred - y_true)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return loss.mean()
