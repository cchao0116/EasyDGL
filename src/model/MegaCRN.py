"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging

import torch
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MegaCRNConfig(object):
    def __init__(self, yaml_config):
        # for personalized recommendation
        data_config = yaml_config['data']
        self.num_nodes = data_config.get('num_nodes')
        self.num_features = data_config.get('num_features', 2)
        self.num_timesteps_in = data_config.get('num_timesteps_in', 12)
        self.num_timesteps_out = data_config.get('num_timesteps_out', 12)

        model_config = yaml_config['model']
        self.n_classes = model_config.get('n_classes', 1)
        self.ycov_dim = model_config.get('ycov_dim', 1)
        self.diffusion_steps = model_config.get('diffution_steps', 3)
        self.num_blocks = model_config.get('num_blocks', 2)
        self.rnn_units = model_config.get('rnn_units', 64)
        self.num_units = model_config.get('num_units', 64)

        self.separate_loss_W = model_config.get('separate_loss_W', 0.01)
        self.compact_loss_W = model_config.get('compact_loss_W', 0.01)
        self.cl_decay_steps = model_config.get('cl_decay_steps')

        self.mem_num = model_config.get('mem_num', 20)
        self.mem_dim = model_config.get('mem_dim', 64)
        logging.info(f"======MegaCRN configure======")
        logging.info(f"num_timesteps_in: {self.num_timesteps_in}")
        logging.info(f"num_timesteps_out: {self.num_timesteps_out}")
        logging.info(f"num_features: {self.num_features}")
        logging.info(f"n_classes: {self.n_classes}")
        logging.info(f"num_blocks: {self.num_blocks}")
        logging.info(f"rnn_units: {self.rnn_units}")
        logging.info(f"mem_num: {self.mem_num}")
        logging.info(f"mem_dim: {self.mem_dim}")
        logging.info(f"diffusion_steps: {self.diffusion_steps}")
        logging.info(f"separate_loss_W: {self.separate_loss_W}")
        logging.info(f"compact_loss_W: {self.compact_loss_W}")
        logging.info(f"cl_decay_steps: {self.cl_decay_steps}")


class MegaGraph(nn.Module):
    def __init__(self, num_nodes, mem_num):
        super(MegaGraph, self).__init__()
        self.emb0 = nn.Parameter(th.empty(num_nodes, mem_num))
        self.emb1 = nn.Parameter(th.empty(num_nodes, mem_num))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # normal_ is must-be, impact the performance
        nn.init.xavier_normal_(self.emb0)
        nn.init.xavier_normal_(self.emb1)

    def forward(self, memory):
        ne0 = th.matmul(self.emb0, memory)
        ne1 = th.matmul(self.emb1, memory)
        g0 = F.softmax(F.relu(th.mm(ne0, ne1.T)), dim=-1)
        g1 = F.softmax(F.relu(th.mm(ne1, ne0.T)), dim=-1)
        return [g1, g0]


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(th.FloatTensor(2 * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(th.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [th.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(th.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(th.einsum("nm,bmc->bnc", support, x))
        x_g = th.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = th.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = th.cat((x, state), dim=-1)
        z_r = th.sigmoid(self.gate(input_and_state, supports))
        z, r = th.split(z_r, self.hidden_dim, dim=-1)
        candidate = th.cat((x, z * state), dim=-1)
        hc = th.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return th.zeros(batch_size, self.node_num, self.hidden_dim)


class EncodingLayer(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(EncodingLayer, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        # shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = th.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class DecodingLayer(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(DecodingLayer, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MemoryNetwork(nn.Module):
    def __init__(self, in_feats, mem_num, mem_dim):
        super(MemoryNetwork, self).__init__()
        self.Wq = nn.Parameter(th.empty(in_feats, mem_dim))
        self.memory = nn.Parameter(th.empty(mem_num, mem_dim))
        self.query = self.pos = self.neg = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.memory)
        nn.init.xavier_normal_(self.Wq)

    def forward(self, h: th.Tensor):
        query = th.matmul(h, self.Wq)
        mem = self.memory

        att_score = th.softmax(th.matmul(query, mem.t()), dim=-1)  # alpha: (B, N, M)
        value = th.matmul(att_score, mem)  # (B, N, d)

        _, ind = th.topk(att_score, k=2, dim=-1)
        pos = mem[ind[:, :, 0]]  # B, N, d
        neg = mem[ind[:, :, 1]]  # B, N, d

        if self.training:
            self.query = query
            self.pos = pos
            self.neg = neg
        return value


class MegaCRN(nn.Module):
    # def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
    #              ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True):
    def __init__(self, config: MegaCRNConfig):
        super(MegaCRN, self).__init__()
        num_nodes = config.num_nodes
        num_features = config.num_features
        diffusion_steps = config.diffusion_steps
        num_layers = config.num_blocks
        rnn_units = config.rnn_units
        ycov_dim = config.ycov_dim
        n_classes = config.n_classes

        self.num_nodes = num_nodes
        self.num_features = num_features
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.num_timesteps_in = config.num_timesteps_in
        self.num_timesteps_out = config.num_timesteps_out

        # graph
        mem_num = config.mem_num
        mem_dim = config.mem_dim
        self.graph = MegaGraph(num_nodes, mem_num)

        # encoder
        self.encoder = EncodingLayer(num_nodes, num_features, rnn_units, diffusion_steps, num_layers)

        # memory
        self.mnet = MemoryNetwork(rnn_units, config.mem_num, mem_dim)

        # deocoder
        self.cl = CurriculumLearner(config.cl_decay_steps)
        self.decoder = DecodingLayer(num_nodes, n_classes + ycov_dim,
                                     rnn_units + mem_dim, diffusion_steps,
                                     num_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(rnn_units + mem_dim, n_classes, bias=True))

        # loss definition
        self.alpha = config.separate_loss_W
        self.beta = config.compact_loss_W
        self.separate_loss = nn.TripletMarginLoss(margin=1.0)
        self.compact_loss = nn.MSELoss()

    def forward(self, _, feat):
        x = feat['x'][..., :self.num_features]
        y_cov = feat['y_cov']

        supports = self.graph(self.mnet.memory)
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, _ = self.encoder(x, init_state, supports)  # B, T, N, hidden
        h_t = h_en[:, -1, :, :]  # B, N, hidden (last state)

        h_att = self.mnet(h_t)
        h_t = th.cat([h_t, h_att], dim=-1)

        ht_list = [h_t] * self.num_layers
        go = th.zeros((x.shape[0], self.num_nodes, self.n_classes), device=x.device)
        out = []
        for timestep_out in range(self.num_timesteps_out):
            y_cov_now = y_cov[:, timestep_out]
            h_de, ht_list = self.decoder(th.cat([go, y_cov_now], dim=-1), ht_list, supports)
            go = self.proj(h_de)
            out.append(go)

            if self.training:
                labels = feat['y'][..., :self.n_classes]
                go = self.cl(go, labels[:, timestep_out, ...])
        output = th.stack(out, dim=1)
        return output.squeeze(3)

    def loss(self, y_pred, y_true):
        if self.training:
            self.cl.batches_seen += 1

        mask = (y_true != 0).float()
        mask /= mask.mean()
        loss = th.abs(y_pred - y_true)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        loss0 = loss.mean()

        query = self.mnet.query
        pos = self.mnet.pos.detach()
        neg = self.mnet.neg.detach()
        loss1 = self.separate_loss(query, pos, neg)
        loss2 = self.compact_loss(query, pos)

        loss = loss0 + self.alpha * loss1 + self.beta * loss2
        return loss


class CurriculumLearner(nn.Module):
    def __init__(self, cl_decay_steps):
        super(CurriculumLearner, self).__init__()
        self.batches_seen = 0.
        self.cl_decay_steps = cl_decay_steps

    def forward(self, y_pred, y_true):
        if self.training:
            sampling_threshold = self.cl_decay_steps / (
                    self.cl_decay_steps + np.exp(self.batches_seen / self.cl_decay_steps))

            c = np.random.uniform(0, 1)
            if c < sampling_threshold:
                return y_true

        return y_pred


class MegaCRNv2(nn.Module):
    # def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
    #              ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True):
    def __init__(self, config: MegaCRNConfig):
        super(MegaCRNv2, self).__init__()
        self.num_nodes = config.num_nodes
        self.input_dim = config.n_classes
        self.rnn_units = config.rnn_units
        self.output_dim = config.n_classes
        self.horizon = config.num_timesteps_in
        self.num_layers = config.num_blocks
        self.cheb_k = config.diffusion_steps
        self.ycov_dim = config.n_classes
        self.cl_decay_steps = config.cl_decay_steps
        self.use_curriculum_learning = True

        self.batches_seen = 0
        self.cl = CurriculumLearner(config.cl_decay_steps)
        self.query = self.pos = self.neg = None

        # memory
        self.mem_num = config.mem_num
        self.mem_dim = config.mem_dim
        self.memory = self.construct_memory()

        # encoder
        self.encoder = EncodingLayer(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)

        # deocoder
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = DecodingLayer(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k,
                                     self.num_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

        self.alpha = config.separate_loss_W
        self.beta = config.compact_loss_W
        self.separate_loss = nn.TripletMarginLoss(margin=1.0)
        self.compact_loss = nn.MSELoss()

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim),
                                         requires_grad=True)  # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]]  # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]]  # B, N, d
        return value, query, pos, neg

    def forward(self, _, feat):
        x = feat['x'][..., :self.input_dim]
        y_cov = feat['y_cov']

        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports)  # B, T, N, hidden
        h_t = h_en[:, -1, :, :]  # B, N, hidden (last state)

        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)

        ht_list = [h_t] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                # labels = feat['y'][..., :self.input_dim]
                # c = np.random.uniform(0, 1)
                # if c < self.compute_sampling_threshold(self.batches_seen):
                #     go = labels[:, t, ...]

                labels = feat['y'][..., :self.input_dim]
                go = self.cl(go, labels[:, t, ...])
        output = torch.stack(out, dim=1)

        if self.training:
            self.query = query
            self.pos = pos
            self.neg = neg
        return output.squeeze(3)

    def loss(self, y_pred, y_true):
        if self.training:
            self.cl.batches_seen += 1

        mask = (y_true != 0).float()
        mask /= mask.mean()
        loss = th.abs(y_pred - y_true)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        loss0 = loss.mean()

        self.batches_seen += 1
        query = self.query
        pos = self.pos.detach()
        neg = self.neg.detach()
        loss1 = self.separate_loss(query, pos, neg)
        loss2 = self.compact_loss(query, pos)

        loss = loss0 + self.alpha * loss1 + self.beta * loss2
        return loss
