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

from model.GraphRNN import DCGRUCell


class AGCRNConfig(object):
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
        logging.info(f"======AGCRN configure======")
        logging.info(f"num_timesteps_in: {self.num_timesteps_in}")
        logging.info(f"num_timesteps_out: {self.num_timesteps_out}")
        logging.info(f"num_features: {self.num_features}")
        logging.info(f"n_classes: {self.n_classes}")
        logging.info(f"num_blocks: {self.num_blocks}")
        logging.info(f"num_units: {self.num_units}")
        logging.info(f"diffusion_steps: {self.diffusion_steps}")


class AVWGCN(nn.Module):
    def __init__(self, in_feats, out_feats, diffusion_steps, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = diffusion_steps
        self.weights_pool = nn.Parameter(th.empty(embed_dim, diffusion_steps, in_feats, out_feats))
        self.bias_pool = nn.Parameter(th.empty(embed_dim, out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights_pool, a=math.sqrt(5))
        nn.init.zeros_(self.bias_pool)

    def forward(self, node_embeddings, feat):
        # node_embeddings: num_nodes, num_units
        # feat: batch_size, num_nodes, num_units
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(th.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [th.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(th.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = th.stack(support_set, dim=0)
        weights = th.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = th.matmul(node_embeddings, self.bias_pool)  # N, dim_out

        # x_g: batch_size, num_nodes, num_units
        x_g = th.einsum("knm,bmc->bknc", supports, feat)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = th.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class EncodingLayer(nn.Module):
    def __init__(self, in_feats, out_feats, diffusion_steps, num_layers):
        super(EncodingLayer, self).__init__()

        self.grucells = nn.ModuleList()
        zrgate = AVWGCN(in_feats + out_feats, 2 * out_feats, diffusion_steps, out_feats)
        hgate = AVWGCN(in_feats + out_feats, out_feats, diffusion_steps, out_feats)
        self.grucells.append(DCGRUCell(zrgate, hgate))
        for _ in range(1, num_layers):
            zrgate = AVWGCN(2 * out_feats, 2 * out_feats, diffusion_steps, out_feats)
            hgate = AVWGCN(2 * out_feats, out_feats, diffusion_steps, out_feats)
            self.grucells.append(DCGRUCell(zrgate, hgate))

    def forward(self, node_embeddings, feat, multilayer_states):
        # node_embeddings: num_nodes, num_units
        # feat: batch_size, num_nodes, num_units
        # multilayer_states [list]: num_layers x [ batch_size, num_nodes, num_units ]
        new_state = list()
        prev_feat = feat
        for i, cell in enumerate(self.grucells):
            prev_feat = cell(node_embeddings, prev_feat, multilayer_states[i])
            # print(i, prev_feat[0, :, 0])
            new_state.append(prev_feat)
        return prev_feat, new_state


class DecodingLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_timesteps_out):
        super(DecodingLayer, self).__init__()
        self.conv = nn.Conv2d(1, num_timesteps_out * out_feats,
                              kernel_size=(1, in_feats), bias=True)

    def forward(self, feat):
        output = self.conv(feat)
        return output


class AGCRN(nn.Module):
    def __init__(self, config: AGCRNConfig):
        super(AGCRN, self).__init__()
        num_nodes = config.num_nodes
        num_features = config.num_features
        num_units = config.num_units
        diffusion_steps = config.diffusion_steps
        num_layers = config.num_blocks
        n_classes = config.n_classes
        num_timesteps_out = config.num_timesteps_out

        self.num_nodes = num_nodes
        self.num_units = num_units
        self.num_timesteps_in = config.num_timesteps_in

        self.node_embeddings = nn.Parameter(th.empty(num_nodes, num_units))
        self.encoder = EncodingLayer(num_features, num_units, diffusion_steps, num_layers)
        self.decoder = DecodingLayer(num_units, n_classes, num_timesteps_out)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.node_embeddings, a=math.sqrt(5))

    def forward(self, _, feat):
        # encoding procedure
        # nfeat: batch_size, num_timesteps_in, num_nodes, num_features
        nfeat: th.Tensor = feat['x']
        batch_size = nfeat.shape[0]

        # nfeat: num_timesteps_in, batch_size, num_nodes, num_features
        nfeat = nfeat.permute(1, 0, 2, 3)
        prev_state = th.zeros(batch_size, self.num_nodes, self.num_units).to(nfeat.device)
        prev_state = [prev_state] * self.num_timesteps_in

        layer_state = None
        for timestep_in in range(self.num_timesteps_in):
            # nfeat[timestep_in]: batch_size, num_nodes, num_features / num_units
            # prev_state: batch_size, num_nodes, num_units
            layer_state, prev_state = self.encoder(self.node_embeddings, nfeat[timestep_in], prev_state)

        # decoding procedure
        # prev_state, batch_size, 1, num_nodes, num_units
        assert layer_state is not None
        layer_state = layer_state.unsqueeze(1)
        out = self.decoder(layer_state)

        # out: batch_size, num_timesteps_out, num_nodes
        out = out.squeeze(-1)
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
