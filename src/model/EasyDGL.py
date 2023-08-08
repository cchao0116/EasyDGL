"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging
import math
import pickle

import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn

from model.util import PositionCoding, TimeSinusoidCoding, batch_gather, GLU


class EasyDGLConfig(object):
    def __init__(self, yaml_config):
        # for personalized recommendation
        data_config = yaml_config['data']
        self.seqs_len = yaml_config.get('seqs_len', 31)
        self.mask_const = data_config.get('mask_const', -1)
        self.mask_len = data_config.get('mask_len', 6)
        self.mask_rate = data_config.get('mask_rate', 0.)
        self.mask_max = data_config.get('mask_max', -1)
        self.time_scalor = data_config.get('time_scalor', 1)

        # for traffic flow forecasting
        self.num_nodes = data_config.get('num_nodes')
        self.num_features = data_config.get('num_features', 2)
        self.num_timesteps_in = data_config.get('num_timesteps_in', 12)
        self.num_timesteps_out = data_config.get('num_timesteps_out', 12)

        model_config = yaml_config['model']
        self.n_classes = model_config.get('n_classes')
        self.num_blocks = model_config.get('num_blocks')
        self.num_units = model_config.get('num_units')
        self.num_heads = model_config.get('num_heads')
        self.feat_drop = model_config.get('feat_drop', 0.)
        self.msg_drop = model_config.get('msg_drop', 0.)
        self.att_drop = model_config.get('att_drop', 0.)

        self.mark_lookup = pickle.load(open(model_config['fmark'], 'rb'))
        self.num_marks = self.mark_lookup.shape[1]

        logging.info(f"======EasyDGL configure======")
        logging.info(f"time_scalor: {self.time_scalor}")
        logging.info(f"n_classes: {self.n_classes}")
        logging.info(f"num_blocks: {self.num_blocks}")
        logging.info(f"num_units: {self.num_units}")
        logging.info(f"num_heads: {self.num_heads}")
        logging.info(f"msg_drop: {self.msg_drop}")
        logging.info(f"att_drop: {self.att_drop}")
        logging.info(f"num_marks: {self.num_marks}")


class AIAConv(nn.Module):
    def __init__(self, in_feats, out_feats,
                 num_heads, num_marks, att_drop=0.):
        super(AIAConv, self).__init__()
        self.num_units = out_feats
        self.num_heads = num_heads
        self.num_marks = num_marks

        # Query, Key, Value, Timespan transformation
        self.fc_q = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_k = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_v = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_t = nn.Linear(in_feats, out_feats, bias=False)
        self.att_drop = nn.Dropout(att_drop)

        # Event transformation
        head_feats = out_feats // num_heads
        self.fc_i = nn.Linear(head_feats + 1, head_feats * self.num_marks)
        self.weight_i = nn.Parameter(th.empty((num_marks, head_feats)))
        self.scale_i = nn.Parameter(th.empty(num_marks))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stddev = 0.02
        nn.init.normal_(self.fc_q.weight, 0., stddev)
        nn.init.normal_(self.fc_k.weight, 0., stddev)
        nn.init.normal_(self.fc_v.weight, 0., stddev)
        nn.init.normal_(self.fc_t.weight, 0., stddev)
        nn.init.normal_(self.fc_i.weight, 0., stddev)
        nn.init.kaiming_uniform_(self.weight_i, a=math.sqrt(5))
        nn.init.zeros_(self.scale_i)

    def intensities(self, events, timespans):
        batch_size = events.shape[0] // self.num_heads

        # Compute intermediate representations
        timespans = th.tile(timespans.unsqueeze(-1), [self.num_heads, 1, 1])
        mark_units = th.cat([events, timespans], dim=2)
        mark_units = self.fc_i(mark_units)
        mark_units = th.sigmoid(mark_units)  # ( h*N, T_q, C/h*E)
        mark_units = th.cat(mark_units.chunk(self.num_marks, dim=2), dim=0)  # ( E*h*N, T_q, C/h)

        # Perform mark-wise projection and intensity
        weight = self.weight_i.tile([1, self.num_heads]).unsqueeze(0).tile([batch_size, 1, 1])  # ( N, E, C )
        weight = th.cat(weight.chunk(self.num_heads, dim=2), dim=0)  # ( h*N, E, C/h )
        weight = th.cat(weight.chunk(self.num_marks, dim=1), dim=0)  # ( E*h*N, 1, C/h )
        weight = weight.permute(0, 2, 1)

        scale = th.exp(self.scale_i)
        scale = scale.unsqueeze(0).tile([batch_size, 1])  # ( N, E )
        scale = scale.tile([self.num_heads, 1])  # ( h*N, E )
        scale = th.cat(scale.chunk(self.num_marks, dim=1), dim=0)  # ( E*h*N, 1)

        all_mark_inty = th.matmul(mark_units, weight).squeeze(2) / scale  # ( h*N*E, T_q)
        all_mark_inty = scale * th.log(1. + th.exp(all_mark_inty))
        all_mark_inty = th.stack(all_mark_inty.chunk(self.num_marks, dim=0), dim=2)  # ( h*N, T_q, E)
        return all_mark_inty

    def forward(self, queries, keys, timespans, attention_masks, event_marks):
        # Linear transform
        Q = self.fc_q(queries)  # (N, T_q, C)
        K = self.fc_k(keys)  # (N, T_k, C)
        V = self.fc_v(keys)  # (N, T_k, C)
        T = self.fc_t(keys)  # (N, T_k, C)

        # Split and concat
        Q_ = th.cat(th.chunk(Q, self.num_heads, dim=2), dim=0)
        K_ = th.cat(th.chunk(K, self.num_heads, dim=2), dim=0)
        V_ = th.cat(th.chunk(V, self.num_heads, dim=2), dim=0)
        T_ = th.cat(th.chunk(T, self.num_heads, dim=2), dim=0)

        # Multiplication
        outs = th.matmul(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outs = outs / math.sqrt(self.num_units)

        # Key Masking
        paddings = th.ones_like(outs) * (-2 ** 32 + 1)
        outs = th.where(attention_masks == 0., paddings, outs)  # (h*N, T_q, T_k)

        # Activation
        outs = th.softmax(outs, dim=2)  # (h*N, T_q, T_k)

        # Weighted sum and dropout for events
        E_ = th.matmul(outs, T_)  # ( h*N, T_q, C/h)

        # TPPs intensity
        all_mark_inty = self.intensities(E_, timespans)

        # Use intensity as the weight of each key
        mark_inty = th.unsqueeze(all_mark_inty, 2)  # (h*N, T_q, 1, E)
        # Hereby the intensitiesfor MASK tokens are zero-out.
        mark_inty = th.sum(mark_inty * event_marks, dim=-1)  # (h*N, T_q, T_k)
        # Be very carefully, below intensities for MASK tokens are set to ones.
        # mark_inty_ones = th.eye(mark_inty.shape[1]).to(mark_inty.device)
        # mark_inty = mark_inty * (1 - mark_inty_ones) + mark_inty_ones

        # Weighted sum and dropout
        outs = outs * mark_inty  # (h*N, T_q, T_k)
        outs = th.matmul(self.att_drop(outs), V_)

        # Residual connection
        outs = th.cat(outs.chunk(self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        outs += queries[..., :self.num_units]
        return outs, all_mark_inty


class CTSMATransformer(nn.Module):

    def __init__(self, in_feats, out_feats,
                 num_heads, num_marks,
                 msg_drop=0., att_drop=0.):
        super(CTSMATransformer, self).__init__()
        self.num_units = out_feats
        self.conv = AIAConv(in_feats, out_feats,
                            num_heads, num_marks, att_drop)
        self.fc_before = nn.Linear(out_feats, out_feats)
        self.drop_before = nn.Dropout(msg_drop)
        self.ln_before = nn.LayerNorm(out_feats)

        self.fc_intermediate = nn.Linear(out_feats, 2 * out_feats)
        self.msg_act = nn.GELU()

        self.fc_after = nn.Linear(2 * out_feats, out_feats)
        self.drop_after = nn.Dropout(msg_drop)
        self.ln_after = nn.LayerNorm(out_feats)

    def forward(self, nfeat, tfeat, attention_masks, event_marks):
        # Standard Attention
        layer_inputs = nfeat
        attention_outs, all_mark_inty = self.conv(
            layer_inputs, layer_inputs, tfeat,
            attention_masks, event_marks)

        # Run a linear projection of `hidden_size` then add a residual
        attention_outs = self.fc_before(attention_outs)
        attention_outs = self.drop_before(attention_outs) + layer_inputs[..., :self.num_units]
        attention_outs = self.ln_before(attention_outs)

        # The activation is only applied to the "intermediate" hidden layer.
        intermediate_outs = self.fc_intermediate(attention_outs)
        intermediate_outs = self.msg_act(intermediate_outs)

        # Down-project back to `hidden_size` then add the residual.
        layer_outs = self.fc_after(intermediate_outs)
        layer_outs = self.drop_after(layer_outs) + attention_outs
        outs = self.ln_after(layer_outs)
        return outs, all_mark_inty


class Recommender(nn.Module):
    """
        for personalized recommendation
    """

    def __init__(self, config: EasyDGLConfig):
        super(Recommender, self).__init__()
        self.mask_const = config.mask_const
        self.mask_len = config.mask_len
        self.seqs_len = config.seqs_len
        self.num_units = config.num_units
        self.num_heads = config.num_heads
        self.time_scalor = config.time_scalor

        self.mark_lookup = th.from_numpy(config.mark_lookup)
        self.num_marks = config.num_marks

        # Embedding, plus one for MASK
        self.pcoding = PositionCoding(config.seqs_len, config.num_units)
        self.tcoding = TimeSinusoidCoding(config.num_units)
        self.ecoding = nn.Embedding(
            self.num_marks, config.num_units, padding_idx=0)
        self.embedding = nn.Embedding(
            config.n_classes + 1, config.num_units, padding_idx=0)
        self.reset_parameters()

        # CTSMATransformer
        self.input_drop = nn.Dropout(config.msg_drop)
        self.layers = nn.ModuleList()
        self.layers.append(CTSMATransformer(
            3 * config.num_units, config.num_units, config.num_heads,
            self.num_marks, config.msg_drop, config.att_drop))
        for i in range(1, config.num_blocks):
            self.layers.append(CTSMATransformer(
                config.num_units, config.num_units, config.num_heads,
                self.num_marks, config.msg_drop, config.att_drop))

        # Estimator Projection
        self.msg_fc = nn.Linear(config.num_units, config.num_units)
        self.msg_act = nn.GELU()
        self.msg_ln = nn.LayerNorm(config.num_units)
        self.bias = nn.Parameter(th.zeros(config.n_classes + 1))

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.ecoding.weight)
        with th.no_grad():
            self.embedding.weight[0].fill_(0)
            self.ecoding.weight[0].fill_(0)

    def forward(self, feat):
        nfeat: th.Tensor = feat['x']
        tfeat: th.Tensor = feat['t'] / self.time_scalor

        timespans = th.clip(tfeat[:, 1:] - tfeat[:, :-1], min=0., max=100.)
        timespans = th.concat([timespans[:, :1], timespans], dim=1)

        # Event marking
        # mask_const -> 0, mark for MASK is UNKNOWN
        event_shape = (*nfeat.shape, self.num_marks)
        event_marks = th.where(
            nfeat == self.mask_const, 0, nfeat).view(-1)
        event_marks = th.index_select(
            self.mark_lookup.to(nfeat.device), 0, event_marks)
        event_marks_id = event_marks.view(event_shape).float()
        event_marks = event_marks_id.unsqueeze(1).tile([
            self.num_heads, self.seqs_len, 1, 1])

        # Attention masking
        attention_masks = th.sign(nfeat).unsqueeze(1).tile(
            [self.num_heads, self.seqs_len, 1])  # ( h*N, Tq, Tk )

        # Embedding and Encoding
        queries = self.embedding(nfeat) * math.sqrt(self.num_units)
        ecodings = th.matmul(event_marks_id, self.ecoding.weight)
        tcodings = self.tcoding(tfeat)
        pcodings = self.pcoding(queries)
        queries = th.concat([queries + tcodings, pcodings, ecodings], dim=2)
        queries = self.input_drop(queries)

        # Run transformers
        layer_in = queries
        for transformer in self.layers:
            layer_in, inty = transformer(layer_in, timespans, attention_masks, event_marks)

        # Predictions only for masked positions
        layer_outs = self.msg_fc(layer_in)
        layer_outs = self.msg_act(layer_outs)
        layer_outs = self.msg_ln(layer_outs)

        masked_positions = feat['p']
        layer_outs = batch_gather(layer_outs, masked_positions)

        # Logits not including MASK
        logits = th.matmul(layer_outs, self.embedding.weight.transpose(1, 0))
        logits = logits + self.bias
        return logits

    def loss(self, logits, label):
        probs = logits.softmax(dim=2)
        probs = th.cat(th.chunk(probs, self.mask_len, dim=1), dim=0)
        probs = probs.squeeze(1)
        log_probs = th.log(probs + 1e-5)

        label = th.cat(th.chunk(label, self.mask_len, dim=1), dim=0).squeeze(1)
        xaxis = th.arange(label.shape[0])
        log_probs = log_probs[xaxis, label]

        nll = -th.sum(log_probs * th.sign(label))
        cnt = th.sum(th.sign(label)) + 1e-5
        loss = nll / cnt
        return loss


class DGLAIAConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads,
                 num_marks, feat_drop=0., att_drop=0., residual=True):
        super(DGLAIAConv, self).__init__()
        self.num_units = out_feats
        self.num_heads = num_heads
        self.num_marks = num_marks
        self.residual = residual

        # Query, Key, Value, Timespan transformation
        self.fc_q = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_k = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_v = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_t = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_e = nn.Linear(1, out_feats, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.att_drop = nn.Dropout(att_drop)

        # Event transformation
        head_feats = out_feats // num_heads
        self.fc_i = nn.Linear(head_feats + 1, head_feats * self.num_marks)
        self.weight_i = nn.Parameter(th.empty((num_marks, head_feats)))
        self.scale_i = nn.Parameter(th.empty(num_marks))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stddev = 0.02
        nn.init.normal_(self.fc_q.weight, 0., stddev)
        nn.init.normal_(self.fc_k.weight, 0., stddev)
        nn.init.normal_(self.fc_v.weight, 0., stddev)
        nn.init.normal_(self.fc_t.weight, 0., stddev)
        nn.init.normal_(self.fc_e.weight, 0., stddev)
        nn.init.normal_(self.fc_i.weight, 0., stddev)
        nn.init.kaiming_uniform_(self.weight_i, a=math.sqrt(5))
        nn.init.zeros_(self.scale_i)

    @staticmethod
    def msg_fn_prod(edges):
        el = edges.src['fk']
        er = edges.dst['fq']

        # optional: consider the distant between sensors
        ef = edges.data['ef']

        # shape: num_edges, batch_size * num_heads, 1
        e = th.sum((el + ef) * er, dim=2, keepdim=True)
        e = e / (el.shape[-1] ** 0.5)
        return {'e': e}

    def msg_fn_inty(self, edges):
        # fi: num_edges, num_heads * batch_size, num_marks
        # fm: num_edges, 1, num_marks
        # => num_edges, num_heads * batch_size, 1
        ei = th.sum(edges.src['fi'] * edges.dst['fm'], dim=2, keepdim=True)
        # ea: num_edges, num_heads * batch_size, 1
        ea = edges.data['ea']
        # ee: num_edges, num_heads * batch_size, 1
        ee = ei * ea
        ee = self.att_drop(ee)
        return {'ee': ee}

    def intensities(self, feat, timespan):
        num_nodes = feat.shape[0]

        # Compute intermediate representations
        # feat: num_nodes, num_heads * batch_size, head_units
        # timespan: num_nodes, num_heads * batch_size, 1
        timespan = timespan.tile([1, self.num_heads, 1])
        mark_units = th.cat([feat, timespan], dim=2)
        mark_units = self.fc_i(mark_units)
        # => num_nodes, num_heads * batch_size, head_units * num_marks
        mark_units = th.sigmoid(mark_units)
        # => 1, num_nodes * num_heads * batch_size, head_units
        mark_units = th.cat(mark_units.chunk(num_nodes, dim=0), dim=1)
        # => num_marks, num_nodes * num_heads * batch_size, head_units
        mark_units = th.cat(mark_units.chunk(self.num_marks, dim=2), dim=0)
        # => num_nodes * num_heads * batch_size, num_marks, head_units
        mark_units = mark_units.permute(1, 0, 2)

        # Perform mark-wise projection and intensity
        prefix_shape = mark_units.shape[0]
        # => num_nodes * num_heads * batch_size, num_marks, head_units
        weight_i = self.weight_i.unsqueeze(0).tile(prefix_shape, 1, 1)

        # => num_marks
        scale_i = th.exp(self.scale_i)
        # =>  num_nodes * num_heads * batch_size, num_marks
        scale_i = scale_i.unsqueeze(0).tile(prefix_shape, 1)

        # => num_nodes * num_heads * batch_size, num_marks
        mark_inty = th.sum(mark_units * weight_i, dim=2) / scale_i
        mark_inty = scale_i * th.log(1. + th.exp(mark_inty))
        # => num_nodes, num_heads * batch_size, num_marks
        mark_inty = th.stack(mark_inty.chunk(num_nodes, dim=0), dim=0)
        return mark_inty

    def forward(self, graph, feat):
        with graph.local_scope():
            if not isinstance(feat, tuple):
                raise RuntimeError("input should be tuples")

            h_src = self.feat_drop(feat[0]['x'])
            h_dst = self.feat_drop(feat[1]['x'])
            timespans = feat[1]['t']
            # event_marks: num_nodes, 1, num_marks
            event_marks = feat[1]['m']

            # Linear transform
            # Q: num_nodes, num_heads * batch_size, head_units
            # K, T, V: num_nodes, num_heads * batch_size, head_units
            Q = th.cat(th.chunk(self.fc_q(h_dst), self.num_heads, dim=2), dim=1)
            K = th.cat(th.chunk(self.fc_k(h_src), self.num_heads, dim=2), dim=1)
            T = th.cat(th.chunk(self.fc_t(h_src), self.num_heads, dim=2), dim=1)
            V = th.cat(th.chunk(self.fc_v(h_src), self.num_heads, dim=2), dim=1)

            #
            # Attention
            graph.srcdata.update({'fk': K, 'ft': T})
            graph.dstdata.update({'fq': Q})
            # edge feature transformation: num_edges, num_heads * batch_size, head_units
            batch_size = Q.shape[1] // self.num_heads
            efeat = self.fc_e(graph.edata['ef']).unsqueeze(1)
            efeat = efeat.tile((1, batch_size, 1))
            graph.edata['ef'] = th.cat(th.chunk(efeat, self.num_heads, dim=2), dim=1)

            graph.apply_edges(self.msg_fn_prod)
            h_edge = graph.edata.pop('e')
            # compute softmax
            graph.edata['ea'] = dglnn.edge_softmax(graph, h_edge)

            # message passing
            graph.update_all(fn.u_mul_e('ft', 'ea', 'm'), fn.sum('m', 'h'))
            # g_dst: num_nodes, num_heads * batch_size, head_units
            g_dst = graph.dstdata.pop('h')

            #
            # TPPs intensity
            # all_mark_inty: num_nodes, num_heads * batch_size, num_marks
            all_mark_inty = self.intensities(g_dst, timespans)
            graph.srcdata.update({'fi': all_mark_inty, 'fv': V})
            graph.dstdata.update({'fm': event_marks})
            # compute self-modulating probability
            graph.apply_edges(self.msg_fn_inty)
            # message passing
            graph.update_all(fn.u_mul_e('fv', 'ee', 'm'),
                             fn.sum('m', 'h'))
            # outs: num_nodes, num_heads * batch_size, head_units
            outs = graph.dstdata.pop('h')

            # Residual connection
            # outs: num_nodes, batch_size, num_units
            outs = th.cat(th.chunk(outs, self.num_heads, dim=1), dim=2)

            if self.residual:
                outs += h_dst
            return outs


class EncodingLayer(nn.Module):
    def __init__(self, config: EasyDGLConfig):
        super(EncodingLayer, self).__init__()
        num_units = config.num_units
        num_nodes = config.num_nodes

        self.msg_act = nn.LeakyReLU()
        self.msg_drop = nn.Dropout(config.msg_drop)

        self.fc_orin = nn.Linear(num_units, num_units)
        self.fc_adju = nn.Linear(num_units, num_units)
        self.gconv = dglnn.SAGEConv(num_units, num_units, 'mean')
        self.instnorm = nn.InstanceNorm1d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.fc_orin.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc_adju.weight, a=math.sqrt(5))

    def forward(self, g, ag, h):
        # predefined graph
        h_geo = self.fc_orin(h)
        h_geo = self.gconv(g, h_geo, edge_weight=g.edata['ef'])

        # learned (adjusted) graph
        h_adj = self.fc_adju(h)
        h_adj = th.matmul(h_adj.permute(1, 2, 0), ag).permute(2, 0, 1)

        h_out = h_geo + h_adj
        h_out = self.instnorm(h_out.permute(1, 0, 2)).permute(1, 0, 2)
        h_out = self.msg_drop(self.msg_act(h_out))
        return h_out


class NodeRegressor(nn.Module):

    def __init__(self, config: EasyDGLConfig):
        super(NodeRegressor, self).__init__()
        self.num_nodes = config.num_nodes
        self.time_scalor = config.time_scalor
        self.num_timesteps_in = config.num_timesteps_in
        self.num_timesteps_out = config.num_timesteps_out

        self.mark_lookup = th.from_numpy(config.mark_lookup).float()
        self.tcoding = TimeSinusoidCoding(config.num_units)

        num_units = config.num_units
        num_features = config.num_features
        self.fc_x = nn.Linear(num_features * self.num_timesteps_in, num_units, bias=False)
        self.embedding = nn.Parameter(th.empty(self.num_nodes, num_units))

        # CaM
        self.mask_rate = config.mask_rate
        self.mask_embedding = nn.Embedding(config.mask_max + 1, num_units, padding_idx=0)
        if self.mask_rate == 0.:
            logging.warning("Masking is diabled in NodeRegressor")

        # adjusted graph,
        # note that it might be better to
        # compute a graph based on temporal data
        # rather than to learn a runtime graph,
        # which is too much time consuming for large graphs
        self.saturation = 3.
        self.rff = nn.Parameter(th.empty(self.num_nodes, num_units))

        self.k = 3
        self.num_blocks = config.num_blocks
        # Linear
        self.fc_h = nn.ModuleList()
        for _ in range(self.k, self.num_blocks):
            self.fc_h.append(nn.Linear(num_units, num_units))

        # SAGEConv
        self.layers = nn.ModuleList()
        for _ in range(self.k):
            self.layers.append(EncodingLayer(config))
        self.msg_glu = GLU(self.k * num_units + num_units, num_units)
        self.msg_act = nn.LeakyReLU()
        self.msg_drop = nn.Dropout(config.msg_drop)

        # AIAConv
        num_marks = config.num_marks
        feat_drop = config.feat_drop
        att_drop = config.att_drop
        num_heads = config.num_heads
        for _ in range(self.k, self.num_blocks):
            self.layers.append(
                DGLAIAConv(num_units, num_units, num_heads,
                           num_marks, feat_drop, att_drop))

        # Estimator Projection
        hidden_units = num_units + self.num_timesteps_out * num_features
        self.fc_combined = nn.Linear(hidden_units, num_units * self.num_timesteps_out)
        self.combined_glu = GLU(num_units, num_units)
        self.qfeat_glu = GLU(num_units, num_units)
        self.layernorm = nn.LayerNorm(2 * num_units)
        self.fc_o = nn.Linear(2 * num_units, 1, bias=False)
        self.bias_o = nn.Parameter(th.empty(self.num_nodes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.fc_x.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc_combined.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc_o.weight, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        nn.init.zeros_(self.mask_embedding.weight)
        nn.init.zeros_(self.bias_o)
        nn.init.kaiming_uniform_(self.rff, a=math.sqrt(5))

    def mask(self, tensor, feat: dict):
        if not self.training or self.mask_rate == 0.:
            return tensor

        # mask_buskets: num_nodes, num_timesteps_out * batch_size, 1
        mask_buskets = feat['p'].int().squeeze(2)
        mask_embedding = self.mask_embedding(mask_buskets)

        mask_sign = feat['p'].sign()
        # print(tensor.shape, mask_sign.shape, mask_embedding.shape)
        tensor = tensor * (1. - mask_sign) + mask_embedding * mask_sign
        return tensor

    def forward(self, graph, feat):
        # nfeat: batch_size, num_timesteps_in, num_nodes, num_features
        nfeat: th.Tensor = feat['x']
        # => num_nodes, batch_size, num_timesteps_in * num_features
        nfeat = th.cat(nfeat.chunk(self.num_timesteps_in, dim=1), dim=3)
        nfeat = nfeat.squeeze(1).permute(1, 0, 2)

        # mark_lookup: num_nodes, 1, num_marks
        mark_lookup = self.mark_lookup.unsqueeze(1).to(nfeat.device)

        # tfeat: batch_size, num_timesteps_out, num_nodes
        tfeat: th.Tensor = feat['t'] / self.time_scalor
        # tfeat: num_nodes, num_timesteps_out * batch_size, 1
        tfeat = th.cat(tfeat.chunk(self.num_timesteps_out, dim=1), dim=0)
        tfeat = tfeat.permute(2, 0, 1)

        # ag: num_nodes, num_nodes
        rrf = self.rff
        ag = th.matmul(rrf, rrf.permute(1, 0))
        ag = th.tanh(ag * self.saturation)
        ag = th.relu(ag) + th.eye(self.num_nodes).to(nfeat.device)

        # ==== SAGEConv ====
        # layer_previous: num_nodes, batch_size, num_units
        layer_previous = th.relu(self.fc_x(nfeat)) + self.embedding.unsqueeze(1)
        msg_prop = [layer_previous]
        for i in range(self.k):
            layer_previous = self.layers[i](graph, ag, layer_previous)
            msg_prop.append(layer_previous)
        sfeat = self.msg_glu(th.cat(msg_prop, dim=2))
        layer_previous = sfeat

        # ==== AIAConv ====
        # tcodings: 1, 13, num_units
        timestamps = th.arange(1 + self.num_timesteps_out).unsqueeze(0)
        tcodings = self.tcoding(timestamps.to(nfeat.device))
        # x_src: num_nodes, num_timesteps_out * batch_size, num_units
        # tcodings_src: num_units
        tcodings_src = tcodings[0, 0]
        x_src = layer_previous + tcodings_src
        x_src = x_src.tile(1, self.num_timesteps_out, 1)
        # tcodings_dst: batch_size, num_timesteps_out, num_units
        tcodings_dst = tcodings[:, 1:].tile(nfeat.shape[1], 1, 1)
        # => num_timesteps_out * batch_size, 1, num_units
        tcodings_dst = th.cat(tcodings_dst.chunk(self.num_timesteps_out, dim=1), dim=0)
        # => 1, num_timesteps_out * batch_size, num_units
        tcodings_dst = tcodings_dst.permute(1, 0, 2)
        # x_dst: num_nodes, num_timesteps_out * batch_size, num_units
        x_dst = layer_previous.tile(1, self.num_timesteps_out, 1)
        x_dst = self.mask(x_dst, feat) + tcodings_dst
        for i in range(self.k, self.num_blocks):
            layer, linear = self.layers[i], self.fc_h[i - self.k]
            # feat_src, x: num_nodes, num_timesteps_out * batch_size, num_units
            feat_src = {'x': x_src}
            # feat_src, x: num_nodes, num_timesteps_out * batch_size, num_units
            #           t: num_nodes, num_timesteps_out * batch_size, num_units
            #           m: num_nodes, 1, num_marks
            feat_dst = {'x': x_dst,
                        't': tfeat,
                        'm': mark_lookup}
            layer_input = (feat_src, feat_dst)
            layer_geo = layer(graph, layer_input)
            layer_adj = linear(x_dst)
            layer_adj = th.matmul(layer_adj.permute(1, 2, 0), ag).permute(2, 0, 1)
            layer_previous = layer_geo + layer_adj
            layer_previous = self.msg_drop(self.msg_act(layer_previous))
            x_src, x_dst = layer_previous, layer_previous

        # ==== DNN Estimator ====
        # layer_previous: num_nodes, num_timesteps_out * batch_size, num_units
        # => num_nodes, batch_size, num_timesteps_out, num_units
        qfeat = th.stack(layer_previous.chunk(self.num_timesteps_out, dim=1), dim=2)
        # => num_nodes * batch_size, num_timesteps_out, num_units
        qfeat = th.cat(qfeat.chunk(self.num_nodes, dim=0), dim=1)
        qfeat = qfeat.squeeze(0)
        qfeat = self.qfeat_glu(qfeat)

        # nfeat: num_nodes, batch_size, num_timesteps_in * num_features
        # sfeat: num_nodes, batch_size, num_units
        combined = th.cat([nfeat, sfeat], dim=2)
        # => 1, num_nodes * batch_size, num_timesteps_in * num_features + num_units
        combined = th.cat(combined.chunk(self.num_nodes, dim=0), dim=1)
        # => num_nodes * batch_size, 1, num_timesteps_out * num_units
        combined = combined.permute(1, 0, 2)
        combined = self.fc_combined(combined)
        # => num_nodes * batch_size, num_timesteps_out, num_units
        combined = th.cat(combined.chunk(self.num_timesteps_out, dim=2), dim=1)
        combined = self.combined_glu(combined)

        # num_nodes * batch_size, num_timesteps_out, 2 * num_units
        layer_outs = th.cat([qfeat, combined], dim=2)
        layer_outs = self.layernorm(layer_outs.permute(1, 0, 2)).permute(1, 0, 2)

        # num_nodes * batch_size, num_timesteps_out, 1
        layer_outs = self.fc_o(layer_outs)
        # => batch_size, num_timesteps_out, num_nodes
        layer_outs = th.cat(layer_outs.chunk(self.num_nodes, dim=0), dim=2)
        layer_outs = layer_outs + self.bias_o
        return layer_outs

    @staticmethod
    def loss(y_pred, y_true):
        mask = (y_true != 0).float()
        mask /= mask.mean()
        loss = th.abs(y_pred - y_true)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return loss.mean()
