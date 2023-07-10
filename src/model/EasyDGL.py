"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging
import math
import pickle

import torch as th
import torch.nn as nn

from model.util import PositionCoding, TimeSinusoidCoding, batch_gather


class EasyDGLConfig(object):
    def __init__(self, yaml_config):
        self.seqs_len = yaml_config.get('seqs_len')
        self.mask_const = yaml_config['data']['mask_const']
        self.mask_len = yaml_config['data']['mask_len']
        self.time_scalor = yaml_config['data']['time_scalor']

        model_config = yaml_config['model']
        self.n_classes = model_config.get('n_classes')
        self.num_blocks = model_config.get('num_blocks')
        self.num_units = model_config.get('num_units')
        self.num_heads = model_config.get('num_heads')
        self.msg_drop = model_config.get('msg_drop')
        self.att_drop = model_config.get('att_drop')

        self.mark_lookup = pickle.load(open(model_config['fmark'], 'rb')).toarray()
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


class Recommener(nn.Module):
    """
        for personalized recommendation
    """

    def __init__(self, config: EasyDGLConfig):
        super(Recommener, self).__init__()
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
