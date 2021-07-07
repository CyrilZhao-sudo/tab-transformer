# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/7/2

'''
reference：
    https://github.com/CyrilZhao-sudo/tab-transformer-pytorch
    https://github.com/rixwew/pytorch-fm
'''

import torch
import math
import torch.nn.functional as F
import numpy as np


class ScaleDotProductAttention(torch.nn.Module):
    def __init__(self, dropout=0.5, **kwargs):
        super(ScaleDotProductAttention, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        d = q.size()[-1]
        attn_scores = torch.matmul(q, k.permute(0, 2, 1)) / math.sqrt(d)
        if mask:
            attn_scores = torch.masked_fill(attn_scores, mask == 0, -1e9)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        att_output = torch.matmul(attn_scores, v)
        return att_output


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_head, dropout=0.5, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.n_head = n_head
        self.head_dim = input_dim // n_head
        self.q_w = torch.nn.Linear(input_dim, n_head * self.head_dim, bias=False)
        self.k_w = torch.nn.Linear(input_dim, n_head * self.head_dim, bias=False)
        self.v_w = torch.nn.Linear(input_dim, n_head * self.head_dim, bias=False)
        self.out = torch.nn.Linear(n_head * self.head_dim, n_head * self.head_dim, bias=False)
        self.attention = ScaleDotProductAttention(dropout=dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, input_dim = q.size()
        q = self.q_w(q).reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3).reshape(
            batch_size * self.n_head, seq_len, self.head_dim)
        k = self.k_w(k).reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3).reshape(
            batch_size * self.n_head, seq_len, self.head_dim)
        v = self.v_w(v).reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3).reshape(
            batch_size * self.n_head, seq_len, self.head_dim)
        if mask:
            mask = mask.unsqueeze(1)
        attn_out = self.attention(q, k, v, mask=mask)
        attn_out = attn_out.reshape(batch_size, self.n_head, seq_len, self.head_dim).permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.n_head * self.head_dim)
        out = self.out(attn_out)
        return out


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(torch.nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class AddNormConnection(torch.nn.Module):
    def __init__(self, dim, dropout):
        super(AddNormConnection, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, layer_out):
        x = x + self.dropout(layer_out)
        return self.layer_norm(x)


class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, mult = 4, dropout=0.5, ff_act='GEGLU'):
        super(FeedForward, self).__init__()
        if ff_act == 'GEGLU':
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim * mult * 2),
                GEGLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_dim * mult, input_dim)
            )
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim * mult),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_dim * mult, input_dim)
            )

    def forward(self, x):
        return self.net(x)


class TabTransformerEncoderBlock(torch.nn.Module):
    def __init__(self, input_dim, n_heads, att_dropout, ffn_mult, ffn_dropout, ffn_act, an_dropout):
        super(TabTransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(input_dim, n_head=n_heads, dropout=att_dropout)
        self.ffn = FeedForward(input_dim, mult=ffn_mult, dropout=ffn_dropout, ff_act=ffn_act)
        self.add_norm1 = AddNormConnection(input_dim, dropout=an_dropout)
        self.add_norm2 = AddNormConnection(input_dim, dropout=an_dropout)

    def forward(self, x):
        '''
        单个encoder block
        :param x: embed_x
        :return:
        '''
        att_out = self.attention(x, x, x)
        add_norm1_out = self.add_norm1(x, att_out)
        ffn_out = self.ffn(add_norm1_out)
        out = self.add_norm2(add_norm1_out, ffn_out)
        return out


class TabTransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim, depth, n_heads, att_dropout, ffn_mult, ffn_dropout, ffn_act, an_dropout):
        super(TabTransformerEncoder, self).__init__()
        transformer = []
        for _ in range(depth):
            transformer.append(TabTransformerEncoderBlock(input_dim, n_heads, att_dropout, ffn_mult, ffn_dropout, ffn_act, an_dropout))
        self.transformer = torch.nn.Sequential(*transformer)

    def forward(self, x):
        '''
        N 个 encoder block
        :param x: embedd_x
        :return:
        '''
        out = self.transformer(x)
        return out


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, layer_dims, dropout=0.5, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        for layer_dim in layer_dims:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.BatchNorm1d(layer_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = layer_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)