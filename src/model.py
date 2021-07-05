# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/7/2


from src.layers import *
import torch


class TabTransformer(torch.nn.Module):
    def __init__(self, cat_field_dims, cons_dims, embed_dim, depth=2, n_heads=4, att_dropout=0.5, ffn_mult=2, ffn_dropout=0.5, ffn_act='GEGLU', an_dropout=0.5, mlp_dims=[10, 10], mlp_dropout=0.5):
        super(TabTransformer, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims=cat_field_dims, embed_dim=embed_dim)
        self.transformer = TabTransformerEncoder(embed_dim, depth, n_heads, att_dropout, ffn_mult, ffn_dropout, ffn_act, an_dropout)
        self.embed_output_dim = len(cat_field_dims)  * embed_dim + cons_dims
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, layer_dims=mlp_dims, dropout=mlp_dropout)
        self.norm = torch.nn.LayerNorm(cons_dims)


    def forward(self, x_cat, x_cons):
        embed_x = self.embedding(x_cat)
        trans_out = self.transformer(embed_x)
        cons_x = self.norm(x_cons)
        all_x = torch.cat([trans_out.flatten(1), cons_x], dim=-1)
        out = self.mlp(all_x)
        return torch.sigmoid(out.squeeze(1))



