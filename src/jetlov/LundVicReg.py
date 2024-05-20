# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function

import dgl

# import numpy as np
# import torch
import torch.nn as nn

from jetlov.EdgeConv import EdgeConvBlock


class LundVicReg(nn.Module):
    def __init__(
        self,
        input_dims,
        conv_params=[[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]],
        **kwargs
    ):
        super(LundVicReg, self).__init__(**kwargs)
        self.bn_fts = nn.BatchNorm1d(input_dims)

        self.encoder = nn.ModuleList()
        for idx, channels in enumerate(conv_params):
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][-1]
            self.encoder.append(EdgeConvBlock(in_feat=in_feat, out_feats=channels))

        channel = conv_params[-1][-1]
        self.head = nn.ModuleList(
            [
                nn.Linear(channel, 2 * channel),
                nn.ReLU(),
                nn.Linear(2 * channel, 4 * channel),
                nn.ReLU(),
                nn.Linear(4 * channel, 8 * channel),
            ]
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, batch_graph):
        fts = self.bn_fts(batch_graph.ndata["features"])
        for conv in self.encoder:
            fts = conv(batch_graph, fts)

        batch_graph.ndata["fts"] = fts
        x = dgl.mean_nodes(batch_graph, "fts")
        for layer in self.head:
            x = layer(x)
            if isinstance(layer, nn.ReLU) and self.training:
                x = self.dropout(x)

        # if return_hidden_layer:
        #    return self.head(x), x
        # else:
        return x
