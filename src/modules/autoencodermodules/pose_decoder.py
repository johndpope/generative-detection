"""
src/modules/autoencodermodules/pose_decoder.py
=======================================================================
Code adapted from https://github.com/tbepler/spatial-VAE.
License provided below.
=======================================================================
MIT License
Copyright (c) 2019 Tristan Bepler
"""

import torch.nn as nn
from src.data.specs import POSE_DIM, LHW_DIM, FILL_FACTOR_DIM

HIDDEN_DIM_1_DIV = 4
HIDDEN_DIM_2_DIV = 8

class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)

class PoseDecoderSpatialVAE(nn.Module):
    def __init__(self, num_classes=2, num_channels=16, n=16, m=16, activation="tanh", **kwargs):
        super(PoseDecoderSpatialVAE, self).__init__()
        n_out = num_channels * n * m # 16 * 16 * 16 = 4096
        inf_dim = ((POSE_DIM + LHW_DIM + FILL_FACTOR_DIM) * 2) + num_classes # (6 + 3 + 1) * 2  + 1 = 21
        activation = nn.Tanh if activation == "tanh" else nn.ReLU
        
        kwargs.update({
            
            "n": n_out,
            "latent_dim": inf_dim,
            "activation": activation,
        })
        latent_dim = inf_dim
        n = n_out
        
        self.latent_dim = latent_dim
        self.n = n
        hidden_dim = kwargs.get("hidden_dim", 500)
        num_layers = kwargs.get("num_layers", 2)
        resid = kwargs.get("resid", False)

        layers = [nn.Linear(n, hidden_dim),
                activation(),
                ]
        for _ in range(1, num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())

        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        # x is (batch,num_coords)
        z = self.layers(x)
        return z