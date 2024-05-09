# src/modules/autoencodermodules/pose_encoder.py
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from torch.autograd import Variable

POSE_DIM = 6
LHW_DIM = 3
HIDDEN_DIM_1_DIV = 8
HIDDEN_DIM_2_DIV = 4

class PoseEncoder(nn.Module): 
    """
    PoseEncoder is a module that encodes pose features into image features.
    """

    def __init__(self, enc_feat_dims, pose_feat_dims, activation="relu"):
        """
        Initializes a new instance of the PoseEncoder class.

        Args:
            enc_feat_dims (int): Dimensionality of the image features.
            pose_feat_dims (int): Dimensionality of the pose features.
        """
        super(PoseEncoder, self).__init__()
        
        hidden_dim_1 = enc_feat_dims // HIDDEN_DIM_1_DIV
        hidden_dim_2 = enc_feat_dims // HIDDEN_DIM_2_DIV
        
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "softplus":
            activation = nn.Softplus()
        else:
            raise ValueError("Invalid activation function. Please provide a valid activation function in ['relu', 'softplus'].")
        
        self.fc = nn.Sequential(
            nn.Linear(pose_feat_dims, hidden_dim_1),
            activation,
            nn.Linear(hidden_dim_1, hidden_dim_2),
            activation,
            nn.Linear(hidden_dim_2, enc_feat_dims)
        )
        
    def forward(self, x):
        """
        Forward pass of the PoseEncoder module.

        Args:
            x (tensor): Input tensor containing pose features.

        Returns:
            tensor: Output tensor containing encoded image features.
        """
        return self.fc(x)

class PoseEncoderSpatialVAE(nn.Module):
    def __init__(self, num_classes=1, num_channels=16, n=16, m=16, activation="swish", hidden_dim=500, num_layers=2):
        #softplus=False, resid=False, expand_coords=False, bilinear=False):
        super(PoseEncoderSpatialVAE, self).__init__()
        latent_dim = POSE_DIM + LHW_DIM + num_classes # 10 = 6 + 3 + 1
        n_out = num_channels * n * m # 16 * 16 * 16 = 4096
        if activation == "swish":
            activation = nn.SiLU
        elif activation == "tanh":
            activation = nn.Tanh
        else:
            activation = nn.ReLU
        
        self.coord_linear = nn.Linear(latent_dim, hidden_dim)
        self.latent_dim = latent_dim
        if latent_dim > 0:
            self.latent_linear = nn.Linear(latent_dim, hidden_dim, bias=False)

        layers = [activation()]
        for _ in range(1,num_layers):
            layers.append(nn.Linear(hidden_dim,hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, n_out))

        self.layers = nn.Sequential(*layers)
        
        ## x coordinate array
        xgrid = np.linspace(-1, 1, m)
        ygrid = np.linspace(1, -1, n)
        x0,x1 = np.meshgrid(xgrid, ygrid)
        x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
        x_coord = torch.from_numpy(x_coord).float()
        if torch.cuda.is_available():
            x_coord = x_coord.cuda()
        
        self.x = Variable(x_coord)
    
    def forward(self, z):        
        x = self.x.contiguous()
        
        # print("x", x.size())
        # print("z", z.size())
        # print("self.coord_linear", self.coord_linear)
        # print("self.latent_linear", self.latent_linear)
        
        h_x = self.coord_linear(x) 
        # print("h_x", h_x.size()) 
        h_z = self.latent_linear(z)
        # print("h_z", h_z.size())
        
        h = h_x + h_z
        # print("h", h.size())
        y = self.layers(h) # (batch*num_coords, nout) 
        # print("self.layers", self.layers)
        # print("y", y.size())      
        return y
    