import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class Automap(nn.Module):
    def __init__(self, m, K, dim_bottleneck=64, conv_channels=64):
        """PyTorch implementation of AUTOMAP
        Zhu, B., Liu, J. Z., Cauley, S. F., Rosen, B. R., & Rosen, M. S. (2018). 
        Image reconstruction by domain-transform manifold learning. Nature, 555(7697), 487-492
        """
        super().__init__()
        self.K = K
        self.m = m
        self.res = nn.Sequential(
            nn.Linear(in_features = 2*m, out_features = dim_bottleneck),
            nn.Tanh(),
            nn.Linear(in_features = dim_bottleneck, out_features = K**2),
            #nn.BatchNorm1d(K**2),
            nn.Unflatten(dim=1, unflattened_size=(1, K, K)),
            nn.Conv2d(in_channels =1, out_channels = conv_channels, kernel_size=5, padding = 2),
            nn.Tanh(),
            nn.Conv2d(in_channels = conv_channels, out_channels = conv_channels, kernel_size=5, padding = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = conv_channels, out_channels = 1, kernel_size=7, padding=3)
        )

    def forward(self, kspace, mask):
        kspace = torch.view_as_real(kspace)
        sampled_kspace = kspace[:, mask > 0]
        x = torch.flatten(sampled_kspace, start_dim = 1)
        x = self.res(x)
        x = x.squeeze(1)

        return x 