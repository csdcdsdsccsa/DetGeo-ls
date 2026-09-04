import torch
import torch.nn as nn


class HiSymPAEFusion(nn.Module):
    """HiSymGeo-style ResidualConvFusion: q + ReLU(BN(Conv3x3([q, P])))."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, rgb, position_map):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('rgb must be [B,3,H,W]')
        if position_map.shape != (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]):
            raise ValueError('position_map must be [B,1,H,W] aligned with rgb')
        residual = self.relu(self.bn(self.conv(torch.cat((rgb, position_map), dim=1))))
        return rgb + residual
