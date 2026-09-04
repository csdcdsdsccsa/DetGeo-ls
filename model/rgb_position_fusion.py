import torch
import torch.nn as nn


class RGBPositionFusion(nn.Module):
    """Zero-residual bidirectional RGB/position interaction before ResNet18."""

    def __init__(self, hidden_channels=16, leaky=0.1):
        super().__init__()
        self.rgb_proj = nn.Sequential(nn.Conv2d(3, hidden_channels, 3, padding=1), nn.LeakyReLU(leaky, inplace=True))
        self.position_proj = nn.Sequential(nn.Conv2d(1, hidden_channels, 3, padding=1), nn.LeakyReLU(leaky, inplace=True))
        self.position_to_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.rgb_to_position = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.out_proj = nn.Conv2d(hidden_channels * 2, 3, 1)
        self.interaction_scale = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, position_map, base_fused):
        rgb_feature = self.rgb_proj(rgb)
        position_feature = self.position_proj(position_map)
        rgb_guided = rgb_feature * (1.0 + torch.sigmoid(self.position_to_rgb(position_feature)))
        position_guided = position_feature * (1.0 + torch.sigmoid(self.rgb_to_position(rgb_feature)))
        interaction = self.out_proj(torch.cat((rgb_guided, position_guided), dim=1))
        return base_fused + self.interaction_scale * interaction
