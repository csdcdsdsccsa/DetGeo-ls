# -*- coding: utf-8 -*-
"""SMGeo-style anchor-free detection head.

The module follows the public SMGeo implementation: a shared two-layer
convolutional tower with a center heatmap branch and a four-value box branch.
"""

import torch.nn as nn


class AnchorFreeHead(nn.Module):
    """Predict a center heatmap and (dx, dy, width, height) per grid cell."""

    def __init__(self, in_channels, feat_channels=256, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.head_heatmap = nn.Conv2d(feat_channels, num_classes, 1)
        self.head_bbox = nn.Conv2d(feat_channels, 4, 1)
        nn.init.constant_(self.head_heatmap.bias, -2.0)

    def forward(self, x):
        features = self.conv(x)
        return self.head_heatmap(features), self.head_bbox(features)
