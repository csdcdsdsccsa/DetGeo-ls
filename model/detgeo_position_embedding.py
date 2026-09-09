"""The original DetGeo 4-to-3 click-position encoder, kept as an explicit module."""

import torch.nn as nn

from .darknet import ConvBatchNormReLU


class DetGeoPositionEmbedding(nn.Module):
    """Exact DetGeo ``combine_clickptns_conv``: Conv1x1 + BN + LeakyReLU."""

    def __init__(self):
        super().__init__()
        self.encoder = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=True, instance=False)

    def forward(self, inputs):
        return self.encoder(inputs)
