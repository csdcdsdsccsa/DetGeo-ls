"""Unchanged shared torchvision Swin-T Stage3/Stage4 extraction."""

import torch.nn as nn
import torchvision.models as models


class SwinTMultiStageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1).features

    def forward(self, x):
        stage3 = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 5:
                stage3 = x.permute(0, 3, 1, 2).contiguous()
        if stage3 is None:
            raise RuntimeError('Swin-T stage-3 feature was not produced')
        return stage3, x.permute(0, 3, 1, 2).contiguous()
