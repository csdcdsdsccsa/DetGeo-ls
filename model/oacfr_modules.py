"""Building blocks for the E05 coarse-to-fine anchor-free experiment."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def spatial_softmax(logits):
    """Spatial softmax whose mean is one, convenient as a feature gate."""
    batch_size, _, height, width = logits.shape
    probability = torch.softmax(logits.flatten(2), dim=-1).view(batch_size, 1, height, width)
    return probability * float(height * width)


class CoarseToFineSearch(nn.Module):
    """Match one fixed query vector against progressively finer satellite maps."""

    def __init__(self, feature_dim, scales, temperature=0.07, prior_gate_init=0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.scales = tuple(int(scale) for scale in scales)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

        gate_logit = math.log(prior_gate_init / (1.0 - prior_gate_init))
        self.prior_gate_logits = nn.ParameterDict({
            str(scale): nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
            for scale in self.scales[1:]
        })

    def forward(self, query_vector, feature_maps):
        query_vector = F.normalize(query_vector, p=2, dim=1)
        search_logits = []
        prior_logits = None

        for scale in self.scales:
            feature = F.normalize(feature_maps[str(scale)], p=2, dim=1)
            logits = torch.sum(feature * query_vector[:, :, None, None], dim=1, keepdim=True)
            logits = logits / self.temperature

            if prior_logits is not None:
                prior = F.interpolate(
                    torch.softmax(prior_logits.flatten(2), dim=-1).view_as(prior_logits),
                    size=logits.shape[-2:], mode="bilinear", align_corners=False)
                prior = prior / prior.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                gate = torch.sigmoid(self.prior_gate_logits[str(scale)])
                logits = logits + gate * torch.log(prior.clamp_min(1e-8))

            search_logits.append(logits)
            prior_logits = logits

        return search_logits


class AnchorFreeHead(nn.Module):
    """Predict center logits, LTRB distances and localization quality."""

    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, 3),
            ConvNormAct(hidden_channels, hidden_channels, 3),
        )
        self.center_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.ltrb_head = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self.quality_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        nn.init.constant_(self.center_head.bias, -4.595)
        nn.init.constant_(self.quality_head.bias, -2.197)
        nn.init.constant_(self.ltrb_head.bias, 1.0)

    def forward(self, features):
        features = self.stem(features)
        return {
            "center_logits": self.center_head(features),
            "ltrb": F.softplus(self.ltrb_head(features)),
            "quality_logits": self.quality_head(features),
        }
