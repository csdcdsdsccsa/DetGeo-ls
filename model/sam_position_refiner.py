import math

import torch
import torch.nn as nn

from .prompt_fusion import PromptFusion


class SAMPositionRefiner(nn.Module):
    """Refine DetGeo's original click prior with an offline SAM multi-mask."""

    def __init__(self, hidden_channels=16, mlp_hidden=16, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        # Keep this first: P10 requires its initialization to match B and D.
        self.prompt_fusion = PromptFusion(hidden_channels=hidden_channels)
        self.mask_weight_mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, 1)
        )
        self.reliability_mlp = nn.Sequential(
            nn.Linear(4, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, 1)
        )

    def forward(self, position_base, sam_masks, sam_scores, return_aux=False):
        if position_base.ndim != 4 or position_base.shape[1] != 1:
            raise ValueError('position_base must be [B,1,H,W]')
        if sam_masks.ndim != 4 or sam_scores.ndim != 2:
            raise ValueError('SAM masks/scores must be [B,K,H,W] and [B,K]')
        position_base = position_base.float()
        sam_masks = sam_masks.to(position_base.device, position_base.dtype)
        sam_scores = sam_scores.to(position_base.device, position_base.dtype)
        batch, count, height, width = sam_masks.shape
        if position_base.shape != (batch, 1, height, width) or sam_scores.shape != (batch, count):
            raise ValueError('incompatible positional-prior and SAM shapes')

        prior_sum = position_base.sum(dim=(-1, -2)).clamp_min(self.eps)
        click_consistency = (sam_masks * position_base).sum(dim=(-1, -2)) / prior_sum
        area_k = sam_masks.mean(dim=(-1, -2))
        features = torch.stack((sam_scores, click_consistency, area_k), dim=-1)
        alpha = torch.softmax(self.mask_weight_mlp(features).squeeze(-1), dim=1)
        soft_mask = (alpha[:, :, None, None] * sam_masks).sum(dim=1, keepdim=True)

        mean_score = (alpha * sam_scores).sum(dim=1)
        mean_consistency = (alpha * click_consistency).sum(dim=1)
        mean_area = soft_mask.mean(dim=(-1, -2)).squeeze(1)
        if count > 1:
            entropy = -(alpha * torch.log(alpha.clamp_min(self.eps))).sum(dim=1) / math.log(count)
            certainty = 1.0 - entropy
        else:
            certainty = torch.ones_like(mean_score)
        gamma_features = torch.stack((mean_score, mean_consistency, mean_area, certainty), dim=1)
        gamma = torch.sigmoid(self.reliability_mlp(gamma_features))

        prompt_inputs = torch.cat((position_base, soft_mask, position_base * soft_mask), dim=1)
        # PromptFusion's public forward adds a Gaussian base; here we require delta only.
        delta_position = self.prompt_fusion.prompt_encoder(prompt_inputs)
        position_map = position_base + gamma.view(batch, 1, 1, 1) * delta_position
        if not return_aux:
            return position_map
        return position_map, {
            'alpha': alpha, 'soft_mask': soft_mask, 'area_k': area_k,
            'click_consistency': click_consistency, 'mean_score': mean_score,
            'mean_consistency': mean_consistency, 'mean_area': mean_area,
            'certainty': certainty, 'gamma': gamma.squeeze(1),
            'delta_position': delta_position,
        }
