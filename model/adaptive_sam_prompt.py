import math

import torch
import torch.nn as nn

from .prompt_fusion import PromptFusion


class AdaptiveSAMPrompt(nn.Module):
    """P = G_sigma + gamma * PromptEncoder([G_sigma, M*, G_sigma*M*])."""

    def __init__(self, hidden_channels=16, mlp_hidden=16, base_sigma=25.0,
                 sigma_min=8.0, sigma_max=50.0, eps=1e-6):
        super().__init__()
        assert sigma_min < base_sigma < sigma_max
        self.base_sigma, self.sigma_min, self.sigma_max, self.eps = (
            float(base_sigma), float(sigma_min), float(sigma_max), float(eps))
        # Must remain first: identical initialization to the P10 SAM encoder.
        self.prompt_fusion = PromptFusion(hidden_channels=hidden_channels)
        self.mask_weight_mlp = nn.Sequential(nn.Linear(3, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, 1))
        self.sigma_mlp = nn.Sequential(nn.Linear(3, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, 1))
        self.reliability_mlp = nn.Sequential(nn.Linear(4, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, 1))
        last = self.sigma_mlp[-1]
        nn.init.zeros_(last.weight)
        ratio = (base_sigma - sigma_min) / (sigma_max - sigma_min)
        nn.init.constant_(last.bias, math.log(ratio / (1.0 - ratio)))

    def make_gaussian(self, click_xy, sigma, height, width, dtype, device):
        click_xy, sigma = click_xy.to(device=device, dtype=dtype), sigma.to(device=device, dtype=dtype)
        batch = click_xy.shape[0]
        yy = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
        xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
        dx = xx - click_xy[:, 0].view(batch, 1, 1)
        dy = yy - click_xy[:, 1].view(batch, 1, 1)
        sigma = sigma.view(batch, 1, 1).clamp_min(self.eps)
        return torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma.square())).unsqueeze(1)

    def forward(self, click_xy, sam_masks, sam_scores, return_aux=False):
        sam_masks = sam_masks.float()
        sam_scores = sam_scores.to(sam_masks)
        click_xy = click_xy.to(sam_masks)
        batch, candidates, height, width = sam_masks.shape
        sigma25 = sam_masks.new_full((batch, 1), self.base_sigma)
        g25 = self.make_gaussian(click_xy, sigma25, height, width, sam_masks.dtype, sam_masks.device)
        consistency = (sam_masks * g25).sum((-1, -2)) / g25.sum((-1, -2)).clamp_min(self.eps)
        area = sam_masks.mean((-1, -2))
        alpha = torch.softmax(self.mask_weight_mlp(torch.stack((sam_scores, consistency, area), -1)).squeeze(-1), 1)
        soft_mask = (alpha[:, :, None, None] * sam_masks).sum(1, keepdim=True)
        mean_score, mean_consistency = (alpha * sam_scores).sum(1), (alpha * consistency).sum(1)
        mean_area = soft_mask.mean((-1, -2)).squeeze(1)
        sigma_pred = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(
            self.sigma_mlp(torch.stack((mean_area, mean_score, mean_consistency), 1)))
        if candidates > 1:
            certainty = 1.0 + (alpha * torch.log(alpha.clamp_min(self.eps))).sum(1) / math.log(candidates)
        else:
            certainty = torch.ones_like(mean_score)
        gamma = torch.sigmoid(self.reliability_mlp(torch.stack((mean_score, mean_consistency, mean_area, certainty), 1)))
        sigma = (1.0 - gamma) * self.base_sigma + gamma * sigma_pred
        gaussian = self.make_gaussian(click_xy, sigma, height, width, sam_masks.dtype, sam_masks.device)
        delta = self.prompt_fusion.prompt_encoder(torch.cat((gaussian, soft_mask, gaussian * soft_mask), 1))
        position = gaussian + gamma.view(batch, 1, 1, 1) * delta
        if not return_aux:
            return position
        return position, {'alpha': alpha, 'soft_mask': soft_mask, 'sigma': sigma.squeeze(1),
                          'sigma_pred': sigma_pred.squeeze(1), 'gamma': gamma.squeeze(1),
                          'click_consistency': consistency, 'area_k': area, 'delta_position': delta}
