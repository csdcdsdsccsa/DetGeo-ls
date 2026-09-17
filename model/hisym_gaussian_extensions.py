import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hisym_pae_fusion import HiSymPAEFusion


def _check_inputs(rgb, position_map, name):
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise RuntimeError('{}: rgb must be [B,3,H,W], got {}'.format(name, tuple(rgb.shape)))
    expected = (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3])
    if tuple(position_map.shape) != expected:
        raise RuntimeError('{}: position_map must be {}, got {}'.format(name, expected, tuple(position_map.shape)))


class DGRPEV2(nn.Module):
    """HiSym-GPE followed by a zero-gated deep residual refinement; natural RNG only."""

    def __init__(self):
        super().__init__()
        self.base_fusion = HiSymPAEFusion()
        self.refine_encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.refine_projection = nn.Conv2d(16, 3, 1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.last_diagnostics = {}

    def forward(self, rgb, position_map):
        _check_inputs(rgb, position_map, 'DGRPE-v2')
        base = self.base_fusion(rgb, position_map)
        delta = self.refine_projection(self.refine_encoder(torch.cat((base, position_map), dim=1)))
        alpha = torch.tanh(self.gamma)
        output = base + alpha * delta
        self.last_diagnostics = {
            'gamma': self.gamma.detach(), 'alpha': alpha.detach(),
            'deep_delta_abs_mean': delta.detach().abs().mean(),
            'refine_abs_mean': (output.detach() - base.detach()).abs().mean(),
            'init_error_to_base': (output.detach() - base.detach()).abs().max(),
        }
        return output


class AdaptiveHiSymGPE(nn.Module):
    """Target-aware Gaussian sigma prediction followed by unchanged HiSym fusion; natural RNG only."""

    def __init__(self, base_sigma=25.0):
        super().__init__()
        self.base_sigma = float(base_sigma)
        self.log_sigma_span = math.log(2.0)
        self.base_fusion = HiSymPAEFusion()
        self.sigma_encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=4, padding=2, bias=False), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.sigma_predictor = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.sigma_predictor[-1].weight)
        nn.init.zeros_(self.sigma_predictor[-1].bias)
        self.last_diagnostics = {}

    @staticmethod
    def _recover_click(gaussian):
        batch, _, _, width = gaussian.shape
        with torch.no_grad():
            flat_index = gaussian[:, 0].reshape(batch, -1).argmax(dim=1)
            return flat_index % width, torch.div(flat_index, width, rounding_mode='floor')

    @staticmethod
    def _build_gaussian(click_x, click_y, sigma, height, width, device, output_dtype):
        yy = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1)
        xx = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width)
        batch = sigma.shape[0]
        cx = click_x.to(device=device, dtype=torch.float32).view(batch, 1, 1)
        cy = click_y.to(device=device, dtype=torch.float32).view(batch, 1, 1)
        sigma32 = sigma.to(dtype=torch.float32).view(batch, 1, 1)
        gaussian = torch.exp(-((xx - cx).square() + (yy - cy).square()) / (2.0 * sigma32.square()))
        return gaussian.unsqueeze(1).to(dtype=output_dtype)

    def forward(self, rgb, fixed_gaussian):
        _check_inputs(rgb, fixed_gaussian, 'Adaptive-HiSym-GPE')
        click_x, click_y = self._recover_click(fixed_gaussian)
        feature = self.sigma_encoder(torch.cat((rgb, fixed_gaussian), dim=1))
        gaussian_small = F.interpolate(fixed_gaussian, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        pooled = (feature * gaussian_small).sum(dim=(2, 3)) / (gaussian_small.sum(dim=(2, 3)) + 1e-6)
        raw_sigma = self.sigma_predictor(pooled).squeeze(1)
        sigma = self.base_sigma * torch.exp(self.log_sigma_span * torch.tanh(raw_sigma))
        height, width = rgb.shape[-2:]
        rebuilt_base = self._build_gaussian(click_x, click_y, torch.full_like(sigma, self.base_sigma), height, width, rgb.device, rgb.dtype)
        rebuilt_adaptive = self._build_gaussian(click_x, click_y, sigma, height, width, rgb.device, rgb.dtype)
        effective_gaussian = fixed_gaussian + rebuilt_adaptive - rebuilt_base
        output = self.base_fusion(rgb, effective_gaussian)
        self.last_diagnostics = {
            'raw_sigma_mean': raw_sigma.detach().mean(), 'sigma_mean': sigma.detach().mean(),
            'sigma_min': sigma.detach().min(), 'sigma_max': sigma.detach().max(),
            'gaussian_diff_mean': (effective_gaussian.detach() - fixed_gaussian.detach()).abs().mean(),
            'gaussian_diff_max': (effective_gaussian.detach() - fixed_gaussian.detach()).abs().max(),
        }
        return output
