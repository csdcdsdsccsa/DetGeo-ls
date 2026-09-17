import torch
import torch.nn as nn


class DeepGaussianResidualPE(nn.Module):
    """Identity-initialized gated residual encoder for an RGB and Gaussian map."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.GELU(),
        )
        # Keep PyTorch's default initialization: gamma, not this branch, is zero-started.
        self.projection = nn.Conv2d(16, 3, 1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.last_diagnostics = {}

    def forward(self, rgb, position_map):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise RuntimeError('DGRPE rgb must be [B,3,H,W]')
        if position_map.shape != (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]):
            raise RuntimeError('DGRPE position map must be [B,1,H,W]')
        delta = self.projection(self.encoder(torch.cat((rgb, position_map), dim=1)))
        alpha = torch.tanh(self.gamma)
        output = rgb + alpha * delta
        self.last_diagnostics = {
            'gamma': self.gamma.detach(), 'alpha': alpha.detach(),
            'delta_abs_mean': delta.detach().abs().mean(),
            'residual_abs_mean': (output.detach() - rgb.detach()).abs().mean(),
            'identity_error': (output.detach() - rgb.detach()).abs().max(),
        }
        return output
