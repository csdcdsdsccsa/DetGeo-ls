"""Gaussian-map geometry refinements for the DetGeo 4-to-3 front end.

Both modules retain the original DetGeo encoder as their base path.  Their
residual projection is zero-initialized, so construction is an exact DetGeo
front-end at step zero while remaining fully trainable under normal RNG use.
"""

import torch
import torch.nn as nn

from .detgeo_position_embedding import DetGeoPositionEmbedding


def recover_gaussian_geometry(gaussian_map, sigma):
    """Recover ``G, Gx, Gy, Gr`` from a Gaussian click map of shape [B,H,W]."""
    if gaussian_map.dim() != 3:
        raise RuntimeError('Gaussian geometry expects [B,H,W], got {}'.format(tuple(gaussian_map.shape)))
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    g = gaussian_map.clamp(min=0.0, max=1.0)
    batch, height, width = g.shape
    with torch.no_grad():
        click = g.reshape(batch, -1).argmax(dim=1)
        click_y = torch.div(click, width, rounding_mode='floor').to(g.dtype)
        click_x = (click % width).to(g.dtype)
        rows = torch.arange(height, device=g.device, dtype=g.dtype).view(1, height, 1)
        cols = torch.arange(width, device=g.device, dtype=g.dtype).view(1, 1, width)
        dx = (cols - click_x.view(batch, 1, 1)) / float(sigma)
        dy = (rows - click_y.view(batch, 1, 1)) / float(sigma)
        gx = dx * g
        gy = dy * g
        gr = (dx.square() + dy.square() - 2.0) * g
    return g, gx, gy, gr


def _residual_encoder(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(4, 16), nn.GELU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(4, 16), nn.GELU(),
    )


class DGPositionEmbedding(nn.Module):
    """DetGeo PE plus a zero-init residual built from ``[G, Gx, Gy]``."""

    def __init__(self, gaussian_sigma=25.0):
        super().__init__()
        self.gaussian_sigma = float(gaussian_sigma)
        self.base_encoder = DetGeoPositionEmbedding()
        self.geometry_encoder = _residual_encoder(3)
        self.output_projection = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.last_diagnostics = {}

    def forward(self, inputs):
        if inputs.dim() != 4 or inputs.shape[1] != 4:
            raise RuntimeError('DG-PE expects [B,4,H,W], got {}'.format(tuple(inputs.shape)))
        base = self.base_encoder(inputs)
        g, gx, gy, _ = recover_gaussian_geometry(inputs[:, 3], self.gaussian_sigma)
        delta = self.output_projection(self.geometry_encoder(torch.stack((g, gx, gy), dim=1)))
        output = base + torch.tanh(self.alpha) * delta
        self.last_diagnostics = {
            'delta_abs_mean': delta.detach().abs().mean(),
            'alpha': torch.tanh(self.alpha.detach()),
            'identity_error': (output.detach() - base.detach()).abs().max(),
        }
        return output


class DDGPositionEmbedding(nn.Module):
    """DetGeo PE plus the DG residual extended only by radial term ``Gr``.

    This deliberately uses the exact DG single-branch layout.  The sole
    architectural difference is the 4-channel ``[G, Gx, Gy, Gr]`` input.
    """

    def __init__(self, gaussian_sigma=25.0):
        super().__init__()
        self.gaussian_sigma = float(gaussian_sigma)
        self.base_encoder = DetGeoPositionEmbedding()
        self.geometry_encoder = _residual_encoder(4)
        self.output_projection = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.last_diagnostics = {}

    def forward(self, inputs):
        if inputs.dim() != 4 or inputs.shape[1] != 4:
            raise RuntimeError('DDG-PE expects [B,4,H,W], got {}'.format(tuple(inputs.shape)))
        base = self.base_encoder(inputs)
        g, gx, gy, gr = recover_gaussian_geometry(inputs[:, 3], self.gaussian_sigma)
        geometry = torch.stack((g, gx, gy, gr), dim=1)
        delta = self.output_projection(self.geometry_encoder(geometry))
        output = base + torch.tanh(self.alpha) * delta
        self.last_diagnostics = {
            'geometry_shape': tuple(geometry.shape),
            'g_abs_mean': g.detach().abs().mean(),
            'gx_abs_mean': gx.detach().abs().mean(),
            'gy_abs_mean': gy.detach().abs().mean(),
            'gr_abs_mean': gr.detach().abs().mean(),
            'delta_abs_mean': delta.detach().abs().mean(),
            'alpha': torch.tanh(self.alpha.detach()),
            'identity_error': (output.detach() - base.detach()).abs().max(),
        }
        return output


class RDGPositionEmbedding(nn.Module):
    """DetGeo PE plus radial/directional geometry residual fusion."""

    def __init__(self, gaussian_sigma=25.0):
        super().__init__()
        self.gaussian_sigma = float(gaussian_sigma)
        self.base_encoder = DetGeoPositionEmbedding()
        self.radial_encoder = _residual_encoder(1)
        self.direction_encoder = _residual_encoder(3)
        self.selector = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1), nn.GELU(), nn.Conv2d(16, 1, kernel_size=1),
        )
        self.output_projection = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.last_diagnostics = {}

    def forward(self, inputs):
        if inputs.dim() != 4 or inputs.shape[1] != 4:
            raise RuntimeError('RDG-PE expects [B,4,H,W], got {}'.format(tuple(inputs.shape)))
        base = self.base_encoder(inputs)
        g, gx, gy, gr = recover_gaussian_geometry(inputs[:, 3], self.gaussian_sigma)
        radial = self.radial_encoder(g.unsqueeze(1))
        directional = self.direction_encoder(torch.stack((gx, gy, gr), dim=1))
        weight = torch.sigmoid(self.selector(torch.cat((radial, directional), dim=1)))
        delta = self.output_projection(weight * radial + (1.0 - weight) * directional)
        output = base + torch.tanh(self.alpha) * delta
        self.last_diagnostics = {
            'delta_abs_mean': delta.detach().abs().mean(),
            'alpha': torch.tanh(self.alpha.detach()),
            'selector_mean': weight.detach().mean(),
            'identity_error': (output.detach() - base.detach()).abs().max(),
        }
        return output
