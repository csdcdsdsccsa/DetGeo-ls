import torch
import torch.nn as nn


class HiSymCoreRingGPE(nn.Module):
    """HiSym residual fusion of an input Gaussian core and a wider ring.

    The incoming core map is deliberately used without reconstruction.  The
    click is recovered only to construct the outer Gaussian in float32, which
    avoids trying to invert underflowed values from the input map.
    """

    def __init__(self, core_sigma=25.0, outer_sigma=50.0, core_sigma_x=None, outer_sigma_x=None):
        super().__init__()
        self.core_sigma_y = float(core_sigma)
        self.core_sigma_x = self.core_sigma_y if core_sigma_x is None else float(core_sigma_x)
        self.outer_sigma_y = float(outer_sigma)
        self.outer_sigma_x = self.outer_sigma_y if outer_sigma_x is None else float(outer_sigma_x)
        if self.core_sigma_y <= 0 or self.core_sigma_x <= 0:
            raise ValueError('core sigmas must be positive')
        if self.outer_sigma_y <= self.core_sigma_y:
            raise ValueError('outer_sigma_y must be larger than core_sigma_y')
        if self.outer_sigma_x <= self.core_sigma_x:
            raise ValueError('outer_sigma_x must be larger than core_sigma_x')
        # Keep legacy attributes for existing diagnostics and external callers.
        self.core_sigma = self.core_sigma_y
        self.outer_sigma = self.outer_sigma_y

        # Natural PyTorch initialization: this module intentionally consumes
        # the ordinary RNG stream, with no padding, private seed, or restore.
        self.conv = nn.Conv2d(5, 3, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.last_diagnostics = {}

    @staticmethod
    def _recover_click(core_gaussian):
        batch, _, _, width = core_gaussian.shape
        with torch.no_grad():
            flat_index = core_gaussian[:, 0].reshape(batch, -1).argmax(dim=1)
            click_y = torch.div(flat_index, width, rounding_mode='floor')
            click_x = flat_index % width
        return click_x, click_y

    @staticmethod
    def _build_gaussian(click_x, click_y, sigma_y, sigma_x, height, width, device, output_dtype):
        yy = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1)
        xx = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width)
        batch = click_x.shape[0]
        cx = click_x.to(device=device, dtype=torch.float32).view(batch, 1, 1)
        cy = click_y.to(device=device, dtype=torch.float32).view(batch, 1, 1)
        dy2 = (yy - cy).square()
        dx2 = (xx - cx).square()
        gaussian = torch.exp(-(dy2 / (2.0 * float(sigma_y) ** 2) + dx2 / (2.0 * float(sigma_x) ** 2)))
        return gaussian.unsqueeze(1).to(dtype=output_dtype)

    def forward(self, rgb, core_gaussian):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('rgb must be [B,3,H,W]')
        expected = (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3])
        if tuple(core_gaussian.shape) != expected:
            raise ValueError('core_gaussian must be {}, got {}'.format(expected, tuple(core_gaussian.shape)))

        # Preserve the dataset-provided G25 exactly.
        g_core = core_gaussian
        click_x, click_y = self._recover_click(g_core)
        g_outer = self._build_gaussian(click_x, click_y, self.outer_sigma_y, self.outer_sigma_x,
                                       rgb.shape[-2], rgb.shape[-1], rgb.device, g_core.dtype)
        g_ring = torch.clamp(g_outer - g_core, min=0.0)
        residual = self.relu(self.bn(self.conv(torch.cat((rgb, g_core, g_ring), dim=1))))
        output = rgb + residual

        indices = torch.arange(g_ring.shape[0], device=g_ring.device)
        self.last_diagnostics = {
            'core_sigma': self.core_sigma,
            'outer_sigma': self.outer_sigma,
            'core_sigma_y': self.core_sigma_y,
            'core_sigma_x': self.core_sigma_x,
            'outer_sigma_y': self.outer_sigma_y,
            'outer_sigma_x': self.outer_sigma_x,
            'core_mean': g_core.detach().mean(),
            'outer_mean': g_outer.detach().mean(),
            'ring_mean': g_ring.detach().mean(),
            'ring_max': g_ring.detach().max(),
            'ring_center_mean': g_ring[indices, 0, click_y, click_x].detach().mean(),
        }
        return output
