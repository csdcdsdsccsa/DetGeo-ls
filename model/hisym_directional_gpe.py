import torch
import torch.nn as nn


def _check_inputs(rgb, gaussian):
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError('rgb must be [B,3,H,W]')
    expected = (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3])
    if tuple(gaussian.shape) != expected:
        raise ValueError('gaussian must be {}, got {}'.format(expected, tuple(gaussian.shape)))


def _recover_click(gaussian):
    batch, _, _, width = gaussian.shape
    with torch.no_grad():
        flat_index = gaussian[:, 0].reshape(batch, -1).argmax(dim=1)
        return flat_index % width, torch.div(flat_index, width, rounding_mode='floor')


def _build_direction_fields(gaussian, click_x, click_y):
    batch, _, height, width = gaussian.shape
    yy = torch.arange(height, device=gaussian.device, dtype=torch.float32).view(1, 1, height, 1)
    xx = torch.arange(width, device=gaussian.device, dtype=torch.float32).view(1, 1, 1, width)
    cx = click_x.to(device=gaussian.device, dtype=torch.float32).view(batch, 1, 1, 1)
    cy = click_y.to(device=gaussian.device, dtype=torch.float32).view(batch, 1, 1, 1)
    dx = ((xx - cx) / float(max(width - 1, 1))) * gaussian.to(torch.float32)
    dy = ((yy - cy) / float(max(height - 1, 1))) * gaussian.to(torch.float32)
    return dx.to(gaussian.dtype), dy.to(gaussian.dtype)


def _build_gaussian(click_x, click_y, sigma, height, width, device, output_dtype):
    yy = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1)
    xx = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width)
    batch = click_x.shape[0]
    cx = click_x.to(device=device, dtype=torch.float32).view(batch, 1, 1)
    cy = click_y.to(device=device, dtype=torch.float32).view(batch, 1, 1)
    return torch.exp(-((xx - cx).square() + (yy - cy).square()) / (2.0 * float(sigma) ** 2)).unsqueeze(1).to(output_dtype)


class HiSymDirectionalGPE(nn.Module):
    """[RGB, G25, Dx, Dy] -> Conv(6,3) -> BN -> ReLU -> +RGB."""
    def __init__(self):
        super().__init__()
        # This direct construction intentionally uses the natural RNG stream.
        self.conv = nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.last_diagnostics = {}

    def forward(self, rgb, gaussian):
        _check_inputs(rgb, gaussian)
        click_x, click_y = _recover_click(gaussian)
        dx, dy = _build_direction_fields(gaussian, click_x, click_y)
        output = rgb + self.relu(self.bn(self.conv(torch.cat((rgb, gaussian, dx, dy), dim=1))))
        indices = torch.arange(gaussian.shape[0], device=gaussian.device)
        self.last_diagnostics = {'dx_abs_mean': dx.detach().abs().mean(), 'dy_abs_mean': dy.detach().abs().mean(),
                                 'dx_center_abs_mean': dx[indices, 0, click_y, click_x].detach().abs().mean(),
                                 'dy_center_abs_mean': dy[indices, 0, click_y, click_x].detach().abs().mean()}
        return output


class HiSymDirectionalCoreRingGPE(nn.Module):
    """[RGB, G25, Dx, Dy, G50-G25] -> Conv(7,3) -> BN -> ReLU -> +RGB."""
    def __init__(self, core_sigma=25.0, outer_sigma=50.0):
        super().__init__()
        if core_sigma <= 0 or outer_sigma <= core_sigma:
            raise ValueError('requires 0 < core_sigma < outer_sigma')
        self.core_sigma, self.outer_sigma = float(core_sigma), float(outer_sigma)
        # Do not inherit CRGPE: only this actual 7->3 convolution is initialized.
        self.conv = nn.Conv2d(7, 3, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.last_diagnostics = {}

    def forward(self, rgb, gaussian):
        _check_inputs(rgb, gaussian)
        click_x, click_y = _recover_click(gaussian)
        dx, dy = _build_direction_fields(gaussian, click_x, click_y)
        g50 = _build_gaussian(click_x, click_y, self.outer_sigma, rgb.shape[-2], rgb.shape[-1], rgb.device, gaussian.dtype)
        ring = torch.clamp(g50 - gaussian, min=0.0)
        output = rgb + self.relu(self.bn(self.conv(torch.cat((rgb, gaussian, dx, dy, ring), dim=1))))
        indices = torch.arange(gaussian.shape[0], device=gaussian.device)
        self.last_diagnostics = {'core_sigma': self.core_sigma, 'outer_sigma': self.outer_sigma,
                                 'dx_abs_mean': dx.detach().abs().mean(), 'dy_abs_mean': dy.detach().abs().mean(),
                                 'ring_mean': ring.detach().mean(), 'ring_max': ring.detach().max(),
                                 'ring_center_abs_mean': ring[indices, 0, click_y, click_x].detach().abs().mean()}
        return output
