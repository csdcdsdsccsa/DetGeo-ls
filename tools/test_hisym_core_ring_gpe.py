import torch

from model.hisym_core_ring_gpe import HiSymCoreRingGPE


def build_g25(batch=2, height=256, width=256, sigma=25.0):
    yy = torch.arange(height, dtype=torch.float32).view(1, height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, 1, width)
    cx = torch.full((batch, 1, 1), 120.0)
    cy = torch.full((batch, 1, 1), 130.0)
    return torch.exp(-((xx - cx).square() + (yy - cy).square()) / (2.0 * sigma ** 2)).unsqueeze(1)


torch.manual_seed(2024)
rgb = torch.randn(2, 3, 256, 256)
module = HiSymCoreRingGPE(core_sigma=25.0, outer_sigma=50.0)
output = module(rgb, build_g25())

assert output.shape == rgb.shape
assert module.last_diagnostics['ring_max'].item() > 0.0
assert module.last_diagnostics['ring_center_mean'].abs().item() < 1e-6
output.square().mean().backward()
assert module.conv.weight.grad is not None
assert module.conv.weight.grad.abs().sum().item() > 0
print('HiSym-CRGPE functional checks passed')
