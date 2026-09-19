import torch

from model.hisym_core_ring_gpe import HiSymCoreRingGPE


def build_g25(batch=2, height=256, width=256, sigma=25.0):
    yy = torch.arange(height, dtype=torch.float32).view(1, height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, 1, width)
    cx = torch.full((batch, 1, 1), 120.0)
    cy = torch.full((batch, 1, 1), 130.0)
    return torch.exp(-((xx - cx).square() + (yy - cy).square()) / (2.0 * sigma ** 2)).unsqueeze(1)


def build_anisotropic_gaussian(batch=2, height=256, width=512, sigma_y=25.0, sigma_x=50.0):
    yy = torch.arange(height, dtype=torch.float32).view(1, height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, 1, width)
    cx = torch.full((batch, 1, 1), 256.0)
    cy = torch.full((batch, 1, 1), 128.0)
    return torch.exp(-((yy - cy).square() / (2.0 * sigma_y ** 2) +
                       (xx - cx).square() / (2.0 * sigma_x ** 2))).unsqueeze(1)


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

rgb_svi = torch.randn(2, 3, 256, 512)
module_svi = HiSymCoreRingGPE(core_sigma=25.0, core_sigma_x=50.0,
                              outer_sigma=50.0, outer_sigma_x=100.0)
output_svi = module_svi(rgb_svi, build_anisotropic_gaussian())
assert output_svi.shape == rgb_svi.shape
assert module_svi.last_diagnostics['core_sigma_y'] == 25.0
assert module_svi.last_diagnostics['core_sigma_x'] == 50.0
assert module_svi.last_diagnostics['outer_sigma_y'] == 50.0
assert module_svi.last_diagnostics['outer_sigma_x'] == 100.0
assert module_svi.last_diagnostics['ring_max'].item() > 0.0
assert module_svi.last_diagnostics['ring_center_mean'].abs().item() < 1e-6
output_svi.square().mean().backward()
assert module_svi.conv.weight.grad is not None
assert module_svi.conv.weight.grad.abs().sum().item() > 0
print('HiSym-CRGPE functional checks passed: isotropic and SVI anisotropic')
