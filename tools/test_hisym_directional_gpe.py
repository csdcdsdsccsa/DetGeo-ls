import torch
from model.hisym_directional_gpe import HiSymDirectionalGPE, HiSymDirectionalCoreRingGPE


def make_gaussian(batch=2, height=256, width=256, sigma=25.0):
    yy = torch.arange(height, dtype=torch.float32).view(1, height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, 1, width)
    return torch.exp(-((xx - 120.0).square() + (yy - 130.0).square()) / (2.0 * sigma ** 2)).unsqueeze(1).expand(batch, -1, -1, -1).clone()


torch.manual_seed(2024)
rgb, g25 = torch.randn(2, 3, 256, 256), make_gaussian()
for module, needs_ring in ((HiSymDirectionalGPE(), False), (HiSymDirectionalCoreRingGPE(), True)):
    output = module(rgb, g25)
    assert output.shape == rgb.shape
    diag = module.last_diagnostics
    assert diag['dx_abs_mean'].item() > 0 and diag['dy_abs_mean'].item() > 0
    assert diag.get('dx_center_abs_mean', torch.tensor(0.)).abs().item() < 1e-6
    assert diag.get('dy_center_abs_mean', torch.tensor(0.)).abs().item() < 1e-6
    if needs_ring:
        assert diag['ring_max'].item() > 0 and diag['ring_center_abs_mean'].item() < 1e-6
    output.square().mean().backward()
    assert module.conv.weight.grad is not None and module.conv.weight.grad.abs().sum().item() > 0
print('HiSym directional GPE functional checks passed')
