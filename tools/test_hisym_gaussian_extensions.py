import torch

from model.hisym_gaussian_extensions import AdaptiveHiSymGPE, DGRPEV2


def make_gaussian(batch, height=256, width=256, sigma=25.0):
    yy = torch.arange(height, dtype=torch.float32).view(1, height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, 1, width)
    g = torch.exp(-((xx - 120.0).square() + (yy - 130.0).square()) / (2.0 * sigma * sigma))
    return g.expand(batch, -1, -1).unsqueeze(1)


torch.manual_seed(2024)
rgb, g25 = torch.randn(2, 3, 256, 256), make_gaussian(2)
v2 = DGRPEV2().eval()
with torch.no_grad():
    base, output = v2.base_fusion(rgb, g25), v2(rgb, g25)
assert torch.equal(output, base)
assert v2.last_diagnostics['deep_delta_abs_mean'] > 0
assert v2.last_diagnostics['init_error_to_base'] < 1e-7
v2.train(); v2.zero_grad(); v2(rgb, g25).square().mean().backward()
assert v2.gamma.grad is not None and v2.gamma.grad.abs() > 0

adaptive = AdaptiveHiSymGPE(base_sigma=25.0).eval()
with torch.no_grad():
    output = adaptive(rgb, g25)
assert output.shape == rgb.shape
assert abs(adaptive.last_diagnostics['sigma_mean'].item() - 25.0) < 1e-6
assert adaptive.last_diagnostics['gaussian_diff_max'].item() < 1e-6
adaptive.train(); adaptive.zero_grad(); adaptive(rgb, g25).square().mean().backward()
assert adaptive.sigma_predictor[-1].weight.grad is not None
assert adaptive.sigma_predictor[-1].weight.grad.abs().sum() > 0
print('HiSym Gaussian extension functional checks passed')
