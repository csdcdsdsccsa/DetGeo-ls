import torch

from model.deep_gaussian_residual_pe import DeepGaussianResidualPE
from model.hisym_pae_fusion import HiSymPAEFusion


torch.manual_seed(2024)
rgb, position = torch.randn(2, 3, 256, 256), torch.randn(2, 1, 256, 256)
assert HiSymPAEFusion()(rgb, position).shape == rgb.shape
dgrpe = DeepGaussianResidualPE()
output = dgrpe(rgb, position)
assert torch.equal(output, rgb)
assert dgrpe.last_diagnostics['delta_abs_mean'] > 0
output.square().mean().backward()
assert dgrpe.gamma.grad is not None and dgrpe.gamma.grad.abs() > 0
print('frontend position encoder functional checks passed')
