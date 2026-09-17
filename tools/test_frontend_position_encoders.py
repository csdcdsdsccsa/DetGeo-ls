import torch

from model.deep_gaussian_residual_pe import DeepGaussianResidualPE
from model.hisym_pae_fusion import HiSymPAEFusion
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


torch.manual_seed(2024)
rgb, position = torch.randn(2, 3, 256, 256), torch.randn(2, 1, 256, 256)
assert HiSymPAEFusion()(rgb, position).shape == rgb.shape
dgrpe = DeepGaussianResidualPE()
output = dgrpe(rgb, position)
assert torch.equal(output, rgb)
assert dgrpe.last_diagnostics['delta_abs_mean'] > 0
assert dgrpe.last_diagnostics['identity_error'] < 1e-7
output.square().mean().backward()
assert dgrpe.gamma.grad is not None and dgrpe.gamma.grad.abs() > 0

states = []
for mode in ('detgeo', 'hisym_pe', 'dgrpe'):
    torch.manual_seed(2024)
    model = TROGeoMSDetectionAblation(variant='h2_ind_amhcsfi_res_bi', position_mode=mode)
    states.append({k: v for k, v in model.state_dict().items() if not k.startswith('position_embedding.')})
assert states[0].keys() == states[1].keys() == states[2].keys()
for key in states[0]:
    assert torch.equal(states[0][key], states[1][key]) and torch.equal(states[0][key], states[2][key]), key
print('frontend position-encoder checks passed')
