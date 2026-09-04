import random
import numpy as np
import torch

from model.DetGeo import DetGeo
from model.DetGeo_sam_prompt import DetGeoSAMPrompt
from model.DetGeo_adaptive_sam import DetGeoAdaptiveSAM


def seed_all(seed=13):
    random.seed(seed); np.random.seed(seed + 1); torch.manual_seed(seed + 2); torch.cuda.manual_seed_all(seed + 3)


seed_all(); square = DetGeo(); square_state = torch.get_rng_state().clone()
seed_all(); sam = DetGeoSAMPrompt(preserve_downstream_rng=True); sam_state = torch.get_rng_state().clone()
seed_all(); adaptive = DetGeoAdaptiveSAM(preserve_downstream_rng=True); adaptive_state = torch.get_rng_state().clone()
assert torch.equal(square_state, sam_state)
assert torch.equal(square_state, adaptive_state)
for key, value in sam.prompt_fusion.state_dict().items():
    assert torch.equal(value, adaptive.adaptive_prompt.prompt_fusion.state_dict()[key]), key
click = torch.tensor([[128., 128.], [80., 160.]])
masks = torch.zeros(2, 3, 256, 256); masks[:, :, 80:180, 80:180] = 1
scores = torch.tensor([[.8, .9, .7], [.9, .6, .8]])
position, aux = adaptive.adaptive_prompt(click, masks, scores, return_aux=True)
g25 = adaptive.adaptive_prompt.make_gaussian(click, torch.full((2, 1), 25.), 256, 256, position.dtype, position.device)
assert torch.allclose(aux['sigma'], torch.full((2,), 25.), atol=1e-6)
assert torch.allclose(position, g25, atol=1e-7)
print('P10_RNG_ISOLATION_OK')
print('PROMPTFUSION_INIT_MATCH_OK')
print('ADAPTIVE_GAUSSIAN_INIT_OK')
