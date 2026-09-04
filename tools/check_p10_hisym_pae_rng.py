import random

import numpy as np
import torch

from model.DetGeo import DetGeo
from model.DetGeo_hisym_pae import DetGeoHiSymPAE
from model.DetGeo_prompt_interaction import DetGeoPromptInteraction


def seed_global_rng(seed):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)


seed_global_rng(13)
model_a = DetGeo()
rng_a = torch.get_rng_state().clone()
seed_global_rng(13)
model_b = DetGeoPromptInteraction(use_sam_refinement=True, use_rgbp_interaction=False,
                                  preserve_downstream_rng=True)
seed_global_rng(13)
model_c = DetGeoHiSymPAE(use_sam_refinement=False, preserve_downstream_rng=True)
rng_c = torch.get_rng_state().clone()
seed_global_rng(13)
model_d = DetGeoHiSymPAE(use_sam_refinement=True, preserve_downstream_rng=True)
rng_d = torch.get_rng_state().clone()
assert torch.equal(rng_a, rng_c) and torch.equal(rng_a, rng_d)
for key, value in model_b.sam_refiner.state_dict().items():
    assert torch.equal(value, model_d.sam_refiner.state_dict()[key]), 'SAM init mismatch: {}'.format(key)
for key, value in model_c.pae_fusion.state_dict().items():
    assert torch.equal(value, model_d.pae_fusion.state_dict()[key]), 'PAE init mismatch: {}'.format(key)
for model in (model_c, model_d):
    assert not hasattr(model, 'combine_clickptns_conv')
    assert not hasattr(model, 'rgbp_interaction')
print('P10_RNG_A_C_D_OK')
print('SAM_INIT_B_D_OK')
print('PAE_INIT_C_D_OK')
print('HISYM_FORWARD_PATH_OK')
