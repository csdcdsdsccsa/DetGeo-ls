import random

import numpy as np
import torch

from model.DetGeo import DetGeo
from model.DetGeo_prompt_interaction import DetGeoPromptInteraction


def seed_global_rng(seed):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)


def make(sam, rgbp):
    seed_global_rng(13)
    model = DetGeoPromptInteraction(use_sam_refinement=sam, use_rgbp_interaction=rgbp,
                                    preserve_downstream_rng=True)
    return model, torch.get_rng_state().clone()


seed_global_rng(13)
model_a = DetGeo()
rng_a = torch.get_rng_state().clone()
model_b, rng_b = make(True, False)
model_c, rng_c = make(False, True)
model_d, rng_d = make(True, True)
for label, rng in (('B', rng_b), ('C', rng_c), ('D', rng_d)):
    assert torch.equal(rng_a, rng), 'post-DetGeo RNG differs for {}'.format(label)
for key, value in model_b.sam_refiner.state_dict().items():
    assert torch.equal(value, model_d.sam_refiner.state_dict()[key]), 'SAM init mismatch: {}'.format(key)
for key, value in model_c.rgbp_interaction.state_dict().items():
    assert torch.equal(value, model_d.rgbp_interaction.state_dict()[key]), 'RGBP init mismatch: {}'.format(key)
assert torch.equal(model_c.rgbp_interaction.interaction_scale.detach(), torch.zeros_like(model_c.rgbp_interaction.interaction_scale.detach()))
print('A == B: True\nA == C: True\nA == D: True')
print('P10_RNG_ISOLATION_OK\nSAM_INIT_MATCH_B_D_OK\nRGBP_INIT_MATCH_C_D_OK\nRGBP_ZERO_RESIDUAL_OK')
