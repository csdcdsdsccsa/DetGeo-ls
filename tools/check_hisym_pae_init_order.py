import random

import numpy as np
import torch

from model.DetGeo import DetGeo
from model.DetGeo_hisym_pae import DetGeoHiSymPAE
from model.DetGeo_hisym_pae_inline import DetGeoHiSymPAEInline


def seed(seed=13):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)


def equal_module(left, right):
    left_state, right_state = left.state_dict(), right.state_dict()
    return left_state.keys() == right_state.keys() and all(torch.equal(left_state[key], right_state[key]) for key in left_state)


seed(); model_a = DetGeo()
seed(); model_old = DetGeoHiSymPAE(use_sam_refinement=False, preserve_downstream_rng=False)
seed(); model_inline = DetGeoHiSymPAEInline()
for name in ('query_mapping_visu', 'reference_mapping_visu', 'fcn_out'):
    assert equal_module(getattr(model_a, name), getattr(model_old, name)), 'A/C-old mismatch: ' + name
for name in ('query_resnet', 'reference_darknet'):
    assert equal_module(getattr(model_a, name), getattr(model_inline, name)), 'A/C-inline mismatch: ' + name
assert any(not torch.equal(left, right) for left, right in zip(model_a.query_mapping_visu.state_dict().values(), model_inline.query_mapping_visu.state_dict().values()))
assert any(not torch.equal(left, right) for left, right in zip(model_a.reference_mapping_visu.state_dict().values(), model_inline.reference_mapping_visu.state_dict().values()))
assert any(not torch.equal(left, right) for left, right in zip(model_a.fcn_out.state_dict().values(), model_inline.fcn_out.state_dict().values()))
print('C_OLD_POSTHOC_INIT_MATCHES_A_OK')
print('C_INLINE_BACKBONE_MATCHES_A_OK')
print('C_INLINE_LATER_MODULES_DIFFER_OK')
