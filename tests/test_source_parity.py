"""Compare standalone parameters with the source Full Model at identical seed.

Pass SOURCE_REPO=/path/to/DetGeo-sam-prompt-yolo. Pretrained Swin weights are
suppressed only for this construction test; training keeps the original weights.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullmodel import FullModel


def main():
    source_repo = os.environ.get('SOURCE_REPO', str(ROOT.parent / 'DetGeo-sam-prompt-yolo'))
    sys.path.insert(1, source_repo)
    from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation

    original_swin_t = models.swin_t

    def no_download(*args, **kwargs):
        kwargs['weights'] = None
        return original_swin_t(*args, **kwargs)

    for name, sigmas in (
        ('DroneAerial', (25, None, 50, None)),
        ('SVI', (25, 50, 50, 100)),
    ):
        core_y, core_x, outer_y, outer_x = sigmas
        with patch.object(models, 'swin_t', side_effect=no_download):
            torch.manual_seed(2024)
            source = TROGeoMSDetectionAblation(
                emb_size=768, backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
                position_mode='hisym_crgpe', gaussian_sigma=core_y,
                gaussian_sigma_x=core_x, crgpe_outer_sigma=outer_y,
                crgpe_outer_sigma_x=outer_x)
            source_state = source.state_dict()
            torch.manual_seed(2024)
            standalone = FullModel(core_y, core_x, outer_y, outer_x)
        standalone_state = standalone.state_dict()
        assert source_state.keys() == standalone_state.keys(), (
            set(source_state) ^ set(standalone_state))
        for key in source_state:
            assert torch.equal(source_state[key], standalone_state[key]), key
        assert sum(p.numel() for p in standalone.parameters()) == 50286711
        print('{}: state_dict keys, tensors, and 50,286,711 parameters match'.format(name))


if __name__ == '__main__':
    main()
