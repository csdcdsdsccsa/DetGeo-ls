"""CPU topology/parameter audit for the four pre-generalization ablations.

No pretrained Swin download and no data or checkpoints are needed here.
"""

import gc
from unittest.mock import patch

import torchvision.models as models

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


original_swin_t = models.swin_t


def without_download(*args, **kwargs):
    kwargs['weights'] = None
    return original_swin_t(*args, **kwargs)


cases = (
    ('baseline', 'bi', 'amhcsfi_res', True, True, True),
    ('direct_ca', 'direct', 'amhcsfi_res', False, False, True),
    ('coarse_only', 'coarse', 'amhcsfi_res', True, False, True),
    ('fine_only', 'fine', 'amhcsfi_res', False, True, True),
    ('original_csfi', 'bi', 'csfi', True, True, False),
)


def main():
    counts = {}
    with patch('model.TROGeo_ms_direct_ca_sh.models.swin_t', side_effect=without_download):
        for name, guidance, cross_scale, coarse, fine, refiner in cases:
            model = TROGeoMSDetectionAblation(
                emb_size=768, backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
                position_mode='hisym_crgpe', gaussian_sigma=25,
                guidance_mode=guidance, cross_scale_module=cross_scale)
            assert hasattr(model, 'coarse_guidance') == coarse, name
            assert hasattr(model, 'fine_guidance') == fine, name
            assert hasattr(model, 'amhcsfi_res_refiner') == refiner, name
            assert hasattr(model, 'cross_scale_interaction'), name
            assert model.encoder is not None and model.position_mode == 'hisym_crgpe'
            assert model.det_head_stage3 is not None and model.det_head_stage4 is not None
            counts[name] = sum(p.numel() for p in model.parameters())
            print('{}\t{}\t{}'.format(name, counts[name], counts[name] / 1e6), flush=True)
            del model
            gc.collect()
    for name in counts:
        if name != 'baseline':
            assert counts[name] < counts['baseline'], name


if __name__ == '__main__':
    main()
