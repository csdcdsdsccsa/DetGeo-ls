"""Structural test for the restricted Full Model ResNet-50 backbone ablation."""

import torch

from model.TROGeo_ms_direct_ca_sh import ResNet50MultiStageEncoder
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def main():
    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone='resnet50', variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe', dadpe_mode='none', amr_pe_mode='none',
        gaussian_sigma=25.0, crgpe_outer_sigma=50.0)
    assert model.backbone_name == 'resnet50'
    assert isinstance(model.encoder, ResNet50MultiStageEncoder)
    assert not hasattr(model, 'query_encoder') and not hasattr(model, 'reference_encoder')
    assert model.encoder.stage3_projection.in_channels == 1024
    assert model.encoder.stage3_projection.out_channels == 384
    assert model.encoder.stage4_projection.in_channels == 2048
    assert model.encoder.stage4_projection.out_channels == 768
    assert all(hasattr(model, name) for name in (
        'cvopm_stage3', 'cvopm_stage4', 'coarse_guidance', 'fine_guidance',
        'cross_scale_interaction', 'amhcsfi_res_refiner',
        'det_head_stage3', 'det_head_stage4'))
    model.encoder.eval()
    with torch.no_grad():
        stage3, stage4 = model.encoder(torch.randn(1, 3, 64, 128))
    assert tuple(stage3.shape) == (1, 384, 4, 8)
    assert tuple(stage4.shape) == (1, 768, 2, 4)
    print('Full Model ResNet-50 topology check passed: shared encoder, 1024->384 Stage3, 2048->768 Stage4.')


if __name__ == '__main__':
    main()
