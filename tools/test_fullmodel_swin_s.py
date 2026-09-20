"""Topology check for the strict Full Model Swin-S backbone ablation."""

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def main():
    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone='swin_s', variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe', dadpe_mode='none', amr_pe_mode='none',
        gaussian_sigma=25.0, crgpe_outer_sigma=50.0,
    )
    required = ('cvopm_stage3', 'cvopm_stage4', 'coarse_guidance', 'fine_guidance',
                'cross_scale_interaction', 'amhcsfi_res_refiner',
                'det_head_stage3', 'det_head_stage4')
    assert model.backbone_name == 'swin_s'
    assert model.variant == 'h2_ind_amhcsfi_res_bi'
    assert model.position_mode == 'hisym_crgpe'
    assert all(hasattr(model, name) for name in required)
    print('Full Model Swin-S topology check passed.')


if __name__ == '__main__':
    main()
