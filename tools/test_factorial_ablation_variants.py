"""Structural checks for the DroneAerial B/A/H full-factorial ablation."""

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def main():
    model_cls = TROGeoMSDetectionAblation
    detgeo_amhcsfi = 'h2_ind_detgeo2s_amhcsfi_res'

    assert detgeo_amhcsfi in model_cls.VALID_VARIANTS
    assert detgeo_amhcsfi in model_cls.DETGEO_TWO_SCALE_VARIANTS
    assert detgeo_amhcsfi in model_cls.AMHCSFI_RES_VARIANTS
    assert detgeo_amhcsfi in model_cls.CSFI_VARIANTS
    assert detgeo_amhcsfi not in model_cls.COARSE_GUIDE_VARIANTS
    assert detgeo_amhcsfi not in model_cls.FINE_GUIDE_VARIANTS
    assert detgeo_amhcsfi not in model_cls.FINE_ONLY_GUIDE_VARIANTS
    assert detgeo_amhcsfi not in model_cls.BIDIR_GUIDE_VARIANTS

    expected_crgpe = {
        'h2_ind_detgeo2s',
        'h2_ind_detgeo2s_amhcsfi_res',
        'h2_ind_bi_nocsfi',
        'h2_ind_amhcsfi_res_bi',
        'h2_ind_amhcsfi_res_bi_qcc_af',
    }
    assert set(model_cls.HISYM_CRGPE_SUPPORTED_VARIANTS) == expected_crgpe

    model = model_cls(variant=detgeo_amhcsfi, position_mode='current')
    required = ('det_head_stage3', 'stage4_align', 'det_head_stage4',
                'cross_scale_interaction', 'amhcsfi_res_refiner')
    forbidden = ('cvopm_stage3', 'cvopm_stage4', 'coarse_guidance', 'fine_guidance')
    assert all(hasattr(model, name) for name in required)
    assert not any(hasattr(model, name) for name in forbidden)

    print('Factorial topology checks passed: 010 is DetGeo matching + AMHCSFI-Res '
          'with independent two heads and no Direct-CA/Bi-Guidance; CRGPE support is scoped.')


if __name__ == '__main__':
    main()
