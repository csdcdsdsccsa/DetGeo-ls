"""Topology and output-shape checks for strict Bi-Res output ablations."""
import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


VARIANTS = (
    ('h2_ind_amhcsfi_res_bi_s3only', 'z3', 384, True),
    ('h2_ind_amhcsfi_res_bi_s4only', 'aligned_z4', 384, True),
    ('h2_ind_amhcsfi_res_bi_concat', 'cat(z3, aligned_z4)', 768, True),
    ('h2_ind_bi_nocsfi_concat', 'cat(z3, aligned_z4)', 768, False),
)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    torch.manual_seed(2024)
    base = TROGeoMSDetectionAblation(variant='h2_ind_amhcsfi_res_bi', position_mode='detgeo')
    print('Bi-Res dual-head trainable_parameters={}'.format(parameter_count(base)))
    for variant, output_name, in_channels, use_amhcsfi in VARIANTS:
        torch.manual_seed(2024)
        model = TROGeoMSDetectionAblation(variant=variant, position_mode='detgeo').train()
        required = ('cvopm_stage3', 'cvopm_stage4', 'coarse_guidance', 'fine_guidance',
                    'det_head_single')
        absent = ('stage34_adaptive_fusion', 'det_head_stage3', 'det_head_stage4')
        if any(not hasattr(model, name) for name in required) or any(hasattr(model, name) for name in absent):
            raise RuntimeError('{} does not preserve strict Bi-Res topology'.format(variant))
        has_refiner = hasattr(model, 'cross_scale_interaction') and hasattr(model, 'amhcsfi_res_refiner')
        if has_refiner != use_amhcsfi:
            raise RuntimeError('{} AMHCSFI-Res topology mismatch'.format(variant))
        if not use_amhcsfi and (variant in model.CSFI_VARIANTS or variant in model.AMHCSFI_RES_VARIANTS):
            raise RuntimeError('{} must not enter a CSFI family'.format(variant))
        if variant not in model.BIDIR_GUIDE_VARIANTS or variant not in model.BIRES_OUTPUT_CONCAT_VARIANTS and variant.endswith('concat'):
            raise RuntimeError('{} must retain bidirectional guidance and concat output'.format(variant))
        if model.det_head_single.in_channels != in_channels or model.det_head_single.out_channels != 45:
            raise RuntimeError('{} has incorrect detector dimensions'.format(variant))
        z3, z4 = torch.randn(2, 384, 64, 64), torch.randn(2, 768, 32, 32)
        if variant.endswith('s3only'):
            prediction = model.det_head_single(z3)
        else:
            if not hasattr(model, 'stage4_align'):
                raise RuntimeError('{} must construct stage4_align'.format(variant))
            aligned4 = model.stage4_align(z4)
            feature = aligned4 if variant.endswith('s4only') else torch.cat((z3, aligned4), dim=1)
            prediction = model.det_head_single(feature)
        if tuple(prediction.shape) != (2, 45, 64, 64):
            raise RuntimeError('{} has invalid prediction shape {}'.format(variant, tuple(prediction.shape)))
        print('{} sanity passed: BiRes=yes AMHCSFIRes={} CSFI={} AFuse=no output={} head={}->45 '
              'dual_head=no trainable_parameters={}'.format(
                  variant, 'yes' if use_amhcsfi else 'no', 'yes' if use_amhcsfi else 'no',
                  output_name, in_channels, parameter_count(model)))


if __name__ == '__main__':
    main()
