"""Construction and parameter-free correlation checks for B0."""
import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation, parameter_free_cosine_correlation


def main():
    torch.manual_seed(2024)
    model = TROGeoMSDetectionAblation(variant='h2_ind_corr_afuse_b0', position_mode='detgeo').train()
    required = ('stage34_adaptive_fusion', 'det_head_single')
    absent = ('cvopm_stage3', 'cvopm_stage4', 'coarse_guidance', 'fine_guidance',
              'cross_scale_interaction', 'amhcsfi_res_refiner', 'det_head_stage3', 'det_head_stage4')
    if any(not hasattr(model, name) for name in required) or any(hasattr(model, name) for name in absent):
        raise RuntimeError('B0 topology is not strict parameter-free Corr + AFuse')
    q3, r3 = torch.randn(2, 384, 16, 16), torch.randn(2, 384, 64, 64)
    q4, r4 = torch.randn(2, 768, 8, 8), torch.randn(2, 768, 32, 32)
    z3, gate3 = parameter_free_cosine_correlation(q3, r3)
    z4, gate4 = parameter_free_cosine_correlation(q4, r4)
    if tuple(z3.shape) != (2, 384, 64, 64) or tuple(gate3.shape) != (2, 1, 64, 64):
        raise RuntimeError('invalid B0 Stage3 Corr shape')
    if tuple(z4.shape) != (2, 768, 32, 32) or tuple(gate4.shape) != (2, 1, 32, 32):
        raise RuntimeError('invalid B0 Stage4 Corr shape')
    if gate3.min() < 0 or gate3.max() > 1 or gate4.min() < 0 or gate4.max() > 1:
        raise RuntimeError('B0 Corr gates must be in [0, 1]')
    fused, weights = model.stage34_adaptive_fusion(z3, model.stage4_align(z4))
    if tuple(model.det_head_single(fused).shape) != (2, 45, 64, 64) or not torch.allclose(weights, torch.full_like(weights, 0.5)):
        raise RuntimeError('invalid B0 AFuse/single-head initialization')
    print('B0 sanity passed: DetGeoPE=yes DirectCA=no BiGuidance=no Corr=yes(parameter-free) CSFI=no AMHCSFIRes=no AFuse=yes SingleHead=yes RNG=natural')


if __name__ == '__main__':
    main()
