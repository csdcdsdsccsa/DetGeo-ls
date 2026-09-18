"""Strict topology and numerical checks for the Corr2S bridge ablation."""
import torch
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation, parameter_free_cosine_correlation


def main():
    torch.manual_seed(2024)
    model = TROGeoMSDetectionAblation(variant='h2_ind_corr2s', position_mode='current').train()
    required = ('position_embedding', 'det_head_stage3', 'stage4_align', 'det_head_stage4')
    absent = ('cvopm_stage3', 'cvopm_stage4', 'stage34_adaptive_fusion', 'det_head_single',
              'coarse_guidance', 'fine_guidance', 'cross_scale_interaction', 'amhcsfi_res_refiner')
    if any(not hasattr(model, n) for n in required) or any(hasattr(model, n) for n in absent):
        raise RuntimeError('Corr2S topology is not strict two-scale DetGeo')
    q3, r3 = torch.randn(2, 384, 16, 16), torch.randn(2, 384, 64, 64)
    q4, r4 = torch.randn(2, 768, 8, 8), torch.randn(2, 768, 32, 32)
    z3, g3 = parameter_free_cosine_correlation(q3, r3)
    z4, g4 = parameter_free_cosine_correlation(q4, r4)
    if tuple(g3.shape) != (2, 1, 64, 64) or tuple(g4.shape) != (2, 1, 32, 32): raise RuntimeError('invalid gate shapes')
    if g3.min() < 0 or g3.max() > 1 or g4.min() < 0 or g4.max() > 1: raise RuntimeError('gate outside [0,1]')
    if not torch.allclose(z3, r3 * (1 + g3)) or not torch.allclose(z4, r4 * (1 + g4)): raise RuntimeError('not residual correlation')
    p3, p4 = model.det_head_stage3(z3), model.det_head_stage4(model.stage4_align(z4))
    if tuple(p3.shape) != (2, 45, 64, 64) or tuple(p4.shape) != (2, 45, 64, 64): raise RuntimeError('invalid two-head shapes')
    (p3.mean() + p4.mean()).backward()
    print('Corr2S sanity passed: CurrentPE=yes DirectCA=no BiGuidance=no CSFI=no AFuse=no B0ResidualQS=yes TwoHeads=yes')


if __name__ == '__main__': main()
