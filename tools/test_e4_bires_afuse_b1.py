"""Construction and AFuse checks for B1: Bi-Guidance without CSFI."""
import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def main():
    torch.manual_seed(2024)
    model = TROGeoMSDetectionAblation(variant='h2_ind_bires_afuse_b1', position_mode='detgeo').train()
    required = ('cvopm_stage3', 'cvopm_stage4', 'coarse_guidance', 'fine_guidance',
                'stage34_adaptive_fusion', 'det_head_single')
    absent = ('cross_scale_interaction', 'amhcsfi_res_refiner', 'det_head_stage3', 'det_head_stage4')
    if any(not hasattr(model, name) for name in required) or any(hasattr(model, name) for name in absent):
        raise RuntimeError('B1 topology is not strict Bi-Guidance + AFuse without CSFI')
    fused, weights = model.stage34_adaptive_fusion(torch.randn(2, 384, 64, 64), torch.randn(2, 384, 64, 64))
    prediction = model.det_head_single(fused)
    if tuple(prediction.shape) != (2, 45, 64, 64) or not torch.allclose(weights, torch.full_like(weights, 0.5)):
        raise RuntimeError('B1 AFuse/single-head initialization is invalid')
    print('B1 sanity passed: DetGeoPE=yes DirectCA=yes BiGuidance=yes CSFI=no AMHCSFIRes=no AFuse=yes SingleHead=yes RNG=natural')


if __name__ == '__main__':
    main()
