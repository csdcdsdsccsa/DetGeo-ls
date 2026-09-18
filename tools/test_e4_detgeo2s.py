"""Topology and numerical smoke checks for the two-scale DetGeo baseline."""
import torch

from model.TROGeo_ms_detection_ablation import (
    TROGeoMSDetectionAblation, detgeo_spatial_fusion,
)


def main():
    torch.manual_seed(2024)
    model = TROGeoMSDetectionAblation(variant='h2_ind_detgeo2s', position_mode='current').train()
    required = ('det_head_stage3', 'stage4_align', 'det_head_stage4')
    absent = ('cvopm_stage3', 'cvopm_stage4', 'coarse_guidance', 'fine_guidance',
              'cross_scale_interaction', 'amhcsfi_res_refiner')
    if any(not hasattr(model, name) for name in required) or any(hasattr(model, name) for name in absent):
        raise RuntimeError('DetGeo2S topology is not a strict no-guidance two-head baseline')
    q3, r3 = torch.randn(2, 384, 16, 16), torch.randn(2, 384, 64, 64)
    q4, r4 = torch.randn(2, 768, 8, 8), torch.randn(2, 768, 32, 32)
    z3, attn3 = detgeo_spatial_fusion(q3, r3)
    z4, attn4 = detgeo_spatial_fusion(q4, r4)
    if tuple(z3.shape) != (2, 384, 64, 64) or tuple(attn3.shape) != (2, 64, 64):
        raise RuntimeError('invalid DetGeo2S Stage3 fusion shape')
    if tuple(z4.shape) != (2, 768, 32, 32) or tuple(attn4.shape) != (2, 32, 32):
        raise RuntimeError('invalid DetGeo2S Stage4 fusion shape')
    if min(attn3.min(), attn4.min()) < 0 or max(attn3.max(), attn4.max()) > 1:
        raise RuntimeError('DetGeo2S attention must be in [0, 1]')
    p3, p4 = model.det_head_stage3(z3), model.det_head_stage4(model.stage4_align(z4))
    if tuple(p3.shape) != (2, 45, 64, 64) or tuple(p4.shape) != (2, 45, 64, 64):
        raise RuntimeError('DetGeo2S two-head shapes are invalid')
    (p3.mean() + p4.mean()).backward()
    print('DetGeo2S sanity passed: DirectCA=no BiGuidance=no CSFI=no AuxLoss=no TwoHeads=yes')


if __name__ == '__main__':
    main()
