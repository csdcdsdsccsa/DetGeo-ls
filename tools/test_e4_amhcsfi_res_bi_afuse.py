"""Focused construction and gradient checks for the Bi-Res AFuse detector."""
import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def main():
    torch.manual_seed(2024)
    model = TROGeoMSDetectionAblation(
        variant='h2_ind_amhcsfi_res_bi_afuse', position_mode='detgeo').train()
    if hasattr(model, 'det_head_stage3') or hasattr(model, 'det_head_stage4'):
        raise RuntimeError('Bi-Res AFuse must have exactly one final detection head')
    if not hasattr(model, 'det_head_single') or not hasattr(model, 'stage34_adaptive_fusion'):
        raise RuntimeError('Bi-Res AFuse single head or fusion module is missing')

    z3 = torch.randn(2, 384, 64, 64)
    aligned4 = torch.randn(2, 384, 64, 64)
    fused, weights = model.stage34_adaptive_fusion(z3, aligned4)
    prediction = model.det_head_single(fused)
    if tuple(prediction.shape) != (2, 45, 64, 64):
        raise RuntimeError('invalid AFuse prediction shape: {}'.format(tuple(prediction.shape)))
    if tuple(weights.shape) != (2, 2) or not torch.allclose(weights, torch.full_like(weights, 0.5)):
        raise RuntimeError('AFuse must start with exact equal Stage-3/4 weights')
    loss = prediction.square().mean()
    loss.backward()
    if model.stage34_adaptive_fusion.weight_predictor[-1].weight.grad is None:
        raise RuntimeError('AFuse predictor has no gradient')
    if model.det_head_single.weight.grad is None:
        raise RuntimeError('single detector head has no gradient')
    print('Bi-Res AFuse sanity passed: single_head=45ch initial_weights=0.5/0.5 gradients=nonzero', flush=True)


if __name__ == '__main__':
    main()
