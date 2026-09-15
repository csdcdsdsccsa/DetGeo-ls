"""CPU/GPU sanity checks for Bi-Res Quality-Calibrated Competition (QCC)."""
import argparse
import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from train import qcc_pairwise_ranking_loss, qcc_quality_loss
from utils.multiscale_detection import (build_qcc_state, select_qcc_a, select_qcc_af, select_qcc_ranked,
                                        select_two_heads)


def manual_state(pair_iou):
    return {
        'box3': torch.tensor([[10., 10., 30., 30.]]),
        'box4': torch.tensor([[12., 12., 32., 32.]]),
        'score3': torch.tensor([0.7]), 'score4': torch.tensor([0.6]),
        'quality3': torch.tensor([0.5]), 'quality4': torch.tensor([0.5]),
        'pair_iou': torch.tensor([pair_iou]),
    }


def check_gpu_forward(variant):
    if not torch.cuda.is_available() or not variant:
        return
    model = TROGeoMSDetectionAblation(variant=variant, position_mode='detgeo').cuda().eval()
    with torch.no_grad():
        predictions, _ = model(torch.randn(1, 3, 256, 256, device='cuda'),
                               torch.randn(1, 3, 1024, 1024, device='cuda'),
                               torch.rand(1, 256, 256, device='cuda'))
    expected = {'stage3', 'stage4', 'coarse_logits', 'fine_logits', 'qcc_quality3', 'qcc_quality4'}
    if set(predictions) != expected or predictions['qcc_quality3'].shape != (1, 9, 64, 64) or \
            predictions['qcc_quality4'].shape != (1, 9, 64, 64):
        raise RuntimeError('QCC forward output contract failed for {}: {}'.format(variant, sorted(predictions)))
    if not (torch.allclose(torch.sigmoid(predictions['qcc_quality3']),
                           torch.full_like(predictions['qcc_quality3'], 0.5)) and
            torch.allclose(torch.sigmoid(predictions['qcc_quality4']),
                           torch.full_like(predictions['qcc_quality4'], 0.5))):
        raise RuntimeError('QCC forward quality heads do not start at 0.5')
    del model, predictions
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_variant', choices=('h2_ind_amhcsfi_res_bi_qcc_a',
                                                   'h2_ind_amhcsfi_res_bi_qcc_af',
                                                   'h2_ind_amhcsfi_res_bi_qcc_b',
                                                   'h2_ind_amhcsfi_res_bi_qcc_full'))
    args = parser.parse_args()
    torch.manual_seed(2024)
    model_a = TROGeoMSDetectionAblation(
        variant='h2_ind_amhcsfi_res_bi_qcc_a', position_mode='detgeo').train()
    torch.manual_seed(2024)
    model_af = TROGeoMSDetectionAblation(
        variant='h2_ind_amhcsfi_res_bi_qcc_af', position_mode='detgeo').train()
    torch.manual_seed(2024)
    model_b = TROGeoMSDetectionAblation(
        variant='h2_ind_amhcsfi_res_bi_qcc_b', position_mode='detgeo').train()
    if set(model_a.state_dict()) != set(model_af.state_dict()):
        raise RuntimeError('QCC-A/AF parameter layouts differ')
    for key in ('qcc_quality_head_stage3.weight', 'qcc_quality_head_stage4.weight',
                'qcc_ranker.mlp.0.weight', 'qcc_ranker.mlp.5.weight'):
        if not torch.equal(model_a.state_dict()[key], model_af.state_dict()[key]) or \
                not torch.equal(model_a.state_dict()[key], model_b.state_dict()[key]):
            raise RuntimeError('QCC A/AF/B initialization differs at {}'.format(key))
    if not torch.equal(model_a.qcc_quality_head_stage3.weight, torch.zeros_like(model_a.qcc_quality_head_stage3.weight)):
        raise RuntimeError('QCC quality heads must be zero initialized')

    # At Q=P=0.5, both competition decoders must choose the original confidence winner.
    low_state = manual_state(0.4)
    a_box, _ = select_qcc_a(low_state)
    af_box, af_diag = select_qcc_af(low_state, fusion_iou=0.5)
    b_box, _ = select_qcc_ranked(low_state, torch.zeros(1))
    full_box, full_diag = select_qcc_ranked(low_state, torch.zeros(1), fusion_iou=0.5)
    if not (torch.equal(a_box, low_state['box3']) and torch.equal(af_box, a_box) and
            af_diag['qcc_fusion_ratio'].item() == 0.0 and torch.equal(b_box, low_state['box3']) and
            torch.equal(full_box, low_state['box3']) and full_diag['qcc_fusion_ratio'].item() == 0.0):
        raise RuntimeError('neutral QCC competition/fallback must reproduce the confidence winner')
    high_state = manual_state(0.6)
    af_fused, af_fused_diag = select_qcc_af(high_state, fusion_iou=0.5)
    fused_box, fused_diag = select_qcc_ranked(high_state, torch.zeros(1), fusion_iou=0.5)
    expected = (0.7 * high_state['box3'] + 0.6 * high_state['box4']) / 1.3
    if not torch.allclose(af_fused, expected) or af_fused_diag['qcc_fusion_ratio'].item() != 1.0 or \
            not torch.allclose(fused_box, expected) or fused_diag['qcc_fusion_ratio'].item() != 1.0:
        raise RuntimeError('QCC-Full agreement fusion is incorrect')

    features3 = torch.randn(2, 384, 64, 64)
    features4 = torch.randn(2, 384, 64, 64)
    q3 = model_af.qcc_quality_head_stage3(features3)
    q4 = model_af.qcc_quality_head_stage4(features4)
    if not (torch.allclose(torch.sigmoid(q3), torch.full_like(q3, 0.5)) and
            torch.allclose(torch.sigmoid(q4), torch.full_like(q4, 0.5))):
        raise RuntimeError('zero QCC logits must imply quality=0.5')
    p3 = torch.randn(2, 9, 5, 64, 64)
    p4 = torch.randn(2, 9, 5, 64, 64)
    anchors = torch.tensor([[37., 41.]] * 9)
    state = build_qcc_state(p3, p4, q3, q4, anchors, 1024)
    q_loss, _, _ = qcc_quality_loss(state, torch.tensor([[300., 300., 500., 500.], [200., 200., 600., 600.]]))
    q_loss.backward()
    if model_af.qcc_quality_head_stage3.weight.grad is None or \
            model_af.qcc_ranker.mlp[-1].weight.grad is not None:
        raise RuntimeError('QCC-AF must train quality heads without ranker gradients')
    q3_b = model_b.qcc_quality_head_stage3(features3)
    q4_b = model_b.qcc_quality_head_stage4(features4)
    state_b = build_qcc_state(p3, p4, q3_b, q4_b, anchors, 1024)
    rank_logit = model_b.qcc_ranker(state_b['rank_input'].detach())
    q_loss_b, _, _ = qcc_quality_loss(state_b, torch.tensor([[300., 300., 500., 500.], [200., 200., 600., 600.]]))
    rank_loss, _ = qcc_pairwise_ranking_loss(rank_logit, torch.tensor([0.8, 0.1]), torch.tensor([0.1, 0.8]), 0.03)
    (q_loss_b + rank_loss).backward()
    if model_b.qcc_quality_head_stage3.weight.grad is None or model_b.qcc_ranker.mlp[-1].weight.grad is None:
        raise RuntimeError('QCC quality head or ranker has no gradient')
    check_gpu_forward(args.gpu_variant)
    print('QCC sanity passed: A/AF disagreement identical AF fusion correct quality gradients active AF ranker gradient absent', flush=True)


if __name__ == '__main__':
    main()
