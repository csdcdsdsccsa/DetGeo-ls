"""CPU sanity checks for Bi-Res Quality-Calibrated Competition (QCC)."""
import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from train import qcc_pairwise_ranking_loss, qcc_quality_loss
from utils.multiscale_detection import (build_qcc_state, select_qcc_a, select_qcc_ranked,
                                        select_two_heads)


def manual_state(pair_iou):
    return {
        'box3': torch.tensor([[10., 10., 30., 30.]]),
        'box4': torch.tensor([[12., 12., 32., 32.]]),
        'score3': torch.tensor([0.7]), 'score4': torch.tensor([0.6]),
        'quality3': torch.tensor([0.5]), 'quality4': torch.tensor([0.5]),
        'pair_iou': torch.tensor([pair_iou]),
    }


def main():
    torch.manual_seed(2024)
    model_a = TROGeoMSDetectionAblation(
        variant='h2_ind_amhcsfi_res_bi_qcc_a', position_mode='detgeo').train()
    torch.manual_seed(2024)
    model_b = TROGeoMSDetectionAblation(
        variant='h2_ind_amhcsfi_res_bi_qcc_b', position_mode='detgeo').train()
    for key in ('qcc_quality_head_stage3.weight', 'qcc_quality_head_stage4.weight',
                'qcc_ranker.mlp.0.weight', 'qcc_ranker.mlp.5.weight'):
        if not torch.equal(model_a.state_dict()[key], model_b.state_dict()[key]):
            raise RuntimeError('QCC-A/B parameter layout or initialization differs at {}'.format(key))
    if not torch.equal(model_a.qcc_quality_head_stage3.weight, torch.zeros_like(model_a.qcc_quality_head_stage3.weight)):
        raise RuntimeError('QCC quality heads must be zero initialized')

    # At Q=P=0.5, both competition decoders must choose the original confidence winner.
    low_state = manual_state(0.4)
    a_box, _ = select_qcc_a(low_state)
    b_box, _ = select_qcc_ranked(low_state, torch.zeros(1))
    full_box, full_diag = select_qcc_ranked(low_state, torch.zeros(1), fusion_iou=0.5)
    if not (torch.equal(a_box, low_state['box3']) and torch.equal(b_box, low_state['box3']) and
            torch.equal(full_box, low_state['box3']) and full_diag['qcc_fusion_ratio'].item() == 0.0):
        raise RuntimeError('neutral QCC competition/fallback must reproduce the confidence winner')
    high_state = manual_state(0.6)
    fused_box, fused_diag = select_qcc_ranked(high_state, torch.zeros(1), fusion_iou=0.5)
    expected = (0.7 * high_state['box3'] + 0.6 * high_state['box4']) / 1.3
    if not torch.allclose(fused_box, expected) or fused_diag['qcc_fusion_ratio'].item() != 1.0:
        raise RuntimeError('QCC-Full agreement fusion is incorrect')

    features3 = torch.randn(2, 384, 64, 64)
    features4 = torch.randn(2, 384, 64, 64)
    q3 = model_b.qcc_quality_head_stage3(features3)
    q4 = model_b.qcc_quality_head_stage4(features4)
    if not (torch.allclose(torch.sigmoid(q3), torch.full_like(q3, 0.5)) and
            torch.allclose(torch.sigmoid(q4), torch.full_like(q4, 0.5))):
        raise RuntimeError('zero QCC logits must imply quality=0.5')
    p3 = torch.randn(2, 9, 5, 64, 64)
    p4 = torch.randn(2, 9, 5, 64, 64)
    anchors = torch.tensor([[37., 41.]] * 9)
    state = build_qcc_state(p3, p4, q3, q4, anchors, 1024)
    rank_logit = model_b.qcc_ranker(state['rank_input'].detach())
    q_loss, _, _ = qcc_quality_loss(state, torch.tensor([[300., 300., 500., 500.], [200., 200., 600., 600.]]))
    rank_loss, _ = qcc_pairwise_ranking_loss(rank_logit, torch.tensor([0.8, 0.1]), torch.tensor([0.1, 0.8]), 0.03)
    (q_loss + rank_loss).backward()
    if model_b.qcc_quality_head_stage3.weight.grad is None or model_b.qcc_ranker.mlp[-1].weight.grad is None:
        raise RuntimeError('QCC quality head or ranker has no gradient')
    if model_a.qcc_ranker.mlp[-1].weight.grad is not None:
        raise RuntimeError('QCC-A must not consume ranker gradients')
    print('QCC sanity passed: neutral-decoder=baseline quality=0.5 fusion-gate=correct gradients=isolated', flush=True)


if __name__ == '__main__':
    main()
