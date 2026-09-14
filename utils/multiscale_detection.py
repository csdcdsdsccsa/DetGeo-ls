"""Decoded-box evaluation for the controlled multi-scale detector ablations."""

import torch

from .utils import bbox_iou, xywh2xyxy

def decode_top1_with_index(prediction, anchors, image_wh):
    """Decode the highest-confidence candidate in one [B,A,5,G,G] head."""
    batch, _, _, grid, _ = prediction.shape
    flat = prediction[:, :, 4].reshape(batch, -1)
    best = flat.argmax(dim=1)
    cells = grid * grid
    anchor = best // cells
    cell = best % cells
    gj, gi = cell // grid, cell % grid
    selected = prediction[torch.arange(batch, device=prediction.device), anchor, :, gj, gi]
    stride = image_wh // grid
    scaled = anchors[anchor] / stride
    xywh = torch.cat(((selected[:, :2].sigmoid() + torch.stack((gi, gj), dim=1)) * stride,
                      torch.exp(selected[:, 2:4]) * scaled * stride), dim=1)
    box = xywh2xyxy(xywh)
    score = flat.softmax(dim=1).max(dim=1).values
    return box, score, best


def decode_top1(prediction, anchors, image_wh):
    """Decode the highest-confidence candidate while preserving the old API."""
    box, score, _ = decode_top1_with_index(prediction, anchors, image_wh)
    return box, score


def build_qcc_state(pred3, pred4, quality_logits3, quality_logits4, anchors, image_wh):
    """Anchor-aware, inference-safe state shared by QCC selection and losses."""
    box3, score3, index3 = decode_top1_with_index(pred3, anchors, image_wh)
    box4, score4, index4 = decode_top1_with_index(pred4, anchors, image_wh)
    batch = pred3.shape[0]
    q3 = torch.sigmoid(quality_logits3.reshape(batch, -1).gather(1, index3[:, None]).squeeze(1))
    q4 = torch.sigmoid(quality_logits4.reshape(batch, -1).gather(1, index4[:, None]).squeeze(1))
    pair_iou = bbox_iou(box3, box4, x1y1x2y2=True).clamp(0.0, 1.0)

    def descriptor(box):
        cx = 0.5 * (box[:, 0] + box[:, 2]) / float(image_wh)
        cy = 0.5 * (box[:, 1] + box[:, 3]) / float(image_wh)
        w = (box[:, 2] - box[:, 0]).clamp_min(0) / float(image_wh)
        h = (box[:, 3] - box[:, 1]).clamp_min(0) / float(image_wh)
        return torch.stack((cx, cy, w, h), dim=1)

    rank_input = torch.cat((score3[:, None], score4[:, None], q3[:, None], q4[:, None],
                            pair_iou[:, None], (descriptor(box3) - descriptor(box4)).abs()), dim=1)
    if rank_input.shape != (batch, 9):
        raise RuntimeError('QCC rank input must be [B,9], got {}'.format(tuple(rank_input.shape)))
    return {'box3': box3, 'box4': box4, 'score3': score3, 'score4': score4,
            'quality3': q3, 'quality4': q4, 'pair_iou': pair_iou, 'rank_input': rank_input}


def select_qcc_a(state):
    r3, r4 = state['score3'] * state['quality3'], state['score4'] * state['quality4']
    use3 = r3 >= r4
    return torch.where(use3[:, None], state['box3'], state['box4']), {
        'qcc_quality3': state['quality3'].mean(), 'qcc_quality4': state['quality4'].mean(),
        'qcc_reliability3': r3.mean(), 'qcc_reliability4': r4.mean(),
        'qcc_head3_ratio': use3.float().mean(), 'qcc_head4_ratio': (~use3).float().mean(),
        'qcc_pair_iou': state['pair_iou'].mean(), 'qcc_fusion_ratio': use3.new_zeros((), dtype=torch.float),
    }


def select_qcc_ranked(state, rank_logit, fusion_iou=None):
    p3 = torch.sigmoid(rank_logit)
    r3 = p3 * state['score3'] * state['quality3']
    r4 = (1.0 - p3) * state['score4'] * state['quality4']
    use3 = r3 >= r4
    winner = torch.where(use3[:, None], state['box3'], state['box4'])
    if fusion_iou is None:
        fusion_mask = use3.new_zeros(use3.shape, dtype=torch.bool)
        selected = winner
    else:
        fusion_mask = state['pair_iou'] >= float(fusion_iou)
        fused = (r3[:, None] * state['box3'] + r4[:, None] * state['box4']) / (r3 + r4).clamp_min(1e-12)[:, None]
        selected = torch.where(fusion_mask[:, None], fused, winner)
    return selected, {
        'qcc_quality3': state['quality3'].mean(), 'qcc_quality4': state['quality4'].mean(),
        'qcc_rank_probability3': p3.mean(), 'qcc_reliability3': r3.mean(), 'qcc_reliability4': r4.mean(),
        'qcc_head3_ratio': use3.float().mean(), 'qcc_head4_ratio': (~use3).float().mean(),
        'qcc_pair_iou': state['pair_iou'].mean(), 'qcc_fusion_ratio': fusion_mask.float().mean(),
    }


def _decode_flat(prediction, anchors, flat_index, image_wh):
    """Decode caller-selected flattened candidates, one per batch element."""
    batch, _, _, grid, _ = prediction.shape
    cells = grid * grid
    anchor = flat_index // cells
    cell = flat_index % cells
    gj, gi = cell // grid, cell % grid
    selected = prediction[torch.arange(batch, device=prediction.device), anchor, :, gj, gi]
    stride = image_wh // grid
    scaled = anchors[anchor] / stride
    xywh = torch.cat(((selected[:, :2].sigmoid() + torch.stack((gi, gj), dim=1)) * stride,
                      torch.exp(selected[:, 2:4]) * scaled * stride), dim=1)
    return xywh2xyxy(xywh)


def decode_multigrid_top1(pred3, pred4, anchors, image_wh):
    """E2: select once over all stage-3 and stage-4 candidates."""
    batch = pred3.shape[0]
    flat3 = pred3[:, :, 4].reshape(batch, -1)
    flat4 = pred4[:, :, 4].reshape(batch, -1)
    combined = torch.cat((flat3, flat4), dim=1)
    best = combined.argmax(dim=1)
    cut = flat3.shape[1]
    use3 = best < cut
    local3 = best.clamp(max=cut - 1)
    local4 = (best - cut).clamp(min=0)
    box3 = _decode_flat(pred3, anchors[3:], local3, image_wh)
    box4 = _decode_flat(pred4, anchors[:3], local4, image_wh)
    return torch.where(use3[:, None], box3, box4)


def select_two_heads(pred3, pred4, anchors, image_wh, fusion=False, iou_threshold=0.5,
                     adaptive=False):
    """Choose H2 or fuse H3 boxes using a fixed or confidence-adaptive IoU gate."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError('iou_threshold must be in [0, 1]')
    box3, score3 = decode_top1(pred3, anchors, image_wh)
    box4, score4 = decode_top1(pred4, anchors, image_wh)
    use3 = score3 >= score4
    selected = torch.where(use3[:, None], box3, box4)
    pair_iou = bbox_iou(box3, box4, x1y1x2y2=True)
    if adaptive:
        balance = 2.0 * torch.minimum(score3, score4) / (score3 + score4 + 1e-12)
        threshold = 0.7 - 0.3 * balance
    else:
        balance = torch.zeros_like(pair_iou)
        threshold = torch.full_like(pair_iou, float(iou_threshold))
    fusion_mask = pair_iou.ge(threshold)
    diagnostics = {
        'stage3_selected': use3.float().mean(),
        'stage4_selected': (~use3).float().mean(),
        'pair_iou': pair_iou.mean(),
        'fusion_ratio': fusion_mask.float().mean(),
        'mean_threshold': threshold.mean(),
        'mean_balance': balance.mean(),
        'min_threshold': threshold.min(),
        'max_threshold': threshold.max(),
        'threshold_lt_0p5_ratio': threshold.lt(0.5).float().mean(),
        'threshold_ge_0p5_ratio': threshold.ge(0.5).float().mean(),
    }
    if fusion:
        weight3 = score3 / (score3 + score4 + 1e-12)
        fused = weight3[:, None] * box3 + (1.0 - weight3)[:, None] * box4
        selected = torch.where(fusion_mask[:, None], fused, selected)
    return selected, diagnostics


def select_two_heads_hqs(pred3, pred4, hqs_logits, anchors, image_wh):
    """Select one complete detection head with learned HQS probabilities."""
    box3, _ = decode_top1(pred3, anchors, image_wh)
    box4, _ = decode_top1(pred4, anchors, image_wh)
    quality = torch.softmax(hqs_logits, dim=1)
    use3 = quality[:, 0] >= quality[:, 1]
    return torch.where(use3[:, None], box3, box4), {
        'hqs_stage3_selected': use3.float().mean(),
        'hqs_stage4_selected': (~use3).float().mean(),
        'hqs_quality_margin': (quality[:, 0] - quality[:, 1]).abs().mean(),
    }


def select_two_heads_hqs_v2a(pred3, pred4, rank_logit, anchors, image_wh, threshold=0.0):
    """HQS-v2a pairwise ranking: rank >= threshold selects Head3.

    ``threshold=0.0`` is the original HQS-v2a decoder.  The optional
    threshold is inference-only and never changes the ranker or its loss.
    """
    box3, _ = decode_top1(pred3, anchors, image_wh)
    box4, _ = decode_top1(pred4, anchors, image_wh)
    use3 = rank_logit >= float(threshold)
    return torch.where(use3[:, None], box3, box4), {
        'hqs_v2a_stage3_selected': use3.float().mean(),
        'hqs_v2a_stage4_selected': (~use3).float().mean(),
        'hqs_v2a_abs_rank_logit': rank_logit.abs().mean(),
        'hqs_v2a_rank_logit_mean': rank_logit.mean(),
        'hqs_v2a_threshold': rank_logit.new_tensor(float(threshold)),
    }


@torch.no_grad()
def analyze_two_head_oracle(pred3, pred4, target_bbox, anchors, image_wh):
    """Validation-only GT-oracle upper bound for two-head selection.

    This diagnostic must never be used by the inference decoder: it chooses a
    head with the ground-truth IoU solely to quantify head complementarity.
    """
    box3, score3 = decode_top1(pred3, anchors, image_wh)
    box4, score4 = decode_top1(pred4, anchors, image_wh)
    iou3 = bbox_iou(box3, target_bbox, x1y1x2y2=True)
    iou4 = bbox_iou(box4, target_bbox, x1y1x2y2=True)
    confidence_use3 = score3 >= score4
    oracle_use3 = iou3 >= iou4
    confidence_box = torch.where(confidence_use3[:, None], box3, box4)
    oracle_box = torch.where(oracle_use3[:, None], box3, box4)
    confidence_iou = torch.where(confidence_use3, iou3, iou4)
    return {
        'box3': box3, 'box4': box4,
        'confidence_box': confidence_box, 'oracle_box': oracle_box,
        'score3': score3, 'score4': score4, 'iou3': iou3, 'iou4': iou4,
        'confidence_iou': confidence_iou, 'oracle_iou': torch.maximum(iou3, iou4),
        'confidence_use3': confidence_use3, 'oracle_use3': oracle_use3,
        'agreement': confidence_use3.eq(oracle_use3).float(),
    }


def select_three_heads(pred2, pred3, pred4, anchors, image_wh):
    """E7: extend E4 winner-takes-all decoding from two heads to three heads."""
    box2, score2 = decode_top1(pred2, anchors, image_wh)
    box3, score3 = decode_top1(pred3, anchors, image_wh)
    box4, score4 = decode_top1(pred4, anchors, image_wh)
    scores = torch.stack((score2, score3, score4), dim=1)
    best_head = scores.argmax(dim=1)
    boxes = torch.stack((box2, box3, box4), dim=1)
    selected = boxes[torch.arange(boxes.shape[0], device=boxes.device), best_head]
    diagnostics = {
        'stage2_selected': best_head.eq(0).float().mean(),
        'stage3_selected': best_head.eq(1).float().mean(),
        'stage4_selected': best_head.eq(2).float().mean(),
    }
    return selected, diagnostics


def eval_decoded_boxes(pred_bbox, target_bbox, image_wh):
    iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
    pred_center = (pred_bbox[:, :2] + pred_bbox[:, 2:4]) * 0.5
    target_center = (target_bbox[:, :2] + target_bbox[:, 2:4]) * 0.5
    pred_grid = (pred_center / (image_wh // 64)).long()
    target_grid = (target_center / (image_wh // 64)).long()
    return iou.gt(0.5).float().mean(), iou.gt(0.25).float().mean(), iou.mean(), \
        ((pred_grid == target_grid).all(dim=1)).float().mean()
