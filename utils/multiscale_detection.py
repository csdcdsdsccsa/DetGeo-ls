"""Decoded-box evaluation for the controlled multi-scale detector ablations."""

import torch

from .utils import bbox_iou, xywh2xyxy

H3_IOU_THRESHOLD = 0.5


def decode_top1(prediction, anchors, image_wh):
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
    return box, score


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


def select_two_heads(pred3, pred4, anchors, image_wh, fusion=False):
    box3, score3 = decode_top1(pred3, anchors, image_wh)
    box4, score4 = decode_top1(pred4, anchors, image_wh)
    use3 = score3 >= score4
    selected = torch.where(use3[:, None], box3, box4)
    pair_iou = bbox_iou(box3, box4, x1y1x2y2=True)
    diagnostics = {'stage3_selected': use3.float().mean(), 'stage4_selected': (~use3).float().mean(),
                   'pair_iou': pair_iou.mean(), 'fusion_ratio': pair_iou.ge(H3_IOU_THRESHOLD).float().mean()}
    if fusion:
        weight3 = score3 / (score3 + score4 + 1e-12)
        fused = weight3[:, None] * box3 + (1.0 - weight3)[:, None] * box4
        selected = torch.where(pair_iou.ge(H3_IOU_THRESHOLD)[:, None], fused, selected)
    return selected, diagnostics


def eval_decoded_boxes(pred_bbox, target_bbox, image_wh):
    iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
    pred_center = (pred_bbox[:, :2] + pred_bbox[:, 2:4]) * 0.5
    target_center = (target_bbox[:, :2] + target_bbox[:, 2:4]) * 0.5
    pred_grid = (pred_center / (image_wh // 64)).long()
    target_grid = (target_center / (image_wh // 64)).long()
    return iou.gt(0.5).float().mean(), iou.gt(0.25).float().mean(), iou.mean(), \
        ((pred_grid == target_grid).all(dim=1)).float().mean()
