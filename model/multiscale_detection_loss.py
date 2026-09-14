"""Losses for true multi-grid (E2) and two-head (E3/E4) detection ablations."""

import math

import torch
import torch.nn.functional as F

from .loss import build_target, yolo_loss


def build_coarse_heatmap(ori_gt_bboxes, image_wh, grid_size=32, sigma=1.5):
    """Gaussian target at the Stage4 coarse-localization resolution."""
    centers = (ori_gt_bboxes[:, :2] + ori_gt_bboxes[:, 2:]) * 0.5
    centers = centers / float(image_wh) * grid_size
    coordinates = torch.arange(grid_size, device=ori_gt_bboxes.device, dtype=ori_gt_bboxes.dtype)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    distance_squared = ((xx.unsqueeze(0) - centers[:, 0, None, None]) ** 2 +
                        (yy.unsqueeze(0) - centers[:, 1, None, None]) ** 2)
    return torch.exp(-distance_squared / (2.0 * sigma * sigma)).unsqueeze(1)


def coarse_heatmap_loss(coarse_logits, ori_gt_bboxes, image_wh, sigma=1.5):
    target = build_coarse_heatmap(ori_gt_bboxes, image_wh, coarse_logits.shape[-1], sigma)
    return F.binary_cross_entropy_with_logits(coarse_logits, target, weight=1.0 + 4.0 * target)


def multigrid_yolo_loss(pred3, pred4, ori_gt_bboxes, anchors_full, image_wh):
    """One global confidence competition over 6*64^2 + 3*32^2 candidates."""
    batch = pred3.shape[0]
    _, best = build_target(ori_gt_bboxes, anchors_full, image_wh, 64)
    global_anchor = best[:, 0]
    gt_xywh = torch.empty_like(ori_gt_bboxes)
    gt_xywh[:, 0:2] = (ori_gt_bboxes[:, 0:2] + ori_gt_bboxes[:, 2:4]) * 0.5
    gt_xywh[:, 2:4] = ori_gt_bboxes[:, 2:4] - ori_gt_bboxes[:, 0:2]
    geo_losses = []
    target_indices = []
    for index in range(batch):
        anchor = global_anchor[index]
        if anchor < 3:
            prediction, local_anchor, grid, offset = pred4, anchor, 32, 6 * 64 * 64
        else:
            prediction, local_anchor, grid, offset = pred3, anchor - 3, 64, 0
        stride = image_wh // grid
        scaled_anchor = anchors_full[anchor] / stride
        gxy = gt_xywh[index, 0:2] / stride
        gij = gxy.long().clamp(0, grid - 1)
        target = torch.cat((gxy - gij, torch.log(gt_xywh[index, 2:4] / stride / scaled_anchor + 1e-16)))
        selected = prediction[index, local_anchor, :, gij[1], gij[0]]
        box_pred = torch.cat((selected[0:2].sigmoid(), selected[2:4]))
        geo_losses.append(F.mse_loss(box_pred, target, reduction='sum'))
        target_indices.append(offset + local_anchor * grid * grid + gij[1] * grid + gij[0])
    loss_geo = torch.stack(geo_losses).mean()
    all_conf = torch.cat((pred3[:, :, 4].reshape(batch, -1), pred4[:, :, 4].reshape(batch, -1)), dim=1)
    loss_cls = F.cross_entropy(all_conf, torch.stack(target_indices).long())
    return loss_geo, loss_cls


def two_head_yolo_loss(pred3, pred4, ori_gt_bboxes, anchors_full, image_wh):
    target, best = build_target(ori_gt_bboxes, anchors_full, image_wh, 64)
    geo3, cls3 = yolo_loss(pred3, target, anchors_full, best, image_wh)
    geo4, cls4 = yolo_loss(pred4, target, anchors_full, best, image_wh)
    return 0.5 * (geo3 + geo4), 0.5 * (cls3 + cls4)


def decode_positive_box(predictions, anchors_full, best_anchor_gi_gj, image_wh):
    """Differentiably decode each GT-matched positive candidate to xyxy pixels."""
    if predictions.ndim != 5 or predictions.shape[2] != 5:
        raise ValueError('decode_positive_box expects [B,A,5,H,W], got {}'.format(tuple(predictions.shape)))
    grid_h, grid_w = predictions.shape[-2:]
    if grid_h != grid_w:
        raise ValueError('decode_positive_box requires a square grid, got {}x{}'.format(grid_h, grid_w))
    batch_size = predictions.shape[0]
    best_anchor = best_anchor_gi_gj[:, 0].long()
    gi, gj = best_anchor_gi_gj[:, 1].long(), best_anchor_gi_gj[:, 2].long()
    batch_idx = torch.arange(batch_size, device=predictions.device)
    selected = predictions[batch_idx, best_anchor, :, gj, gi]
    stride = float(image_wh) / float(grid_w)
    dtype = selected.dtype
    center = (selected[:, :2].sigmoid() + torch.stack((gi, gj), dim=1).to(dtype)) * stride
    anchor_wh = anchors_full.to(device=predictions.device, dtype=dtype)[best_anchor]
    wh = torch.exp(selected[:, 2:4]) * anchor_wh
    return torch.cat((center - 0.5 * wh, center + 0.5 * wh), dim=1)


def positive_box_iou(predictions, ori_gt_bboxes, anchors_full, best_anchor_gi_gj, image_wh, eps=1e-16):
    """Differentiable IoU between every positive candidate and its GT box."""
    pred_box = decode_positive_box(predictions, anchors_full, best_anchor_gi_gj, image_wh)
    gt_box = ori_gt_bboxes.to(device=pred_box.device, dtype=pred_box.dtype)
    top_left = torch.maximum(pred_box[:, :2], gt_box[:, :2])
    bottom_right = torch.minimum(pred_box[:, 2:], gt_box[:, 2:])
    inter = (bottom_right - top_left).clamp_min(0.0).prod(dim=1)
    pred_area = (pred_box[:, 2:] - pred_box[:, :2]).clamp_min(0.0).prod(dim=1)
    gt_area = (gt_box[:, 2:] - gt_box[:, :2]).clamp_min(0.0).prod(dim=1)
    return (inter / (pred_area + gt_area - inter + eps)).clamp(0.0, 1.0)


def threshold_aware_iou_loss(iou, temperature=0.05, weight25=0.5, weight50=1.0):
    """Smooth losses targeted at the Acc@0.25 and Acc@0.50 thresholds."""
    if temperature <= 0.0:
        raise ValueError('threshold temperature must be > 0')
    if weight25 < 0.0 or weight50 < 0.0:
        raise ValueError('threshold weights must be >= 0')
    loss25 = temperature * F.softplus((0.25 - iou) / temperature)
    loss50 = temperature * F.softplus((0.50 - iou) / temperature)
    return (weight25 * loss25 + weight50 * loss50).mean()


def two_head_yolo_threshold_reg_loss(pred3, pred4, ori_gt_bboxes, anchors_full, image_wh,
                                     reg_weight=0.2, temperature=0.05, weight25=0.5, weight50=1.0):
    """Unchanged two-head YOLO loss plus a positive-candidate threshold regularizer."""
    if reg_weight < 0.0:
        raise ValueError('reg_weight must be >= 0')
    target, best = build_target(ori_gt_bboxes, anchors_full, image_wh, 64)
    geo3_mse, cls3 = yolo_loss(pred3, target, anchors_full, best, image_wh)
    geo4_mse, cls4 = yolo_loss(pred4, target, anchors_full, best, image_wh)
    iou3 = positive_box_iou(pred3, ori_gt_bboxes, anchors_full, best, image_wh)
    iou4 = positive_box_iou(pred4, ori_gt_bboxes, anchors_full, best, image_wh)
    reg3 = threshold_aware_iou_loss(iou3, temperature, weight25, weight50)
    reg4 = threshold_aware_iou_loss(iou4, temperature, weight25, weight50)
    loss_geo = 0.5 * ((geo3_mse + reg_weight * reg3) + (geo4_mse + reg_weight * reg4))
    return loss_geo, 0.5 * (cls3 + cls4)


def three_head_yolo_loss(pred2, pred3, pred4, ori_gt_bboxes, anchors_full, image_wh):
    """E7: every independent 9-anchor head receives the same 64x64 target."""
    target, best = build_target(ori_gt_bboxes, anchors_full, image_wh, 64)
    geo2, cls2 = yolo_loss(pred2, target, anchors_full, best, image_wh)
    geo3, cls3 = yolo_loss(pred3, target, anchors_full, best, image_wh)
    geo4, cls4 = yolo_loss(pred4, target, anchors_full, best, image_wh)
    return (geo2 + geo3 + geo4) / 3.0, (cls2 + cls3 + cls4) / 3.0


def three_head_yolo_loss_stage2_cls_half(pred2, pred3, pred4, ori_gt_bboxes, anchors_full, image_wh):
    """E8: retain equal geometry supervision but downweight Stage2 confidence to 0.5."""
    target, best = build_target(ori_gt_bboxes, anchors_full, image_wh, 64)
    geo2, cls2 = yolo_loss(pred2, target, anchors_full, best, image_wh)
    geo3, cls3 = yolo_loss(pred3, target, anchors_full, best, image_wh)
    geo4, cls4 = yolo_loss(pred4, target, anchors_full, best, image_wh)
    loss_geo = (geo2 + geo3 + geo4) / 3.0
    loss_cls = (0.5 * cls2 + cls3 + cls4) / 2.5
    return loss_geo, loss_cls


def rccd_consensus_loss(pred3, pred4, temperature=2.0, weight_temperature=0.2, eps=1e-8):
    """Training-only reliability-aware consensus over two heads' confidence maps."""
    if pred3.shape != pred4.shape:
        raise ValueError('RCCD requires identical prediction shapes, got {} and {}'.format(
            tuple(pred3.shape), tuple(pred4.shape)))
    if pred3.ndim != 5 or pred3.shape[1:3] != (9, 5):
        raise ValueError('RCCD expects [B,9,5,H,W], got {}'.format(tuple(pred3.shape)))
    if temperature <= 0.0 or weight_temperature <= 0.0:
        raise ValueError('RCCD temperatures must be positive')

    batch_size = pred3.shape[0]
    logits3 = pred3[:, :, 4].reshape(batch_size, -1)
    logits4 = pred4[:, :, 4].reshape(batch_size, -1)
    log_q3 = F.log_softmax(logits3 / temperature, dim=1)
    log_q4 = F.log_softmax(logits4 / temperature, dim=1)
    q3, q4 = log_q3.exp(), log_q4.exp()

    # The teacher-selection path is explicitly stop-gradient: confidence cannot
    # be sharpened merely to gain a larger teacher share.
    with torch.no_grad():
        prob3, prob4 = F.softmax(logits3, dim=1), F.softmax(logits4, dim=1)
        entropy3 = -(prob3 * prob3.clamp_min(eps).log()).sum(dim=1)
        entropy4 = -(prob4 * prob4.clamp_min(eps).log()).sum(dim=1)
        normalizer = math.log(float(logits3.shape[1]))
        reliability3, reliability4 = 1.0 - entropy3 / normalizer, 1.0 - entropy4 / normalizer
        weights = F.softmax(torch.stack((reliability3, reliability4), dim=1) / weight_temperature, dim=1)
        weight3, weight4 = weights[:, :1], weights[:, 1:]
        teacher = (weight3 * q3 + weight4 * q4).clamp_min(eps)
        teacher = teacher / teacher.sum(dim=1, keepdim=True)

    loss3 = F.kl_div(log_q3, teacher, reduction='batchmean')
    loss4 = F.kl_div(log_q4, teacher, reduction='batchmean')
    loss = 0.5 * (temperature ** 2) * (loss3 + loss4)
    diagnostics = {
        'rccd_reliability3': reliability3.mean(), 'rccd_reliability4': reliability4.mean(),
        'rccd_weight3': weight3.mean(), 'rccd_weight4': weight4.mean(),
        'rccd_entropy3': entropy3.mean(), 'rccd_entropy4': entropy4.mean(),
    }
    return loss, diagnostics
