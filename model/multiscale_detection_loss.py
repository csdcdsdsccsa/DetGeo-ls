"""Losses for true multi-grid (E2) and two-head (E3/E4) detection ablations."""

import torch
import torch.nn.functional as F

from .loss import build_target, yolo_loss


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


def three_head_yolo_loss(pred2, pred3, pred4, ori_gt_bboxes, anchors_full, image_wh):
    """E7: every independent 9-anchor head receives the same 64x64 target."""
    target, best = build_target(ori_gt_bboxes, anchors_full, image_wh, 64)
    geo2, cls2 = yolo_loss(pred2, target, anchors_full, best, image_wh)
    geo3, cls3 = yolo_loss(pred3, target, anchors_full, best, image_wh)
    geo4, cls4 = yolo_loss(pred4, target, anchors_full, best, image_wh)
    return (geo2 + geo3 + geo4) / 3.0, (cls2 + cls3 + cls4) / 3.0
