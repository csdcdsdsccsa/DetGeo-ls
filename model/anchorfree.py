# -*- coding: utf-8 -*-
"""Targets, loss, decoding, and metrics for the SMGeo-style detection head."""

import math

import torch
import torch.nn.functional as F

from utils.utils import bbox_iou, xywh2xyxy


def build_target_anchorfree(gt_bboxes, feat_h, feat_w, image_size, sigma=2):
    """Map xyxy pixel boxes to a Gaussian center heatmap and box targets."""
    batch_size = gt_bboxes.shape[0]
    heatmap = gt_bboxes.new_zeros((batch_size, 1, feat_h, feat_w))
    bbox_target = gt_bboxes.new_zeros((batch_size, 4, feat_h, feat_w))
    mask = gt_bboxes.new_zeros((batch_size, 1, feat_h, feat_w))

    for index in range(batch_size):
        x1, y1, x2, y2 = gt_bboxes[index]
        center_x = (x1 + x2) * 0.5 / image_size * feat_w
        center_y = (y1 + y2) * 0.5 / image_size * feat_h
        width = (x2 - x1).clamp_min(0) / image_size * feat_w
        height = (y2 - y1).clamp_min(0) / image_size * feat_h
        center_x_int = int(center_x.clamp(0, feat_w - 1).item())
        center_y_int = int(center_y.clamp(0, feat_h - 1).item())

        for dx in range(-2 * sigma, 2 * sigma + 1):
            for dy in range(-2 * sigma, 2 * sigma + 1):
                grid_x, grid_y = center_x_int + dx, center_y_int + dy
                if 0 <= grid_x < feat_w and 0 <= grid_y < feat_h:
                    heatmap[index, 0, grid_y, grid_x] = math.exp(
                        -(dx * dx + dy * dy) / (2 * sigma * sigma)
                    )
                    bbox_target[index, :, grid_y, grid_x] = torch.stack((
                        center_x - grid_x, center_y - grid_y, width, height
                    ))
                    mask[index, 0, grid_y, grid_x] = 1
    return heatmap, bbox_target, mask


def anchorfree_loss(pred_heatmap, pred_bbox, gt_heatmap, gt_bbox, mask):
    """SMGeo heatmap focal loss plus masked L1 box regression loss."""
    pred_heatmap = pred_heatmap.sigmoid().clamp(1e-4, 1 - 1e-4)
    positive = gt_heatmap.eq(1)
    negative = gt_heatmap.lt(1)

    if positive.any():
        positive_loss = (-torch.log(pred_heatmap[positive])
                         * (1 - pred_heatmap[positive]).pow(2.0) * 0.25).mean()
    else:
        positive_loss = pred_heatmap.new_zeros(())
    if negative.any():
        negative_loss = (-torch.log(1 - pred_heatmap[negative])
                         * pred_heatmap[negative].pow(2.0) * 0.75).mean()
    else:
        negative_loss = pred_heatmap.new_zeros(())
    heatmap_loss = (positive_loss + negative_loss).clamp(max=10.0) * 2.0

    center_mask = mask.expand_as(pred_bbox[:, :2])
    size_mask = mask.expand_as(pred_bbox[:, 2:])
    center_loss = F.l1_loss(pred_bbox[:, :2] * center_mask,
                            gt_bbox[:, :2] * center_mask,
                            reduction='sum') / (center_mask.sum() + 1e-4)
    size_loss = F.l1_loss(pred_bbox[:, 2:] * size_mask,
                          gt_bbox[:, 2:] * size_mask,
                          reduction='sum') / (size_mask.sum() + 1e-4)
    bbox_loss = (0.7 * center_loss + 0.3 * size_loss).clamp(max=5.0) * 3.0
    return heatmap_loss, bbox_loss


def decode_anchorfree(heatmap_logits, bbox_offsets, image_size):
    """Decode the highest-scoring cell of each sample into xyxy pixel boxes."""
    batch_size, _, feat_h, feat_w = heatmap_logits.shape
    scores = heatmap_logits.sigmoid().flatten(1)
    best_indices = scores.argmax(dim=1)
    grid_y = torch.div(best_indices, feat_w, rounding_mode='floor')
    grid_x = best_indices.remainder(feat_w)
    batch_indices = torch.arange(batch_size, device=heatmap_logits.device)
    offsets = bbox_offsets[batch_indices, :, grid_y, grid_x]

    center_x = (grid_x.to(offsets.dtype) + offsets[:, 0]) * image_size / feat_w
    center_y = (grid_y.to(offsets.dtype) + offsets[:, 1]) * image_size / feat_h
    width = offsets[:, 2].clamp_min(1e-4) * image_size / feat_w
    height = offsets[:, 3].clamp_min(1e-4) * image_size / feat_h
    boxes = xywh2xyxy(torch.stack((center_x, center_y, width, height), dim=1))
    return boxes.clamp(0, image_size), grid_x, grid_y


def eval_anchorfree_acc(heatmap_logits, bbox_offsets, target_boxes, image_size,
                        iou_threshold_list=(0.5,)):
    """Return DetGeo-compatible IoU and grid-center metrics for the new head."""
    pred_boxes, pred_grid_x, pred_grid_y = decode_anchorfree(
        heatmap_logits, bbox_offsets, image_size
    )
    ious = bbox_iou(pred_boxes, target_boxes, x1y1x2y2=True)
    target_center_x = ((target_boxes[:, 0] + target_boxes[:, 2]) * 0.5
                       / image_size * heatmap_logits.shape[3]).long().clamp(
                           0, heatmap_logits.shape[3] - 1)
    target_center_y = ((target_boxes[:, 1] + target_boxes[:, 3]) * 0.5
                       / image_size * heatmap_logits.shape[2]).long().clamp(
                           0, heatmap_logits.shape[2] - 1)
    accuracies = [(ious > threshold).float().mean() for threshold in iou_threshold_list]
    each_accuracies = [ious > threshold for threshold in iou_threshold_list]
    center_accuracy = ((pred_grid_x == target_center_x)
                       & (pred_grid_y == target_center_y)).float().mean()
    return accuracies, center_accuracy, ious.mean(), each_accuracies, pred_boxes, target_boxes
