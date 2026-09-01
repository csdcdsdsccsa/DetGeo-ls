"""Losses and targets for the E05 anchor-free localization experiment."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_radius(height, width, min_overlap=0.7):
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    radius1 = (b1 + math.sqrt(max(0.0, b1 ** 2 - 4.0 * a1 * c1))) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    radius2 = (b2 + math.sqrt(max(0.0, b2 ** 2 - 4.0 * a2 * c2))) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    radius3 = (b3 + math.sqrt(max(0.0, b3 ** 2 - 4.0 * a3 * c3))) / 2.0
    return max(0, int(min(radius1, radius2, radius3)))


def draw_gaussian(heatmap, center_x, center_y, radius):
    diameter = 2 * radius + 1
    sigma = max(diameter / 6.0, 1e-6)
    coordinates = torch.arange(diameter, device=heatmap.device, dtype=heatmap.dtype) - radius
    gaussian = torch.exp(-(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / (2 * sigma ** 2))

    height, width = heatmap.shape
    left, right = min(center_x, radius), min(width - center_x - 1, radius)
    top, bottom = min(center_y, radius), min(height - center_y - 1, radius)
    heatmap_slice = heatmap[center_y - top:center_y + bottom + 1,
                            center_x - left:center_x + right + 1]
    gaussian_slice = gaussian[radius - top:radius + bottom + 1,
                              radius - left:radius + right + 1]
    torch.maximum(heatmap_slice, gaussian_slice, out=heatmap_slice)


def build_center_targets(boxes_xyxy, height, width, image_size):
    batch_size = boxes_xyxy.shape[0]
    stride_x = float(image_size) / width
    stride_y = float(image_size) / height
    heatmap = boxes_xyxy.new_zeros((batch_size, 1, height, width))
    indices = torch.zeros((batch_size, 2), dtype=torch.long, device=boxes_xyxy.device)
    ltrb_targets = boxes_xyxy.new_zeros((batch_size, 4))

    for batch_index in range(batch_size):
        x1, y1, x2, y2 = boxes_xyxy[batch_index]
        center_x_float = ((x1 + x2) * 0.5 / stride_x).clamp(0, width - 1e-4)
        center_y_float = ((y1 + y2) * 0.5 / stride_y).clamp(0, height - 1e-4)
        center_x = int(center_x_float)
        center_y = int(center_y_float)
        indices[batch_index] = torch.tensor(
            [center_y, center_x], device=boxes_xyxy.device, dtype=torch.long)

        box_width = max(float((x2 - x1) / stride_x), 1.0)
        box_height = max(float((y2 - y1) / stride_y), 1.0)
        draw_gaussian(heatmap[batch_index, 0], center_x, center_y,
                      gaussian_radius(box_height, box_width))

        cell_x = center_x + 0.5
        cell_y = center_y + 0.5
        ltrb_targets[batch_index] = torch.stack((
            boxes_xyxy.new_tensor(cell_x) - x1 / stride_x,
            boxes_xyxy.new_tensor(cell_y) - y1 / stride_y,
            x2 / stride_x - boxes_xyxy.new_tensor(cell_x),
            y2 / stride_y - boxes_xyxy.new_tensor(cell_y),
        )).clamp_min(0)
    return heatmap, indices, ltrb_targets


def center_focal_loss(logits, targets):
    probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    positive = targets.eq(1.0).to(logits.dtype)
    negative = targets.lt(1.0).to(logits.dtype)
    negative_weight = (1.0 - targets).pow(4)
    positive_loss = -torch.log(probability) * (1.0 - probability).pow(2) * positive
    negative_loss = -torch.log(1.0 - probability) * probability.pow(2) * negative_weight * negative
    positive_count = positive.sum().clamp_min(1.0)
    return (positive_loss.sum() + negative_loss.sum()) / positive_count


def gather_at_indices(features, indices):
    batch_indices = torch.arange(features.shape[0], device=features.device)
    return features[batch_indices, :, indices[:, 0], indices[:, 1]]


def decode_ltrb_at_indices(ltrb, indices, image_size):
    height, width = ltrb.shape[-2:]
    stride_x = float(image_size) / width
    stride_y = float(image_size) / height
    selected = gather_at_indices(ltrb, indices)
    cell_x = (indices[:, 1].to(ltrb.dtype) + 0.5) * stride_x
    cell_y = (indices[:, 0].to(ltrb.dtype) + 0.5) * stride_y
    return torch.stack((
        cell_x - selected[:, 0] * stride_x,
        cell_y - selected[:, 1] * stride_y,
        cell_x + selected[:, 2] * stride_x,
        cell_y + selected[:, 3] * stride_y,
    ), dim=1)


def aligned_iou(boxes1, boxes2):
    inter_x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    intersection = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    return intersection / (area1 + area2 - intersection + 1e-7)


def generalized_iou_loss(boxes1, boxes2):
    iou = aligned_iou(boxes1, boxes2)
    enclose_x1 = torch.minimum(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.minimum(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.maximum(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.maximum(boxes1[:, 3], boxes2[:, 3])
    enclosing = (enclose_x2 - enclose_x1).clamp_min(0) * (enclose_y2 - enclose_y1).clamp_min(0)

    inter_x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    intersection = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    union = area1 + area2 - intersection
    giou = iou - (enclosing - union) / enclosing.clamp_min(1e-7)
    return (1.0 - giou).mean()


class E05Loss(nn.Module):
    def __init__(self, image_size=1024, search_weight=1.0, center_weight=1.0,
                 box_weight=5.0, quality_weight=1.0):
        super().__init__()
        self.image_size = int(image_size)
        self.search_weight = float(search_weight)
        self.center_weight = float(center_weight)
        self.box_weight = float(box_weight)
        self.quality_weight = float(quality_weight)

    def forward(self, outputs, boxes_xyxy):
        center_logits = outputs["center_logits"]
        height, width = center_logits.shape[-2:]
        center_target, indices, ltrb_target = build_center_targets(
            boxes_xyxy, height, width, self.image_size)

        loss_center = center_focal_loss(center_logits, center_target)
        selected_ltrb = gather_at_indices(outputs["ltrb"], indices)
        distance_normalizer = float(max(height, width))
        loss_l1 = F.smooth_l1_loss(
            selected_ltrb / distance_normalizer,
            ltrb_target / distance_normalizer)
        predicted_boxes = decode_ltrb_at_indices(outputs["ltrb"], indices, self.image_size)
        loss_giou = generalized_iou_loss(predicted_boxes, boxes_xyxy)
        loss_box = loss_giou + loss_l1

        quality_logits = gather_at_indices(outputs["quality_logits"], indices).squeeze(1)
        quality_target = aligned_iou(predicted_boxes.detach(), boxes_xyxy).clamp(0, 1)
        loss_quality = F.binary_cross_entropy_with_logits(quality_logits, quality_target)

        search_losses = []
        for search_logits in outputs["search_logits"]:
            _, search_indices, _ = build_center_targets(
                boxes_xyxy, search_logits.shape[-2], search_logits.shape[-1], self.image_size)
            search_width = search_logits.shape[-1]
            search_target = search_indices[:, 0] * search_width + search_indices[:, 1]
            search_losses.append(F.cross_entropy(search_logits.flatten(1), search_target))
        if len(search_losses) == 3:
            loss_search = 0.25 * search_losses[0] + 0.5 * search_losses[1] + search_losses[2]
        else:
            loss_search = sum(search_losses) / len(search_losses)

        total = (
            self.search_weight * loss_search
            + self.center_weight * loss_center
            + self.box_weight * loss_box
            + self.quality_weight * loss_quality
        )
        return {
            "total": total,
            "search": loss_search,
            "center": loss_center,
            "box": loss_box,
            "quality": loss_quality,
            "giou": loss_giou,
            "l1": loss_l1,
        }
