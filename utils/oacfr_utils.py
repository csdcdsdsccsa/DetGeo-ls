"""Decode and metric helpers for anchor-free OA-CFRGeo experiments."""

import torch

from model.oacfr_loss import aligned_iou


def decode_anchor_free(outputs, image_size):
    center = outputs["center_logits"].sigmoid()
    quality = outputs["quality_logits"].sigmoid()
    scores = center * quality
    batch_size, _, height, width = scores.shape
    flat_indices = scores.flatten(1).argmax(dim=1)
    grid_y = torch.div(flat_indices, width, rounding_mode="floor")
    grid_x = flat_indices % width
    batch_indices = torch.arange(batch_size, device=scores.device)
    ltrb = outputs["ltrb"][batch_indices, :, grid_y, grid_x]

    stride_x = float(image_size) / width
    stride_y = float(image_size) / height
    center_x = (grid_x.to(ltrb.dtype) + 0.5) * stride_x
    center_y = (grid_y.to(ltrb.dtype) + 0.5) * stride_y
    boxes = torch.stack((
        center_x - ltrb[:, 0] * stride_x,
        center_y - ltrb[:, 1] * stride_y,
        center_x + ltrb[:, 2] * stride_x,
        center_y + ltrb[:, 3] * stride_y,
    ), dim=1).clamp(0, image_size - 1)
    selected_scores = scores[batch_indices, 0, grid_y, grid_x]
    return boxes, selected_scores, grid_y, grid_x


def localization_metrics(outputs, target_boxes, image_size):
    predicted_boxes, scores, grid_y, grid_x = decode_anchor_free(outputs, image_size)
    iou = aligned_iou(predicted_boxes, target_boxes)
    target_center_x = ((target_boxes[:, 0] + target_boxes[:, 2]) * 0.5
                       / (float(image_size) / outputs["center_logits"].shape[-1])).long()
    target_center_y = ((target_boxes[:, 1] + target_boxes[:, 3]) * 0.5
                       / (float(image_size) / outputs["center_logits"].shape[-2])).long()
    center_correct = (grid_x == target_center_x) & (grid_y == target_center_y)
    return {
        "boxes": predicted_boxes,
        "scores": scores,
        "iou": iou,
        "correct_025": iou > 0.25,
        "correct_050": iou > 0.5,
        "center_correct": center_correct,
    }
