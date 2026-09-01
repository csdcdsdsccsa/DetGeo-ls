"""DetGeo detection-head decoding, NMS and frozen feature extraction helpers."""

import torch
from torchvision.ops import nms, roi_align


def decode_candidates(prediction, anchors_full, image_size):
    """Decode all 9 x H x W DetGeo detection-head proposals for one image."""
    anchor_count, _, height, width = prediction.shape
    if height != width:
        raise ValueError("DetGeo ranking expects a square detection map")
    total = anchor_count * height * width
    flat_indices = torch.arange(total, device=prediction.device)
    cells_per_anchor = height * width
    anchor_index = torch.div(flat_indices, cells_per_anchor, rounding_mode="floor")
    cell_index = flat_indices % cells_per_anchor
    grid_y = torch.div(cell_index, width, rounding_mode="floor")
    grid_x = cell_index % width
    scores = prediction[:, 4].reshape(-1)

    stride = image_size // height
    scaled_anchors = anchors_full / stride
    centre_x = (prediction[anchor_index, 0, grid_y, grid_x].sigmoid() + grid_x) * stride
    centre_y = (prediction[anchor_index, 1, grid_y, grid_x].sigmoid() + grid_y) * stride
    box_w = torch.exp(prediction[anchor_index, 2, grid_y, grid_x]) * scaled_anchors[anchor_index, 0] * stride
    box_h = torch.exp(prediction[anchor_index, 3, grid_y, grid_x]) * scaled_anchors[anchor_index, 1] * stride
    boxes = torch.stack((centre_x - box_w / 2, centre_y - box_h / 2,
                         centre_x + box_w / 2, centre_y + box_h / 2), dim=1)
    return scores, boxes


def nms_topk(prediction, anchors_full, image_size, candidate_count, nms_iou,
             return_logits=False):
    """Produce score-sorted NMS Top-K boxes for a [B, 9, 5, H, W] prediction."""
    all_boxes, all_scores = [], []
    for item in range(prediction.shape[0]):
        scores, boxes = decode_candidates(prediction[item], anchors_full, image_size)
        kept = nms(boxes, scores, nms_iou)
        if kept.numel() < candidate_count:
            raise RuntimeError("NMS returned fewer candidates than requested")
        kept = kept[:candidate_count]
        all_boxes.append(boxes[kept])
        kept_scores = scores[kept]
        all_scores.append(kept_scores if return_logits else kept_scores.sigmoid())
    return torch.stack(all_boxes), torch.stack(all_scores)


def pairwise_iou_with_gt(boxes_xyxy, gt_boxes_xyxy):
    """IoU of [B, K, 4] candidate boxes against the matching [B, 4] GT box."""
    target = gt_boxes_xyxy.unsqueeze(1)
    inter_x1 = torch.maximum(boxes_xyxy[..., 0], target[..., 0])
    inter_y1 = torch.maximum(boxes_xyxy[..., 1], target[..., 1])
    inter_x2 = torch.minimum(boxes_xyxy[..., 2], target[..., 2])
    inter_y2 = torch.minimum(boxes_xyxy[..., 3], target[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_boxes = (boxes_xyxy[..., 2] - boxes_xyxy[..., 0]).clamp(min=0) * (boxes_xyxy[..., 3] - boxes_xyxy[..., 1]).clamp(min=0)
    area_target = (target[..., 2] - target[..., 0]).clamp(min=0) * (target[..., 3] - target[..., 1]).clamp(min=0)
    return inter / (area_boxes + area_target - inter + 1e-16)


def roi_align_candidates(reference_features, boxes_xyxy, image_size, output_size=7):
    """Extract [B, K, C, output_size, output_size] RoI features from satellite features."""
    batch_size, candidate_count = boxes_xyxy.shape[:2]
    batch_ids = torch.arange(batch_size, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    batch_ids = batch_ids[:, None, None].expand(-1, candidate_count, 1)
    rois = torch.cat((batch_ids, boxes_xyxy), dim=-1).reshape(-1, 5)
    scale = reference_features.shape[-1] / float(image_size)
    pooled = roi_align(reference_features, rois, output_size=output_size, spatial_scale=scale, aligned=True)
    return pooled.reshape(batch_size, candidate_count, *pooled.shape[1:])
