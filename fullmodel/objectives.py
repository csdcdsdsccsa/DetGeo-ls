"""The fixed two-head Full Model training loss and validation decoder."""

import torch
import torch.nn.functional as F


ANCHORS = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'


def anchors_tensor(device):
    import numpy as np
    anchors = np.array([float(x.strip()) for x in ANCHORS.split(',')])
    return torch.tensor(anchors.reshape(-1, 2)[::-1].copy(), dtype=torch.float32, device=device)


def xyxy_to_xywh(box):
    out = torch.zeros_like(box)
    out[:, :2] = (box[:, :2] + box[:, 2:]) / 2
    out[:, 2:] = box[:, 2:] - box[:, :2]
    return out


def xywh_to_xyxy(box):
    out = torch.zeros_like(box)
    out[:, :2] = box[:, :2] - box[:, 2:] / 2
    out[:, 2:] = box[:, :2] + box[:, 2:] / 2
    return out


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        box1, box2 = xywh_to_xyxy(box1), xywh_to_xyxy(box2)
    top_left = torch.maximum(box1[:, :2], box2[:, :2])
    bottom_right = torch.minimum(box1[:, 2:], box2[:, 2:])
    inter = (bottom_right - top_left).clamp_min(0).prod(dim=1)
    area1 = (box1[:, 2:] - box1[:, :2]).prod(dim=1)
    area2 = (box2[:, 2:] - box2[:, :2]).prod(dim=1)
    return inter / (area1 + area2 - inter + 1e-16)


def build_target(boxes, anchors, image_wh, grid_wh):
    batch, dims, stride, count = boxes.shape[0], boxes.shape[1], image_wh // grid_wh, anchors.shape[0]
    gt = xyxy_to_xywh(boxes) / image_wh * grid_wh
    scaled = anchors / stride
    gxy, gwh = gt[:, :2], gt[:, 2:]
    gij = gxy.long()
    gt_tmp = torch.zeros_like(gt)
    gt_tmp[:, 2:4] = gwh
    gt_tmp = gt_tmp.unsqueeze(1).repeat(1, count, 1).view(-1, dims)
    anchor_tmp = torch.zeros((count, dims), device=gt.device)
    anchor_tmp[:, 2:4] = scaled
    anchor_tmp = anchor_tmp.unsqueeze(0).repeat(batch, 1, 1).view(-1, dims)
    best = bbox_iou(gt_tmp, anchor_tmp, x1y1x2y2=False).view(batch, -1).argmax(dim=1)
    twh = torch.log(gwh / scaled[best] + 1e-16)
    return torch.cat((gxy - gij, twh), dim=1), torch.cat((best[:, None], gij), dim=1)


def yolo_loss(prediction, target, best):
    batch = prediction.shape[0]
    anchor, gi, gj = best[:, 0], best[:, 1], best[:, 2]
    selected = prediction[range(batch), anchor, :, gj, gi]
    pred_box = torch.zeros_like(target)
    pred_box[:, :2] = selected[:, :2].sigmoid()
    pred_box[:, 2:] = selected[:, 2:4]
    geo = sum(F.mse_loss(pred_box[:, index], target[:, index]) for index in range(4))
    confidence = prediction[:, :, 4].reshape(batch, -1)
    gt_confidence = torch.zeros_like(prediction[:, :, 4])
    gt_confidence[range(batch), anchor, gj, gi] = 1
    cls = F.cross_entropy(confidence, gt_confidence.reshape(batch, -1).max(1)[1])
    return geo, cls


def coarse_heatmap_loss(logits, boxes, image_wh, sigma):
    grid = logits.shape[-1]
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5 / float(image_wh) * grid
    coordinates = torch.arange(grid, device=boxes.device, dtype=boxes.dtype)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    dist2 = ((xx.unsqueeze(0) - centers[:, 0, None, None]) ** 2 +
             (yy.unsqueeze(0) - centers[:, 1, None, None]) ** 2)
    target = torch.exp(-dist2 / (2.0 * sigma * sigma)).unsqueeze(1)
    return F.binary_cross_entropy_with_logits(logits, target, weight=1.0 + 4.0 * target)


def fullmodel_loss(outputs, boxes, anchors, image_wh=1024):
    p3 = outputs['stage3'].view(-1, 9, 5, 64, 64)
    p4 = outputs['stage4'].view(-1, 9, 5, 64, 64)
    target, best = build_target(boxes, anchors, image_wh, 64)
    geo3, cls3 = yolo_loss(p3, target, best)
    geo4, cls4 = yolo_loss(p4, target, best)
    geo, cls = 0.5 * (geo3 + geo4), 0.5 * (cls3 + cls4)
    coarse = coarse_heatmap_loss(outputs['coarse_logits'], boxes, image_wh, 1.5)
    fine = coarse_heatmap_loss(outputs['fine_logits'], boxes, image_wh, 3.0)
    return cls + geo + 0.2 * 0.5 * (coarse + fine)


def decode_top1(prediction, anchors, image_wh):
    batch, _, _, grid, _ = prediction.shape
    flat = prediction[:, :, 4].reshape(batch, -1)
    best = flat.argmax(dim=1)
    cells = grid * grid
    anchor, cell = best // cells, best % cells
    gj, gi = cell // grid, cell % grid
    selected = prediction[torch.arange(batch, device=prediction.device), anchor, :, gj, gi]
    stride = image_wh // grid
    scaled = anchors[anchor] / stride
    xywh = torch.cat(((selected[:, :2].sigmoid() + torch.stack((gi, gj), dim=1)) * stride,
                      torch.exp(selected[:, 2:4]) * scaled * stride), dim=1)
    return xywh_to_xyxy(xywh), flat.softmax(dim=1).max(dim=1).values


def decode_fullmodel(outputs, anchors, image_wh=1024):
    p3 = outputs['stage3'].view(-1, 9, 5, 64, 64)
    p4 = outputs['stage4'].view(-1, 9, 5, 64, 64)
    box3, score3 = decode_top1(p3, anchors, image_wh)
    box4, score4 = decode_top1(p4, anchors, image_wh)
    return torch.where((score3 >= score4)[:, None], box3, box4)


def per_sample_metrics(pred, target, image_wh=1024):
    iou = bbox_iou(pred, target)
    pred_center = (pred[:, :2] + pred[:, 2:]) * 0.5
    target_center = (target[:, :2] + target[:, 2:]) * 0.5
    pred_grid = (pred_center / (float(image_wh) / 64)).long()
    target_grid = (target_center / (float(image_wh) / 64)).long()
    return torch.stack((iou.gt(0.5).float(), iou.gt(0.25).float(), iou,
                        (pred_grid == target_grid).all(dim=1).float()), dim=1)
