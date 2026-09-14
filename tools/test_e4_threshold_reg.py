"""Threshold-Reg loss equivalence, differentiability, and monotonicity checks."""

import torch

from model.multiscale_detection_loss import (
    positive_box_iou,
    threshold_aware_iou_loss,
    two_head_yolo_loss,
    two_head_yolo_threshold_reg_loss,
)
from model.loss import build_target


def main():
    torch.manual_seed(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch, image_wh = 2, 1024
    pred3 = torch.randn(batch, 9, 5, 64, 64, device=device, requires_grad=True)
    pred4 = torch.randn(batch, 9, 5, 64, 64, device=device, requires_grad=True)
    anchors = torch.tensor([[550, 573], [395, 342], [246, 280], [198, 179], [194, 82],
                            [129, 129], [96, 215], [78, 84], [37, 41]], dtype=torch.float32, device=device)
    gt = torch.tensor([[200, 220, 360, 390], [500, 400, 700, 620]], dtype=torch.float32, device=device)

    old_geo, old_cls = two_head_yolo_loss(pred3, pred4, gt, anchors, image_wh)
    zero_geo, zero_cls = two_head_yolo_threshold_reg_loss(
        pred3, pred4, gt, anchors, image_wh, reg_weight=0.0)
    if not torch.allclose(old_geo, zero_geo, atol=1e-7, rtol=1e-6) or not torch.allclose(old_cls, zero_cls, atol=1e-7, rtol=1e-6):
        raise RuntimeError('reg_weight=0 must exactly reproduce two_head_yolo_loss')

    geo, cls = two_head_yolo_threshold_reg_loss(pred3, pred4, gt, anchors, image_wh)
    if not torch.isfinite(geo) or not torch.isfinite(cls):
        raise RuntimeError('Threshold-Reg loss must be finite')
    (geo + cls).backward()
    if pred3.grad is None or pred4.grad is None or not torch.isfinite(pred3.grad).all() or not torch.isfinite(pred4.grad).all():
        raise RuntimeError('Threshold-Reg backward failed')

    penalties = [threshold_aware_iou_loss(torch.tensor([value], device=device)).item()
                 for value in (0.10, 0.24, 0.30, 0.49, 0.51, 0.80)]
    if any(left <= right for left, right in zip(penalties, penalties[1:])):
        raise RuntimeError('threshold penalty must decrease as IoU increases')
    _, best = build_target(gt, anchors, image_wh, 64)
    iou = positive_box_iou(pred3.detach(), gt, anchors, best, image_wh)
    if not torch.isfinite(iou).all() or not bool(((iou >= 0.0) & (iou <= 1.0)).all()):
        raise RuntimeError('positive IoU must be finite and in [0, 1]')
    print('Threshold-Reg sanity passed: old_geo={:.6f} new_geo={:.6f} penalties={}'.format(
        old_geo.item(), geo.item(), [round(value, 6) for value in penalties]))


if __name__ == '__main__':
    main()
