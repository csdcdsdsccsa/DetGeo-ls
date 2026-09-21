"""CPU checks for VIGOR's 640px and CVOGL's 1024px detector plumbing."""

import numpy as np
import torch

from model.multiscale_detection_loss import two_head_yolo_loss, two_head_yolo_threshold_reg_loss
from utils.multiscale_detection import eval_decoded_boxes


def main():
    torch.manual_seed(13)
    anchors_str = ('137,82, 144,164, 479,243, 255,537, 73,202, '
                   '242,117, 175,359, 259,260, 74,108')
    anchors = torch.tensor(np.asarray([float(value) for value in anchors_str.split(',')], dtype=np.float32)
                           .reshape(-1, 2)[::-1].copy())
    boxes = torch.tensor([[140., 160., 350., 420.], [80., 90., 290., 310.]])
    p3 = torch.randn(2, 9, 5, 40, 40, requires_grad=True)
    p4 = torch.randn(2, 9, 5, 40, 40, requires_grad=True)
    geo, cls = two_head_yolo_loss(p3, p4, boxes, anchors, 640)
    (geo + cls).backward()
    assert p3.grad is not None and p4.grad is not None
    reg_geo, reg_cls = two_head_yolo_threshold_reg_loss(p3.detach(), p4.detach(), boxes, anchors, 640)
    assert torch.isfinite(reg_geo + reg_cls)
    metrics = eval_decoded_boxes(boxes, boxes, 640, grid_size=40)
    assert all(torch.isclose(value, torch.tensor(1.0)) for value in metrics)
    legacy_anchors = torch.tensor(np.asarray([
        37, 41, 78, 84, 96, 215, 129, 129, 194, 82,
        198, 179, 246, 280, 395, 342, 550, 573], dtype=np.float32).reshape(-1, 2)[::-1].copy())
    legacy_boxes = boxes * (1024.0 / 640.0)
    legacy_p3 = torch.randn(2, 9, 5, 64, 64, requires_grad=True)
    legacy_p4 = torch.randn(2, 9, 5, 64, 64, requires_grad=True)
    legacy_geo, legacy_cls = two_head_yolo_loss(legacy_p3, legacy_p4, legacy_boxes, legacy_anchors, 1024)
    (legacy_geo + legacy_cls).backward()
    assert legacy_p3.grad is not None and legacy_p4.grad is not None
    print('VIGOR/CVOGL dynamic-grid loss/decode checks passed: 640/40 and 1024/64')


if __name__ == '__main__':
    main()
