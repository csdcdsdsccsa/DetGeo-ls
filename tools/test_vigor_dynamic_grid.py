"""CPU checks for VIGOR's 640px / 40x40 detector plumbing."""

import torch

from model.multiscale_detection_loss import two_head_yolo_loss, two_head_yolo_threshold_reg_loss
from utils.multiscale_detection import eval_decoded_boxes


def main():
    torch.manual_seed(13)
    anchors = torch.tensor([
        [82., 137.], [164., 144.], [243., 479.], [537., 255.], [202., 73.],
        [117., 242.], [359., 175.], [260., 259.], [108., 74.],
    ])
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
    print('VIGOR dynamic-grid loss/decode checks passed: 40x40 at image_wh=640')


if __name__ == '__main__':
    main()
