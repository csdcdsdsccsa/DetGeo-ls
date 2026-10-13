"""Compare the standalone fixed protocol with the original multi-variant code."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('SOURCE_REPO', ROOT.parent / 'DetGeo-sam-prompt-yolo'))
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(SOURCE))

from fullmodel.objectives import (anchors_tensor, decode_fullmodel, fullmodel_loss,
                                  per_sample_metrics)
from fullmodel.data import CVOGLFullModelDataset
from dataset.trogeo_loader import TROGeoRSDataset
from model.multiscale_detection_loss import coarse_heatmap_loss, two_head_yolo_loss
from utils.multiscale_detection import eval_decoded_boxes, select_two_heads


def main():
    for shape, sx in (((256, 256), 25.0), ((256, 512), 50.0)):
        original_map = TROGeoRSDataset.make_click_map(shape, 95, 117, 'gaussian', 25.0, sx)
        standalone_map = CVOGLFullModelDataset.make_click_map(shape, 95, 117, 25.0, sx)
        assert (original_map == standalone_map).all(), shape
    torch.manual_seed(123)
    anchors = anchors_tensor('cpu')
    boxes = torch.tensor([[200., 240., 440., 510.], [100., 160., 290., 370.]])
    outputs = {
        'stage3': torch.randn(2, 45, 64, 64, requires_grad=True),
        'stage4': torch.randn(2, 45, 64, 64, requires_grad=True),
        'coarse_logits': torch.randn(2, 1, 32, 32, requires_grad=True),
        'fine_logits': torch.randn(2, 1, 64, 64, requires_grad=True),
    }
    p3 = outputs['stage3'].view(2, 9, 5, 64, 64)
    p4 = outputs['stage4'].view(2, 9, 5, 64, 64)
    geo, cls = two_head_yolo_loss(p3, p4, boxes, anchors, 1024)
    aux = 0.5 * (coarse_heatmap_loss(outputs['coarse_logits'], boxes, 1024, 1.5) +
                 coarse_heatmap_loss(outputs['fine_logits'], boxes, 1024, 3.0))
    source_loss = cls + geo + 0.2 * aux
    standalone_loss = fullmodel_loss(outputs, boxes, anchors)
    torch.testing.assert_close(standalone_loss, source_loss, rtol=0, atol=0)

    source_box, _ = select_two_heads(p3, p4, anchors, 1024, fusion=False,
                                     iou_threshold=0.5, adaptive=False)
    standalone_box = decode_fullmodel(outputs, anchors)
    torch.testing.assert_close(standalone_box, source_box, rtol=0, atol=0)
    source_metrics = eval_decoded_boxes(source_box, boxes, 1024, grid_size=64)
    standalone_metrics = per_sample_metrics(standalone_box, boxes).mean(0)
    for new, old in zip(standalone_metrics, source_metrics):
        torch.testing.assert_close(new, old, rtol=0, atol=0)
    print('click maps, loss, two-head decoding, and all four metrics match source exactly')


if __name__ == '__main__':
    main()
