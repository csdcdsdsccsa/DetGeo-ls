"""CUDA smoke test for the standalone single-scale cross-attention ablation."""

import torch

from model.DetGeo_single_scale_ca import DetGeoSingleScaleCA
from model.loss import build_target, yolo_loss


def main():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = DetGeoSingleScaleCA().cuda().train()
    query = torch.randn(1, 3, 256, 256, device='cuda')
    reference = torch.randn(1, 3, 1024, 1024, device='cuda')
    click = torch.randn(1, 256, 256, device='cuda')
    outbox, attention = model(query, reference, click)
    predictions = outbox.view(1, 9, 5, 64, 64)
    anchors = torch.tensor(
        [[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129], [96, 215], [78, 84], [37, 41]],
        dtype=torch.float32,
        device='cuda',
    )
    gt_xyxy = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda')
    target, anchor_index = build_target(gt_xyxy, anchors, 1024, predictions.shape[-1])
    geo_loss, cls_loss = yolo_loss(predictions, target, anchors, anchor_index, 1024)
    smoke_loss = geo_loss + cls_loss
    if not torch.isfinite(smoke_loss):
        raise RuntimeError('non-finite smoke loss')
    smoke_loss.backward()
    mhca_grad = model.spatial_cross_attention.mhca.in_proj_weight.grad
    ffn_grad = model.spatial_cross_attention.ffn[0].weight.grad
    if mhca_grad is None or not torch.count_nonzero(mhca_grad):
        raise RuntimeError('MHCA did not receive a nonzero gradient')
    if ffn_grad is None or not torch.count_nonzero(ffn_grad):
        raise RuntimeError('FFN did not receive a nonzero gradient')
    if outbox.shape != (1, 45, 64, 64) or attention.shape != (1, 64, 64):
        raise RuntimeError('unexpected output shapes: {} {}'.format(tuple(outbox.shape), tuple(attention.shape)))
    print(
        'single_scale_ca smoke_loss={:.8f} total_params={} extra_params={} peak_mib={:.1f}'.format(
            smoke_loss.item(),
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in model.spatial_cross_attention.parameters()),
            torch.cuda.max_memory_allocated() / 1024 / 1024,
        )
    )


if __name__ == '__main__':
    main()
