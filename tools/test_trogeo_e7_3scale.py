"""One real GPU forward/backward and memory check for E7 before long training."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.multiscale_detection_loss import three_head_yolo_loss, three_head_yolo_loss_stage2_cls_half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--variant', choices=('h2_ind_3scale', 'h2_ind_3scale_stage2cls05'),
                        default='h2_ind_3scale')
    args = parser.parse_args()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = torch.nn.DataParallel(
        TROGeoMSDetectionAblation(variant=args.variant)
    ).cuda().train()
    batch = args.batch_size
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.randn(batch, 256, 256, device='cuda')
    predictions, _ = model(query, reference, click)
    expected = {'stage2', 'stage3', 'stage4'}
    if set(predictions) != expected:
        raise RuntimeError('expected E7 prediction heads {}, got {}'.format(expected, set(predictions)))
    for name, prediction in predictions.items():
        if prediction.shape != (batch, 45, 64, 64):
            raise RuntimeError('{} shape {}'.format(name, tuple(prediction.shape)))
    heads = [predictions[name].view(batch, 9, 5, 64, 64) for name in ('stage2', 'stage3', 'stage4')]
    anchors = torch.tensor(
        [[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129],
         [96, 215], [78, 84], [37, 41]], dtype=torch.float32, device='cuda'
    )
    gt_xyxy = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)
    loss_fn = three_head_yolo_loss_stage2_cls_half if args.variant.endswith('stage2cls05') else three_head_yolo_loss
    geo_loss, cls_loss = loss_fn(*heads, gt_xyxy, anchors, 1024)
    loss = geo_loss + cls_loss
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite E7 loss')
    loss.backward()
    module = model.module
    checks = ('det_head_stage2', 'det_head_stage3', 'det_head_stage4')
    if any(not torch.count_nonzero(getattr(module, name).weight.grad) for name in checks):
        raise RuntimeError('one or more E7 independent heads have zero gradient')
    if not torch.count_nonzero(module.cvopm_stage2.proj_out.weight.grad):
        raise RuntimeError('stage2 zero-init proj_out did not receive a gradient')
    print('trogeo_3scale variant={} batch={} loss={:.8f} total_params={} peak_mib={:.1f}'.format(
        args.variant, batch, loss.item(), sum(parameter.numel() for parameter in model.parameters()),
        torch.cuda.max_memory_allocated() / 1024 / 1024
    ))


if __name__ == '__main__':
    main()
