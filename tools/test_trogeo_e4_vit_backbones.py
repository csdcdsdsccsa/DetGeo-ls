"""Forward/backward and shape sanity for strict E4 ViT-T/ViT-S backbones."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=('vit_t', 'vit_s'), required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone=args.backbone, variant='h2_ind'
    ).cuda().train()
    query = torch.randn(args.batch_size, 3, 256, 256, device='cuda')
    satellite = torch.randn(args.batch_size, 3, 1024, 1024, device='cuda')
    click = torch.rand(args.batch_size, 256, 256, device='cuda')
    predictions, _ = model(query, satellite, click)
    assert predictions['stage3'].shape == (args.batch_size, 45, 64, 64)
    assert predictions['stage4'].shape == (args.batch_size, 45, 64, 64)
    loss = predictions['stage3'].square().mean() + predictions['stage4'].square().mean()
    loss.backward()
    grad = model.encoder.stage3_projection.weight.grad
    if grad is None or not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
        raise RuntimeError('ViT E4 adapter did not receive a finite nonzero gradient')
    print('{} sanity passed; loss={:.6f}; params={}'.format(
        args.backbone, loss.item(), sum(parameter.numel() for parameter in model.parameters())
    ))


if __name__ == '__main__':
    main()
