"""Single-GPU structural, forward and backward check for TROGeo Direct-CA."""

import argparse

import torch

from model.TROGeo_wo_ost import TROGeoWoOST
from model.loss import build_target, yolo_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--backbone', choices=('swin_s', 'swin_t', 'resnet50'), default='swin_s')
    args = parser.parse_args()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = torch.nn.DataParallel(
        TROGeoWoOST(use_satellite_self_attention=False, backbone=args.backbone)
    ).cuda().train()
    block = model.module.cvopm.transformer_blocks[0]
    assert block.attn1 is None and block.norm1 is None
    assert block.attn2 is not None
    assert not any(
        name.startswith('cvopm.transformer_blocks.') and ('.attn1.' in name or '.norm1.' in name)
        for name, _ in model.module.named_parameters()
    )
    batch = args.batch_size
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.randn(batch, 256, 256, device='cuda')
    with torch.no_grad():
        query_input = model.module.position_embedding(torch.cat((query, click.unsqueeze(1)), dim=1))
        query_raw_features, query_features = model.module.encoder(query_input)
        reference_raw_features, reference_features = model.module.encoder(reference)
        context = query_features.flatten(2).transpose(1, 2)
        identity_output = model.module.cvopm(reference_features, context=context)
        identity_diff = (identity_output - reference_features).abs().max()
        query_raw_shape = tuple(query_raw_features.shape)
        query_shape = tuple(query_features.shape)
        reference_raw_shape = tuple(reference_raw_features.shape)
        reference_shape = tuple(reference_features.shape)
        q_shape = (batch, reference_features.shape[-2] * reference_features.shape[-1], reference_features.shape[1])
        kv_shape = tuple(context.shape)
    if identity_diff.item() != 0.0:
        raise RuntimeError('CVOPM zero-init residual check failed: {}'.format(identity_diff.item()))
    del query_input, query_features, reference_features, context, identity_output
    torch.cuda.empty_cache()
    outbox, _ = model(query, reference, click)
    predictions = outbox.view(batch, 9, 5, 64, 64)
    anchors = torch.tensor([[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129], [96, 215], [78, 84], [37, 41]], dtype=torch.float32, device='cuda')
    gt_xyxy = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)
    target, anchor_index = build_target(gt_xyxy, anchors, 1024, 64)
    geo_loss, cls_loss = yolo_loss(predictions, target, anchors, anchor_index, 1024)
    loss = geo_loss + cls_loss
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite detection loss')
    loss.backward()
    if not torch.count_nonzero(model.module.cvopm.proj_out.weight.grad):
        raise RuntimeError('zero-init CVOPM proj_out did not receive a gradient')
    print('trogeo_direct_ca backbone={} batch={} loss={:.8f} params={} Fq_raw={} Fq={} Fr_raw={} Fr={} Q={} KV={} '
          'outbox={} identity_max_abs={} peak_mib={:.1f}'.format(
        args.backbone, batch, loss.item(), sum(parameter.numel() for parameter in model.parameters()),
        query_raw_shape, query_shape, reference_raw_shape, reference_shape, q_shape, kv_shape, tuple(outbox.shape), identity_diff.item(),
        torch.cuda.max_memory_allocated() / 1024 / 1024
    ))


if __name__ == '__main__':
    main()
