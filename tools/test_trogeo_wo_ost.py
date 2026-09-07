"""Single-GPU forward/backward memory check for TROGeo-w/o-OST."""

import argparse

import torch

from model.TROGeo_wo_ost import TROGeoWoOST
from model.loss import build_target, yolo_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = torch.nn.DataParallel(TROGeoWoOST()).cuda().train()
    batch = args.batch_size
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.randn(batch, 256, 256, device='cuda')
    assert hasattr(model.module, 'encoder') and not hasattr(model.module, 'query_model') and not hasattr(model.module, 'reference_model')
    with torch.no_grad():
        query_input = model.module.position_embedding(torch.cat((query, click.unsqueeze(1)), dim=1))
        query_features = model.module.encoder(query_input)
        reference_features = model.module.encoder(reference)
        identity_output = model.module.cvopm(reference_features, context=query_features.flatten(2).transpose(1, 2))
        identity_diff = (identity_output - reference_features).abs().max()
    if identity_diff.item() != 0.0:
        raise RuntimeError('CVOPM zero-init residual check failed: {}'.format(identity_diff.item()))
    del query_input, query_features, reference_features, identity_output
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
    if not torch.count_nonzero(model.module.cvopm.transformer_blocks[0].attn1.to_q.weight.grad):
        raise RuntimeError('CVOPM gradient is zero')
    print('trogeo_wo_ost batch={} loss={:.8f} total_params={} cvopm_identity_max_abs={} peak_mib={:.1f}'.format(
        batch, loss.item(), sum(parameter.numel() for parameter in model.parameters()), identity_diff.item(),
        torch.cuda.max_memory_allocated() / 1024 / 1024
    ))


if __name__ == '__main__':
    main()
