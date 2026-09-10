"""Two-step GPU sanity test for E4 original-DetGeo-PE plus PGCA-B."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.multiscale_detection_loss import two_head_yolo_loss


def has_gradient(parameter):
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def loss_for(predictions, batch, anchors, target):
    p3 = predictions['stage3'].view(batch, 9, 5, 64, 64)
    p4 = predictions['stage4'].view(batch, 9, 5, 64, 64)
    geo, cls = two_head_yolo_loss(p3, p4, target, anchors, 1024)
    return geo + cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=7)
    args = parser.parse_args()
    batch = args.batch_size
    variant = 'h2_ind_pgca_b_dynamic'
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(
        variant=variant, position_mode='detgeo')).cuda().train()
    module = model.module
    if type(module.position_embedding).__name__ != 'DetGeoPositionEmbedding':
        raise RuntimeError('expected original DetGeo position encoder')
    if hasattr(module, 'cvopm_stage2') or hasattr(module, 'det_head_stage2'):
        raise RuntimeError('E4 PGCA-B must remain strict two-scale')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.rand(batch, 256, 256, device='cuda')
    anchors = torch.tensor(
        [[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129],
         [96, 215], [78, 84], [37, 41]], dtype=torch.float32, device='cuda')
    target = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)

    predictions, _ = model(query, reference, click)
    if set(predictions) != {'stage3', 'stage4'} or any(
            value.shape != (batch, 45, 64, 64) for value in predictions.values()):
        raise RuntimeError('invalid E4 two-head prediction layout')
    loss1 = loss_for(predictions, batch, anchors, target)
    if not torch.isfinite(loss1):
        raise RuntimeError('non-finite first loss')
    loss1.backward()
    if not has_gradient(module.cvopm_stage3.proj_out.weight) or not has_gradient(module.cvopm_stage4.proj_out.weight):
        raise RuntimeError('zero-init CVOPM projection did not receive first-step gradient')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    predictions, _ = model(query, reference, click)
    loss2 = loss_for(predictions, batch, anchors, target)
    if not torch.isfinite(loss2):
        raise RuntimeError('non-finite second loss')
    loss2.backward()
    for name, parameter in (
            ('pe_bias3', module.pe_bias3.weight),
            ('pe_bias4', module.pe_bias4.weight),
            ('lambda_predictor3', module.lambda_predictor3[0].weight),
            ('lambda_predictor4', module.lambda_predictor4[0].weight)):
        if not has_gradient(parameter):
            raise RuntimeError('{} has zero second-step gradient'.format(name))
    print('e4_detgeo_pe_pgca_b_sanity loss1={:.6f} loss2={:.6f} params={}'.format(
        loss1.item(), loss2.item(), sum(p.numel() for p in model.parameters())), flush=True)


if __name__ == '__main__':
    main()
