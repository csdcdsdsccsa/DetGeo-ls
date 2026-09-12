"""Two-step GPU sanity check for P1: A0 h2_ind plus PQRA."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.multiscale_detection_loss import two_head_yolo_loss


ANCHORS = torch.tensor(
    [[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129],
     [96, 215], [78, 84], [37, 41]], dtype=torch.float32)


def has_gradient(parameter):
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def two_head_loss(predictions, target):
    batch = target.shape[0]
    p3 = predictions['stage3'].view(batch, 9, 5, 64, 64)
    p4 = predictions['stage4'].view(batch, 9, 5, 64, 64)
    geo, cls = two_head_yolo_loss(p3, p4, target, ANCHORS.to(target.device), 1024)
    return geo + cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required')
    torch.manual_seed(2024)
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant='h2_ind_pqra')).cuda().train()
    module = model.module
    for refine, tokens in ((module.query_refine3, (256, 384)), (module.query_refine4, (64, 768))):
        query = torch.randn(args.batch_size, *tokens, device='cuda')
        position = torch.randn_like(query)
        with torch.no_grad():
            refined, _ = refine(query, position)
        if not torch.equal(refined, query) or refine.alpha.item() != 0.0:
            raise RuntimeError('PQRA must be an exact identity at zero alpha')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    torch.manual_seed(100)
    query = torch.randn(args.batch_size, 3, 256, 256, device='cuda')
    reference = torch.randn(args.batch_size, 3, 1024, 1024, device='cuda')
    click = torch.rand(args.batch_size, 256, 256, device='cuda')
    target = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(args.batch_size, 1)
    predictions, _ = model(query, reference, click)
    if set(predictions) != {'stage3', 'stage4'} or any(value.shape != (args.batch_size, 45, 64, 64)
                                                     for value in predictions.values()):
        raise RuntimeError('PQRA must preserve the A0 dual-head prediction contract')
    loss1 = two_head_loss(predictions, target)
    if not torch.isfinite(loss1):
        raise RuntimeError('PQRA first loss is non-finite')
    loss1.backward()
    # Direct-CA's proj_out is intentionally zero initialized, so its output is
    # initially independent of Query context.  Step 1 activates Direct-CA;
    # PQRA alpha receives gradient on step 2 and its inner attention on step 3.
    for cvopm in (module.cvopm_stage3, module.cvopm_stage4):
        if not has_gradient(cvopm.proj_out.weight):
            raise RuntimeError('Direct-CA proj_out has zero first-step gradient')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    predictions, _ = model(query, reference, click)
    loss2 = two_head_loss(predictions, target)
    if not torch.isfinite(loss2):
        raise RuntimeError('PQRA second loss is non-finite')
    loss2.backward()
    for refine in (module.query_refine3, module.query_refine4):
        if not has_gradient(refine.alpha):
            raise RuntimeError('PQRA alpha has zero second-step gradient')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    predictions, _ = model(query, reference, click)
    loss3 = two_head_loss(predictions, target)
    if not torch.isfinite(loss3):
        raise RuntimeError('PQRA third loss is non-finite')
    loss3.backward()
    for refine, position_mlp in ((module.query_refine3, module.pqra_pos_proj3),
                                 (module.query_refine4, module.pqra_pos_proj4)):
        for projection in (refine.attn.to_q, refine.attn.to_k, refine.attn.to_v):
            if not has_gradient(projection.weight):
                raise RuntimeError('PQRA attention has zero second-step gradient')
        if not has_gradient(position_mlp[-1].weight):
            raise RuntimeError('PQRA position MLP has zero second-step gradient')
    print('e4_pqra_sanity batch={} loss1={:.6f} loss2={:.6f} loss3={:.6f} params={} peak_mib={:.1f}'.format(
        args.batch_size, loss1.item(), loss2.item(), loss3.item(), sum(p.numel() for p in model.parameters()),
        torch.cuda.max_memory_allocated() / 1024 / 1024), flush=True)


if __name__ == '__main__':
    main()
