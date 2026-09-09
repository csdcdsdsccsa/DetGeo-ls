"""Two-step GPU sanity checks for strict E4 two-scale PGCA variants."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.multiscale_detection_loss import two_head_yolo_loss


VARIANTS = (
    'h2_ind_pgca_c_direct',
    'h2_ind_pgca_a_conv',
    'h2_ind_pgca_b_dynamic',
)


def has_gradient(parameter):
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def loss_for(predictions, batch, anchors, target):
    p3 = predictions['stage3'].view(batch, 9, 5, 64, 64)
    p4 = predictions['stage4'].view(batch, 9, 5, 64, 64)
    geo, cls = two_head_yolo_loss(p3, p4, target, anchors, 1024)
    return geo + cls


def check_variant(variant, batch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant=variant)).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.rand(batch, 256, 256, device='cuda')
    anchors = torch.tensor(
        [[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129],
         [96, 215], [78, 84], [37, 41]], dtype=torch.float32, device='cuda')
    target = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)

    predictions, _ = model(query, reference, click)
    module = model.module
    if set(predictions) != {'stage3', 'stage4'}:
        raise RuntimeError('{} must be strict two-scale, got {}'.format(variant, set(predictions)))
    if any(value.shape != (batch, 45, 64, 64) for value in predictions.values()):
        raise RuntimeError('{} emitted invalid prediction shape'.format(variant))
    for name in ('cvopm_stage2', 'det_head_stage2'):
        if hasattr(module, name):
            raise RuntimeError('{} unexpectedly owns {}'.format(variant, name))

    first_loss = loss_for(predictions, batch, anchors, target)
    if not torch.isfinite(first_loss):
        raise RuntimeError('{} has non-finite first loss'.format(variant))
    first_loss.backward()
    for name in ('det_head_stage3', 'det_head_stage4'):
        if not has_gradient(getattr(module, name).weight):
            raise RuntimeError('{} {} has zero first-step gradient'.format(variant, name))
    for name in ('cvopm_stage3', 'cvopm_stage4'):
        if not has_gradient(getattr(module, name).proj_out.weight):
            raise RuntimeError('{} {}.proj_out has zero first-step gradient'.format(variant, name))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # CVOPM proj_out is zero-initialized, so PGCA receives gradients after the
    # first update opens that residual path.  Check its gradients on step two.
    predictions, _ = model(query, reference, click)
    second_loss = loss_for(predictions, batch, anchors, target)
    if not torch.isfinite(second_loss):
        raise RuntimeError('{} has non-finite second loss'.format(variant))
    second_loss.backward()
    if variant == 'h2_ind_pgca_c_direct':
        gradient_names = ('lambda3', 'lambda4')
        gradients = (module.lambda3, module.lambda4)
    elif variant == 'h2_ind_pgca_a_conv':
        gradient_names = ('pe_bias3', 'pe_bias4', 'lambda3', 'lambda4')
        gradients = (module.pe_bias3.weight, module.pe_bias4.weight, module.lambda3, module.lambda4)
    else:
        gradient_names = ('pe_bias3', 'pe_bias4', 'lambda_predictor3', 'lambda_predictor4')
        gradients = (module.pe_bias3.weight, module.pe_bias4.weight,
                     module.lambda_predictor3[0].weight, module.lambda_predictor4[0].weight)
    for name, parameter in zip(gradient_names, gradients):
        if not has_gradient(parameter):
            raise RuntimeError('{} {} has zero second-step gradient'.format(variant, name))

    print('pgca_2scale_sanity variant={} loss1={:.6f} loss2={:.6f} params={} peak_mib={:.1f}'.format(
        variant, first_loss.item(), second_loss.item(), sum(p.numel() for p in model.parameters()),
        torch.cuda.max_memory_allocated() / 1024 / 1024), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=7)
    args = parser.parse_args()
    for variant in VARIANTS:
        check_variant(variant, args.batch_size)


if __name__ == '__main__':
    main()
