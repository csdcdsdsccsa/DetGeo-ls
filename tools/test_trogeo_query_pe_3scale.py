"""Two-step GPU sanity checks for all E7 Query-PE/LE variants."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.multiscale_detection_loss import three_head_yolo_loss


VARIANTS = (
    'h2_ind_3scale_pe_ln_amp',
    'h2_ind_3scale_pe_all_add',
    'h2_ind_3scale_pe_all_key',
    'h2_ind_3scale_le_ind_res',
    'h2_ind_3scale_le_stage2_res',
)


def has_gradient(parameter):
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def loss_for(predictions, batch, anchors, target):
    heads = [predictions[name].view(batch, 9, 5, 64, 64) for name in ('stage2', 'stage3', 'stage4')]
    geo, cls = three_head_yolo_loss(*heads, target, anchors, 1024)
    return geo + cls


def check_variant(variant, batch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant=variant)).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.randn(batch, 256, 256, device='cuda')
    anchors = torch.tensor(
        [[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129],
         [96, 215], [78, 84], [37, 41]], dtype=torch.float32, device='cuda'
    )
    target = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)

    predictions, _ = model(query, reference, click)
    if set(predictions) != {'stage2', 'stage3', 'stage4'}:
        raise RuntimeError('{} returned {}'.format(variant, set(predictions)))
    if any(value.shape != (batch, 45, 64, 64) for value in predictions.values()):
        raise RuntimeError('{} emitted invalid prediction shape'.format(variant))
    first_loss = loss_for(predictions, batch, anchors, target)
    if not torch.isfinite(first_loss):
        raise RuntimeError('{} has non-finite first loss'.format(variant))
    first_loss.backward()
    module = model.module
    for name in ('det_head_stage2', 'det_head_stage3', 'det_head_stage4'):
        if not has_gradient(getattr(module, name).weight):
            raise RuntimeError('{} {} has zero first-step gradient'.format(variant, name))
    for name in ('cvopm_stage2', 'cvopm_stage3', 'cvopm_stage4'):
        if not has_gradient(getattr(module, name).proj_out.weight):
            raise RuntimeError('{} {}.proj_out has zero first-step gradient'.format(variant, name))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # proj_out is zero-initialized, so PE/LE paths receive their first nonzero
    # gradients only after the output projections' first optimizer update.
    second_predictions, _ = model(query, reference, click)
    second_loss = loss_for(second_predictions, batch, anchors, target)
    second_loss.backward()
    if variant.startswith('h2_ind_3scale_pe_'):
        for name in ('pe_proj2', 'pe_proj3', 'pe_proj4', 'pe_alpha2', 'pe_alpha3', 'pe_alpha4'):
            parameter = getattr(module, name)[0].weight if name.startswith('pe_proj') else getattr(module, name)
            if not has_gradient(parameter):
                raise RuntimeError('{} {} has zero second-step gradient'.format(variant, name))
    elif variant == 'h2_ind_3scale_le_ind_res':
        for name in ('le_stage2', 'le_stage3', 'le_stage4', 'le_beta2_logit', 'le_beta3_logit', 'le_beta4_logit'):
            parameter = getattr(module, name)[0].weight if name.startswith('le_stage') else getattr(module, name)
            if not has_gradient(parameter):
                raise RuntimeError('{} {} has zero second-step gradient'.format(variant, name))
    else:
        if hasattr(module, 'le_stage3') or hasattr(module, 'le_stage4'):
            raise RuntimeError('{} must only own le_stage2'.format(variant))
        for name in ('le_stage2', 'le_beta2_logit', 'le_beta3_logit', 'le_beta4_logit'):
            parameter = getattr(module, name)[0].weight if name == 'le_stage2' else getattr(module, name)
            if not has_gradient(parameter):
                raise RuntimeError('{} {} has zero second-step gradient'.format(variant, name))
    print('query_pe_3scale_sanity variant={} loss1={:.6f} loss2={:.6f} params={} peak_mib={:.1f}'.format(
        variant, first_loss.item(), second_loss.item(), sum(p.numel() for p in model.parameters()),
        torch.cuda.max_memory_allocated() / 1024 / 1024
    ), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=7)
    args = parser.parse_args()
    for variant in VARIANTS:
        check_variant(variant, args.batch_size)


if __name__ == '__main__':
    main()
