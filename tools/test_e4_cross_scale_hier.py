"""Two-step GPU checks for the first E4 cross-scale collaboration ablation."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.multiscale_detection_loss import two_head_yolo_loss


ANCHORS = torch.tensor(
    [[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129],
     [96, 215], [78, 84], [37, 41]], dtype=torch.float32)


def has_gradient(parameter):
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def assert_base_state_matches(baseline, csfi):
    baseline_state = baseline.state_dict()
    csfi_state = csfi.state_dict()
    base_keys = [key for key in baseline_state if not key.startswith('cross_scale_interaction.')]
    mismatched = [key for key in base_keys if not torch.equal(baseline_state[key], csfi_state[key])]
    if mismatched:
        raise RuntimeError('E4 base initialization mismatch: {}'.format(', '.join(mismatched[:5])))


def two_head_loss(predictions, target):
    batch = target.shape[0]
    p3 = predictions['stage3'].view(batch, 9, 5, 64, 64)
    p4 = predictions['stage4'].view(batch, 9, 5, 64, 64)
    geo, cls = two_head_yolo_loss(p3, p4, target, ANCHORS.to(target.device), 1024)
    return geo + cls


def check_zero_gate_equivalence(batch):
    torch.manual_seed(2024)
    baseline = TROGeoMSDetectionAblation(variant='h2_ind').cuda().eval()
    torch.manual_seed(2024)
    csfi = TROGeoMSDetectionAblation(variant='h2_ind_csfi').cuda().eval()
    assert_base_state_matches(baseline, csfi)

    torch.manual_seed(99)
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.rand(batch, 256, 256, device='cuda')
    with torch.no_grad():
        baseline_predictions, _ = baseline(query, reference, click)
        csfi_predictions, _ = csfi(query, reference, click)
    max_difference = max((baseline_predictions[name] - csfi_predictions[name]).abs().max().item()
                         for name in ('stage3', 'stage4'))
    if max_difference != 0.0:
        raise RuntimeError('zero-gate E4 equivalence failed: max_abs_diff={}'.format(max_difference))
    del baseline, csfi, baseline_predictions, csfi_predictions
    torch.cuda.empty_cache()
    return max_difference


def check_two_step_backward(batch):
    torch.manual_seed(2024)
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant='h2_ind_csfi')).cuda().train()
    module = model.module
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    torch.manual_seed(100)
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.rand(batch, 256, 256, device='cuda')
    target = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)

    predictions, _ = model(query, reference, click)
    if set(predictions) != {'stage3', 'stage4'} or any(value.shape != (batch, 45, 64, 64)
                                                         for value in predictions.values()):
        raise RuntimeError('CSFI must preserve E4 two-head prediction contract')
    if hasattr(module, 'cvopm_stage2') or hasattr(module, 'det_head_stage2'):
        raise RuntimeError('CSFI must not create Stage2 detection modules')
    loss1 = two_head_loss(predictions, target)
    if not torch.isfinite(loss1):
        raise RuntimeError('CSFI first loss is non-finite')
    loss1.backward()
    for name in ('alpha3', 'alpha4'):
        if not has_gradient(getattr(module.cross_scale_interaction, name)):
            raise RuntimeError('CSFI {} has zero first-step gradient'.format(name))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    predictions, _ = model(query, reference, click)
    loss2 = two_head_loss(predictions, target)
    if not torch.isfinite(loss2):
        raise RuntimeError('CSFI second loss is non-finite')
    loss2.backward()
    for name in ('proj_4to3', 'proj_3to4', 'gate3', 'gate4'):
        if not has_gradient(getattr(module.cross_scale_interaction, name)[0].weight):
            raise RuntimeError('CSFI {} has zero second-step gradient'.format(name))
    return loss1.item(), loss2.item(), sum(parameter.numel() for parameter in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    difference = check_zero_gate_equivalence(args.batch_size)
    loss1, loss2, parameters = check_two_step_backward(args.batch_size)
    print('e4_csfi_sanity batch={} identity_max_abs={} loss1={:.6f} loss2={:.6f} '
          'params={} peak_mib={:.1f}'.format(
              args.batch_size, difference, loss1, loss2, parameters,
              torch.cuda.max_memory_allocated() / 1024 / 1024), flush=True)


if __name__ == '__main__':
    main()
