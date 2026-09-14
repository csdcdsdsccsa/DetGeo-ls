"""AMR-PE identity, range, and ordinary-RNG fairness checks."""

import torch

from model.TROGeo_ms_detection_ablation import (
    AdaptiveMultiRangePositionField,
    TROGeoMSDetectionAblation,
)


def check_field(mode):
    torch.manual_seed(31)
    field = AdaptiveMultiRangePositionField(mode=mode)
    query = torch.randn(2, 3, 64, 64)
    distance = torch.rand(2, 64, 64)
    output, weights, multi, alpha = field(query, distance)
    if not torch.equal(output, distance) or alpha.item() != 0.0:
        raise RuntimeError('{} AMR-PE must be an exact identity at gamma=0'.format(mode))
    expected = torch.full_like(weights, 1.0 / 3.0)
    if not torch.allclose(weights, expected, atol=1e-7, rtol=0.0):
        raise RuntimeError('{} AMR-PE must start from uniform weights'.format(mode))
    d = distance.clamp(0.0, 1.0)
    if not torch.all(d.square() <= d) or not torch.all(d <= torch.sqrt(d.clamp_min(1e-6))):
        raise RuntimeError('AMR-PE range ordering D^2 <= D <= sqrt(D) failed')
    with torch.no_grad():
        field.gamma.fill_(1.0)
    changed, _, _, _ = field(query, distance)
    if torch.equal(changed, distance):
        raise RuntimeError('{} AMR-PE did not change the field after opening gamma'.format(mode))
    if multi.shape != distance.shape or changed.shape != distance.shape:
        raise RuntimeError('{} AMR-PE has an invalid field shape'.format(mode))


def check_rng_and_shared_initialization():
    kwargs = dict(backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
                  position_mode='detgeo', dadpe_mode='none')
    torch.manual_seed(2026)
    baseline = TROGeoMSDetectionAblation(**kwargs, amr_pe_mode='none')
    baseline_state = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
    baseline_rng = torch.get_rng_state().clone()

    torch.manual_seed(2026)
    fixed = TROGeoMSDetectionAblation(**kwargs, amr_pe_mode='fixed')
    fixed_state = {key: value.detach().clone() for key, value in fixed.state_dict().items()}
    fixed_rng = torch.get_rng_state().clone()

    torch.manual_seed(2026)
    adaptive = TROGeoMSDetectionAblation(**kwargs, amr_pe_mode='adaptive')
    adaptive_state = {key: value.detach().clone() for key, value in adaptive.state_dict().items()}
    adaptive_rng = torch.get_rng_state().clone()

    shared_keys = [key for key in baseline_state if not key.startswith('amr_position_field.')]
    if not all(torch.equal(baseline_state[key], adaptive_state[key]) for key in shared_keys):
        raise RuntimeError('AMR-PE changed a shared Bi-Res initialization tensor')
    if not torch.equal(fixed_rng, adaptive_rng):
        raise RuntimeError('fixed/adaptive AMR-PE must consume the same RNG trajectory')
    if torch.equal(baseline_rng, adaptive_rng):
        raise RuntimeError('AMR-PE must naturally consume RNG under --standard_rng')
    if fixed_state.keys() != adaptive_state.keys() or not all(
            torch.equal(fixed_state[key], adaptive_state[key]) for key in fixed_state):
        raise RuntimeError('fixed/adaptive AMR-PE must have identical initialized parameter tensors')


def check_bires_zero_start_forward():
    if not torch.cuda.is_available():
        return
    kwargs = dict(backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
                  position_mode='detgeo', dadpe_mode='none')
    torch.manual_seed(77)
    baseline = TROGeoMSDetectionAblation(**kwargs, amr_pe_mode='none').cuda().eval()
    torch.manual_seed(77)
    adaptive = TROGeoMSDetectionAblation(**kwargs, amr_pe_mode='adaptive').cuda().eval()
    query = torch.randn(1, 3, 256, 256, device='cuda')
    reference = torch.randn(1, 3, 1024, 1024, device='cuda')
    click = torch.zeros(1, 256, 256, device='cuda')
    click[:, 128, 128] = 1.0
    with torch.no_grad():
        baseline_predictions, _ = baseline(query, reference, click)
        adaptive_predictions, _ = adaptive(query, reference, click)
    if baseline_predictions.keys() != adaptive_predictions.keys() or not all(
            torch.equal(baseline_predictions[key], adaptive_predictions[key])
            for key in baseline_predictions):
        raise RuntimeError('AMR-PE gamma=0 must exactly recover Bi-Res + DetGeo-PE outputs')


if __name__ == '__main__':
    check_field('fixed')
    check_field('adaptive')
    check_rng_and_shared_initialization()
    check_bires_zero_start_forward()
    print('AMR-PE checks passed')
