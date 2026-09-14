"""DADPE geometry, natural-RNG, and two-step gradient checks."""

import torch

from model.TROGeo_ms_detection_ablation import (
    InputDirectionResidual, MultiScaleDirectionResidual, TROGeoMSDetectionAblation,
    build_directional_geometry,
)


def has_gradient(parameter):
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def check_geometry():
    distance = torch.zeros(1, 256, 256)
    y, x = 100, 80
    distance[0, y, x] = 1.0
    geometry = build_directional_geometry(distance)
    assert geometry.shape == (1, 3, 256, 256)
    assert geometry[0, 0, y, x].item() == 1.0
    assert geometry[0, 1, y, x].item() == 0.0 and geometry[0, 2, y, x].item() == 0.0
    assert geometry[0, 1, y, x + 10].item() > 0 and geometry[0, 1, y, x - 10].item() < 0
    assert geometry[0, 2, y + 10, x].item() > 0 and geometry[0, 2, y - 10, x].item() < 0


def check_natural_rng_consumption():
    for variant in ('h2_ind_fg_amhcsfi_res', 'h2_ind_amhcsfi_res_bi'):
        states = {}
        models = {}
        for mode in ('none', 'input', 'multiscale'):
            torch.manual_seed(2026)
            models[mode] = TROGeoMSDetectionAblation(
                backbone='swin_t', variant=variant, position_mode='detgeo', dadpe_mode=mode)
            states[mode] = torch.get_rng_state().clone()
        if torch.equal(states['none'], states['input']) or torch.equal(states['input'], states['multiscale']):
            raise RuntimeError('{}: DADPE construction did not naturally advance the CPU RNG state'.format(variant))
        if hasattr(models['none'], 'input_direction_residual'):
            raise RuntimeError('{}: dadpe_mode=none unexpectedly created DADPE modules'.format(variant))
        if not hasattr(models['input'], 'input_direction_residual'):
            raise RuntimeError('{}: dadpe_mode=input did not create the input residual'.format(variant))
        if not hasattr(models['multiscale'], 'multiscale_direction_residual'):
            raise RuntimeError('{}: dadpe_mode=multiscale did not create the multiscale residual'.format(variant))


def check_bires_forward():
    if not torch.cuda.is_available():
        return
    batch = 1
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.zeros(batch, 256, 256, device='cuda')
    click[:, 128, 128] = 1.0
    expected = {'stage3', 'stage4', 'coarse_logits', 'fine_logits'}
    for mode in ('input', 'multiscale'):
        model = TROGeoMSDetectionAblation(
            backbone='swin_t', variant='h2_ind_amhcsfi_res_bi', position_mode='detgeo', dadpe_mode=mode).cuda().eval()
        with torch.no_grad():
            predictions, _ = model(query, reference, click)
        if set(predictions) != expected:
            raise RuntimeError('{}: invalid Bi-Res prediction keys {}'.format(mode, sorted(predictions)))
        for key in ('stage3', 'stage4'):
            if predictions[key].shape != (batch, 45, 64, 64):
                raise RuntimeError('{}: {} has shape {}'.format(mode, key, tuple(predictions[key].shape)))
        if predictions['coarse_logits'].shape != (batch, 1, 32, 32):
            raise RuntimeError('{}: coarse_logits has shape {}'.format(mode, tuple(predictions['coarse_logits'].shape)))
        if predictions['fine_logits'].shape != (batch, 1, 64, 64):
            raise RuntimeError('{}: fine_logits has shape {}'.format(mode, tuple(predictions['fine_logits'].shape)))
        if hasattr(model, 'cvopm_stage4_second'):
            raise RuntimeError('{}: Bi-Res must reuse cvopm_stage4'.format(mode))
        del model
        torch.cuda.empty_cache()


def check_zero_residuals_and_two_steps():
    torch.manual_seed(9)
    geometry = torch.randn(2, 3, 32, 32)
    base = torch.randn(2, 3, 32, 32)
    input_residual = InputDirectionResidual().train()
    input_out, _ = input_residual(base, geometry)
    if not torch.equal(input_out, base):
        raise RuntimeError('gamma=0 must make the B input residual an exact identity')

    q3, q4 = torch.randn(2, 384, 16, 16), torch.randn(2, 768, 8, 8)
    multiscale = MultiScaleDirectionResidual().train()
    q3_out, q4_out, _, _ = multiscale(q3, q4, geometry)
    if not torch.equal(q3_out, q3) or not torch.equal(q4_out, q4):
        raise RuntimeError('beta3=beta4=0 must make the D residuals exact identities')

    optimizer = torch.optim.SGD(list(input_residual.parameters()) + list(multiscale.parameters()), lr=1e-2)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        input_out, _ = input_residual(base, geometry)
        q3_out, q4_out, _, _ = multiscale(q3, q4, geometry)
        (input_out.square().mean() + q3_out.square().mean() + q4_out.square().mean()).backward()
        if step == 0:
            for parameter in (input_residual.gamma, multiscale.beta3, multiscale.beta4):
                if not has_gradient(parameter):
                    raise RuntimeError('zero gate lacks first-step gradient')
        else:
            for layer in (input_residual.encoder[0], multiscale.stage3_encoder[0], multiscale.stage4_encoder[0]):
                if not has_gradient(layer.weight):
                    raise RuntimeError('DADPE encoder lacks second-step gradient')
        optimizer.step()


if __name__ == '__main__':
    check_geometry()
    check_natural_rng_consumption()
    check_zero_residuals_and_two_steps()
    check_bires_forward()
    print('DADPE checks passed')
