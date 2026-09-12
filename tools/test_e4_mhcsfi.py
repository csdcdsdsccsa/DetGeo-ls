"""Focused GPU sanity checks for the A3-Bi MH-CSFI variants."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def has_gradient(parameter):
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def check_identity_and_two_steps(variant):
    model = TROGeoMSDetectionAblation(variant=variant).cuda().train()
    module = model.mh_cross_scale_interaction
    z3 = torch.randn(2, 384, 64, 64, device='cuda', requires_grad=True)
    z4 = torch.randn(2, 768, 32, 32, device='cuda', requires_grad=True)
    with torch.no_grad():
        out3, out4, diagnostics = module(z3, z4)
    if not torch.equal(out3, z3) or not torch.equal(out4, z4):
        raise RuntimeError('{} must be an exact identity at zero alpha'.format(variant))
    if module.alpha3.item() != 0.0 or module.alpha4.item() != 0.0:
        raise RuntimeError('{} alpha values are not zero initialized'.format(variant))
    expected_shapes = ((2, 1, 64, 64), (2, 1, 32, 32), (2, 3), (2, 3))
    actual_shapes = (tuple(diagnostics['mask3'].shape), tuple(diagnostics['mask4'].shape),
                     tuple(diagnostics['weights3'].shape), tuple(diagnostics['weights4'].shape))
    if actual_shapes != expected_shapes:
        raise RuntimeError('{} invalid MH-CSFI diagnostic shapes {}'.format(variant, actual_shapes))
    for name in ('mask3', 'mask4'):
        if not torch.all((diagnostics[name] >= 0) & (diagnostics[name] <= 1)):
            raise RuntimeError('{} {} is outside [0, 1]'.format(variant, name))
    for name in ('weights3', 'weights4'):
        if not torch.allclose(diagnostics[name].sum(1), torch.ones(2, device='cuda'), atol=1e-6):
            raise RuntimeError('{} {} does not sum to one'.format(variant, name))
    if variant == 'h2_ind_amhcsfi_bi':
        equal = torch.full((2, 3), 1.0 / 3.0, device='cuda')
        if not torch.allclose(diagnostics['weights3'], equal, atol=1e-6) or \
                not torch.allclose(diagnostics['weights4'], equal, atol=1e-6):
            raise RuntimeError('adaptive MH-CSFI must start at exactly equal scale weights')

    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
    out3, out4, _ = module(z3, z4)
    loss1 = out3.square().mean() + out4.square().mean()
    loss1.backward()
    for name in ('alpha3', 'alpha4'):
        if not has_gradient(getattr(module, name)):
            raise RuntimeError('{} {} has zero first-step gradient'.format(variant, name))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    z3b = torch.randn(2, 384, 64, 64, device='cuda')
    z4b = torch.randn(2, 768, 32, 32, device='cuda')
    out3, out4, _ = module(z3b, z4b)
    loss2 = out3.square().mean() + out4.square().mean()
    loss2.backward()
    for layer in (module.proj_4to3[0], module.proj_3to4[0], module.gate3[0], module.gate4[0],
                  module.relation_mask3.relation_encoder[0], module.relation_mask3.head3[0],
                  module.relation_mask4.relation_encoder[0], module.relation_mask4.head5[0]):
        if not has_gradient(layer.weight):
            raise RuntimeError('{} has zero second-step MH-CSFI gradient'.format(variant))
    if variant == 'h2_ind_amhcsfi_bi' and not has_gradient(module.relation_mask3.scale_selector[-1].weight):
        raise RuntimeError('adaptive MH-CSFI selector has zero second-step gradient')
    print('{} passed: loss1={:.6f} loss2={:.6f} w3={} w4={}'.format(
        variant, loss1.item(), loss2.item(), diagnostics['weights3'].mean(0).tolist(),
        diagnostics['weights4'].mean(0).tolist()), flush=True)
    del model
    torch.cuda.empty_cache()


def check_model_contract(variant, batch_size):
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant=variant)).cuda().eval()
    module = model.module
    if hasattr(module, 'cvopm_stage2') or hasattr(module, 'det_head_stage2') or hasattr(module, 'cvopm_stage4_second'):
        raise RuntimeError('{} changed the strict A3-Bi module topology'.format(variant))
    with torch.no_grad():
        predictions, _ = model(torch.randn(batch_size, 3, 256, 256, device='cuda'),
                               torch.randn(batch_size, 3, 1024, 1024, device='cuda'),
                               torch.rand(batch_size, 256, 256, device='cuda'))
    expected = {'stage3', 'stage4', 'coarse_logits', 'fine_logits'}
    if set(predictions) != expected or predictions['stage3'].shape != (batch_size, 45, 64, 64) or \
            predictions['stage4'].shape != (batch_size, 45, 64, 64) or \
            predictions['coarse_logits'].shape != (batch_size, 1, 32, 32) or \
            predictions['fine_logits'].shape != (batch_size, 1, 64, 64):
        raise RuntimeError('{} violates the A3-Bi two-head/guidance contract'.format(variant))
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required')
    for variant in ('h2_ind_mhcsfi_bi', 'h2_ind_amhcsfi_bi'):
        check_identity_and_two_steps(variant)
        check_model_contract(variant, args.batch_size)
    print('MH-CSFI sanity passed.', flush=True)


if __name__ == '__main__':
    main()
