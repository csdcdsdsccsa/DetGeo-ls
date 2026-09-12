"""GPU checks for A3-Bi plus residual adaptive multi-receptive CSFI."""
import argparse
import torch
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def has_gradient(parameter):
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def check_variant(base_variant, variant, batch_size):
    torch.manual_seed(2024)
    base = TROGeoMSDetectionAblation(variant=base_variant).cuda().train()
    torch.manual_seed(2024)
    model = TROGeoMSDetectionAblation(variant=variant).cuda().train()
    base_state, new_state = base.state_dict(), model.state_dict()
    missing = [key for key in base_state if key not in new_state]
    changed = [key for key in base_state if key in new_state and not torch.equal(base_state[key], new_state[key])]
    if missing or changed:
        raise RuntimeError('{} public initialization diverged: missing={} changed={}'.format(
            variant, missing[:3], changed[:3]))
    refiner, csfi = model.amhcsfi_res_refiner, model.cross_scale_interaction
    if refiner.refine_beta3.item() != 0.0 or refiner.refine_beta4.item() != 0.0:
        raise RuntimeError('refine beta must start at zero')
    z3 = torch.randn(batch_size, 384, 64, 64, device='cuda')
    z4 = torch.randn(batch_size, 768, 32, 32, device='cuda')
    with torch.no_grad():
        base3, base4, _, _ = csfi(z3, z4)
        out3, out4, d = refiner(z3, z4, csfi)
    if not torch.equal(base3, out3) or not torch.equal(base4, out4):
        raise RuntimeError('{} beta=0 does not exactly reproduce original CSFI'.format(variant))
    if not d['modulation3'].eq(1.0).all() or not d['modulation4'].eq(1.0).all():
        raise RuntimeError('zero beta must produce exactly one modulation')
    equal = torch.full((batch_size, 3), 1.0 / 3.0, device='cuda')
    if tuple(d['mask3'].shape) != (batch_size, 1, 64, 64) or tuple(d['mask4'].shape) != (batch_size, 1, 32, 32) or \
            not torch.allclose(d['weights3'], equal) or not torch.allclose(d['weights4'], equal):
        raise RuntimeError('invalid initial mask or adaptive weights')
    optimizer = torch.optim.SGD(list(csfi.parameters()) + list(refiner.parameters()), lr=1e-3)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        a3 = torch.randn(batch_size, 384, 64, 64, device='cuda')
        a4 = torch.randn(batch_size, 768, 32, 32, device='cuda')
        y3, y4, _ = refiner(a3, a4, csfi)
        loss = y3.square().mean() + y4.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite loss at step {}'.format(step + 1))
        loss.backward()
        if step == 0 and not (has_gradient(csfi.alpha3) and has_gradient(csfi.alpha4)):
            raise RuntimeError('alpha lacks first-step gradient')
        if step == 1:
            parameters = (csfi.proj_4to3[0].weight, csfi.proj_3to4[0].weight, csfi.gate3[0].weight,
                          csfi.gate4[0].weight, refiner.refine_beta3, refiner.refine_beta4)
            if not all(has_gradient(value) for value in parameters):
                raise RuntimeError('base CSFI or beta lacks second-step gradient')
        if step == 2:
            parameters = (refiner.mask3.relation_encoder[0].weight, refiner.mask3.head1[0].weight,
                          refiner.mask3.head3[0].weight, refiner.mask3.head5[0].weight,
                          refiner.mask4.relation_encoder[0].weight, refiner.mask4.head1[0].weight,
                          refiner.mask4.head3[0].weight, refiner.mask4.head5[0].weight,
                          refiner.mask3.scale_selector[-1].weight, refiner.mask4.scale_selector[-1].weight)
            if not all(has_gradient(value) for value in parameters):
                raise RuntimeError('multi-RF branch lacks third-step gradient')
        optimizer.step()
    if hasattr(model, 'cvopm_stage2') or hasattr(model, 'det_head_stage2') or hasattr(model, 'cvopm_stage4_second'):
        raise RuntimeError('two-scale topology changed')
    print('{} passed: public_state=equal beta=zero modulation=one three_step_gradients=nonzero'.format(variant), flush=True)
    del base, model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required')
    for base_variant, variant in (('h2_ind_csfi_bi', 'h2_ind_amhcsfi_res_bi'),
                                  ('h2_ind_cg', 'h2_ind_cg_amhcsfi_res'),
                                  ('h2_ind_fg_nocsfi', 'h2_ind_fg_amhcsfi_res')):
        check_variant(base_variant, variant, args.batch_size)


if __name__ == '__main__':
    main()
