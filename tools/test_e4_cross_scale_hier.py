"""Two-step GPU checks for the first E4 cross-scale collaboration ablation."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.multiscale_detection_loss import coarse_heatmap_loss, two_head_yolo_loss


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


def check_zero_residual():
    """Zero-initialized CSFI gates must be an identity within the same model."""
    module = TROGeoMSDetectionAblation(variant='h2_ind_csfi').cuda().eval().cross_scale_interaction
    z3 = torch.randn(2, 384, 64, 64, device='cuda')
    z4 = torch.randn(2, 768, 32, 32, device='cuda')
    with torch.no_grad():
        out3, out4, _, _ = module(z3, z4)
    max_difference = max((out3 - z3).abs().max().item(), (out4 - z4).abs().max().item())
    if max_difference != 0.0 or module.alpha3.item() != 0.0 or module.alpha4.item() != 0.0:
        raise RuntimeError('CSFI zero-gate residual is not an exact identity')
    del module, z3, z4, out3, out4
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


def check_coarse_variants(batch):
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.rand(batch, 256, 256, device='cuda')
    target = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)
    for variant in ('h2_ind_cg', 'h2_ind_csfi_cg', 'h2_ind_hier', 'h2_ind_cg_amhcsfi_res'):
        model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant=variant)).cuda().train()
        module = model.module
        # gamma=0 must leave the Stage3 satellite feature exactly unchanged;
        # this is a residual-initialization test, not an RNG comparison.
        probe_z4 = torch.randn(batch, 768, 32, 32, device='cuda')
        probe_r3 = torch.randn(batch, 384, 64, 64, device='cuda')
        with torch.no_grad():
            probe_r3_guided, probe_logits, probe_up = module.coarse_guidance(probe_z4, probe_r3)
        if not torch.equal(probe_r3_guided, probe_r3) or probe_logits.shape != (batch, 1, 32, 32) or \
                probe_up.shape != (batch, 1, 64, 64):
            raise RuntimeError('{} zero-gamma coarse guidance is not an identity'.format(variant))
        before_head = {}
        hook = None
        if variant == 'h2_ind_hier':
            hook = module.det_head_stage3.register_forward_hook(
                lambda _layer, _inputs, output: before_head.setdefault('prediction', output.detach().clone()))
        predictions, _ = model(query, reference, click)
        if hook is not None:
            hook.remove()
        if predictions['coarse_logits'].shape != (batch, 1, 32, 32):
            raise RuntimeError('{} has invalid coarse heatmap shape'.format(variant))
        if module.coarse_guidance.gamma.item() != 0.0:
            raise RuntimeError('{} gamma must be zero initialized'.format(variant))
        if variant == 'h2_ind_hier':
            if set(predictions) != {'stage3', 'coarse_logits'}:
                raise RuntimeError('hierarchical variant must decode Stage3 only')
            loss = coarse_heatmap_loss(predictions['coarse_logits'], target, 1024)
            if module.hier_conf_alpha.item() != 0.0:
                raise RuntimeError('hier_conf_alpha must be zero initialized')
            if not torch.equal(predictions['stage3'], before_head['prediction']):
                raise RuntimeError('zero hierarchical confidence prior changed Stage3 predictions')
        else:
            if set(predictions) != {'stage3', 'stage4', 'coarse_logits'}:
                raise RuntimeError('{} must retain both E4 heads'.format(variant))
            loss = two_head_loss(predictions, target) + coarse_heatmap_loss(predictions['coarse_logits'], target, 1024)
        if not torch.isfinite(loss):
            raise RuntimeError('{} has non-finite coarse loss'.format(variant))
        loss.backward()
        if not has_gradient(module.coarse_guidance.coarse_head.weight):
            raise RuntimeError('{} coarse head has zero gradient'.format(variant))
        del model, predictions
        torch.cuda.empty_cache()


def check_fine_guidance_variants(batch):
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.rand(batch, 256, 256, device='cuda')
    target = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)
    no_csfi_variants = ('h2_ind_fg_nocsfi', 'h2_ind_bi_nocsfi')
    # NoCSFI variants are standalone natural-RNG models.  They must not
    # construct/register CSFI or use synthetic RNG padding to imitate it.
    bidir_variants = ('h2_ind_csfi_bi', 'h2_ind_mhcsfi_bi', 'h2_ind_amhcsfi_bi',
                      'h2_ind_amhcsfi_res_bi', 'h2_ind_bi_nocsfi')
    for variant in ('h2_ind_csfi_fg', 'h2_ind_fg_amhcsfi_res') + bidir_variants + no_csfi_variants[:1]:
        model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant=variant)).cuda().train()
        module = model.module
        if variant in no_csfi_variants and hasattr(module, 'cross_scale_interaction'):
            raise RuntimeError('{} must not register CSFI'.format(variant))
        probe_z3 = torch.randn(batch, 384, 64, 64, device='cuda')
        probe_r4 = torch.randn(batch, 768, 32, 32, device='cuda')
        with torch.no_grad():
            probe_r4_guided, probe_logits, probe_down = module.fine_guidance(probe_z3, probe_r4)
        if not torch.equal(probe_r4_guided, probe_r4) or probe_logits.shape != (batch, 1, 64, 64) or \
                probe_down.shape != (batch, 1, 32, 32) or module.fine_guidance.gamma.item() != 0.0:
            raise RuntimeError('{} zero-gamma fine guidance is not an exact identity'.format(variant))
        if hasattr(module, 'cvopm_stage4_second'):
            raise RuntimeError('A3-Bi must reuse cvopm_stage4')
        predictions, _ = model(query, reference, click)
        expected_keys = ({'stage3', 'stage4', 'fine_logits'} if variant in ('h2_ind_csfi_fg', 'h2_ind_fg_nocsfi',
                                                                             'h2_ind_fg_amhcsfi_res')
                         else {'stage3', 'stage4', 'coarse_logits', 'fine_logits'})
        if set(predictions) != expected_keys or predictions['stage3'].shape != (batch, 45, 64, 64) or \
                predictions['stage4'].shape != (batch, 45, 64, 64) or \
                predictions['fine_logits'].shape != (batch, 1, 64, 64):
            raise RuntimeError('{} violates the dual-head fine-guidance contract'.format(variant))
        fine_loss = coarse_heatmap_loss(predictions['fine_logits'], target, 1024, sigma=3.0)
        loss = two_head_loss(predictions, target) + fine_loss
        if variant in bidir_variants:
            if predictions['coarse_logits'].shape != (batch, 1, 32, 32):
                raise RuntimeError('A3-Bi has invalid coarse heatmap')
            loss = loss + 0.5 * coarse_heatmap_loss(predictions['coarse_logits'], target, 1024, sigma=1.5)
        if not torch.isfinite(loss):
            raise RuntimeError('{} has non-finite loss'.format(variant))
        loss.backward()
        if not has_gradient(module.fine_guidance.fine_head.weight) or not has_gradient(module.fine_guidance.gamma):
            raise RuntimeError('{} fine-guidance branch has zero gradient'.format(variant))
        if variant in bidir_variants and not has_gradient(module.coarse_guidance.coarse_head.weight):
            raise RuntimeError('A3-Bi coarse-guidance branch has zero gradient')
        del model, predictions
        torch.cuda.empty_cache()


def check_adaptive_csfi_variants(batch):
    variants = (
        'h2_ind_csfi_cg_channel', 'h2_ind_csfi_cg_dir', 'h2_ind_csfi_cg_ar')
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.rand(batch, 256, 256, device='cuda')
    target = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)
    for variant in variants:
        model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant=variant)).cuda().train()
        module = model.module
        z3, z4 = (torch.randn(batch, 384, 64, 64, device='cuda'),
                  torch.randn(batch, 768, 32, 32, device='cuda'))
        with torch.no_grad():
            out3, out4, diagnostics = module.adaptive_csfi_reliability(
                z3, z4, module.cross_scale_interaction)
        if not torch.equal(out3, z3) or not torch.equal(out4, z4):
            raise RuntimeError('{} changes features at zero alpha'.format(variant))
        if diagnostics['channel3_mean'].item() != 1.0 or diagnostics['channel4_mean'].item() != 1.0 or \
                diagnostics['lambda43'].item() != 0.5 or diagnostics['lambda34'].item() != 0.5:
            raise RuntimeError('{} does not have neutral reliability initialization'.format(variant))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        predictions, _ = model(query, reference, click)
        if set(predictions) != {'stage3', 'stage4', 'coarse_logits'} or \
                predictions['stage3'].shape != (batch, 45, 64, 64) or \
                predictions['stage4'].shape != (batch, 45, 64, 64) or \
                predictions['coarse_logits'].shape != (batch, 1, 32, 32):
            raise RuntimeError('{} violates the A3 prediction contract'.format(variant))
        loss = two_head_loss(predictions, target) + coarse_heatmap_loss(predictions['coarse_logits'], target, 1024)
        if not torch.isfinite(loss):
            raise RuntimeError('{} has a non-finite first loss'.format(variant))
        loss.backward()
        for parameter in (module.cross_scale_interaction.alpha3, module.cross_scale_interaction.alpha4,
                          module.coarse_guidance.coarse_head.weight):
            if not has_gradient(parameter):
                raise RuntimeError('{} has zero first-step shared-module gradient'.format(variant))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        predictions, _ = model(query, reference, click)
        loss = two_head_loss(predictions, target) + coarse_heatmap_loss(predictions['coarse_logits'], target, 1024)
        loss.backward()
        if variant in module.ADAPTIVE_CSFI_CHANNEL_VARIANTS:
            for layer in (module.adaptive_csfi_reliability.channel3[-1],
                          module.adaptive_csfi_reliability.channel4[-1]):
                if not has_gradient(layer.weight):
                    raise RuntimeError('{} channel reliability has zero second-step gradient'.format(variant))
        if variant in module.ADAPTIVE_CSFI_DIRECTION_VARIANTS and not has_gradient(
                module.adaptive_csfi_reliability.direction_predictor[-1].weight):
            raise RuntimeError('{} direction reliability has zero second-step gradient'.format(variant))
        del model, predictions, z3, z4, out3, out4
        torch.cuda.empty_cache()


def check_cg_habr_prior(batch):
    variant = 'h2_ind_cg_habr_prior'
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant=variant)).cuda().train()
    module = model.module
    if not hasattr(module, 'coarse_guidance') or not hasattr(module, 'habr_former') or \
            hasattr(module, 'cross_scale_interaction'):
        raise RuntimeError('CG-HABR must contain coarse guidance and HABR, but no CSFI')
    if module.habr_former.mode != 'prior' or not module.habr_former.use_prior or \
            module.habr_former.rounds != 1 or module.coarse_guidance.gamma.item() != 0.0 or \
            module.habr_former.block.alpha43.item() != 0.0 or module.habr_former.block.alpha34.item() != 0.0:
        raise RuntimeError('CG-HABR does not preserve the required prior/zero-residual configuration')
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.rand(batch, 256, 256, device='cuda')
    predictions, _ = model(query, reference, click)
    expected = {'stage3', 'stage4', 'coarse_logits', 'habr_prior3_logits', 'habr_prior4_logits'}
    if set(predictions) != expected or predictions['stage3'].shape != (batch, 45, 64, 64) or \
            predictions['stage4'].shape != (batch, 45, 64, 64) or \
            predictions['coarse_logits'].shape != (batch, 1, 32, 32) or \
            predictions['habr_prior3_logits'].shape != (batch, 1, 64, 64) or \
            predictions['habr_prior4_logits'].shape != (batch, 1, 32, 32):
        raise RuntimeError('CG-HABR prediction contract mismatch')
    del model, predictions
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    difference = check_zero_residual()
    loss1, loss2, parameters = check_two_step_backward(args.batch_size)
    check_coarse_variants(args.batch_size)
    check_fine_guidance_variants(args.batch_size)
    check_adaptive_csfi_variants(args.batch_size)
    check_cg_habr_prior(args.batch_size)
    print('e4_csfi_sanity batch={} identity_max_abs={} loss1={:.6f} loss2={:.6f} '
          'params={} peak_mib={:.1f}'.format(
              args.batch_size, difference, loss1, loss2, parameters,
              torch.cuda.max_memory_allocated() / 1024 / 1024), flush=True)


if __name__ == '__main__':
    main()
