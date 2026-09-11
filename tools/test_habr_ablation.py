"""GPU contracts for the post-matching HABR ablation ladder."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.multiscale_detection_loss import coarse_heatmap_loss, two_head_yolo_loss


ANCHORS = torch.tensor(
    [[550, 573], [395, 342], [246, 280], [198, 179], [194, 82], [129, 129],
     [96, 215], [78, 84], [37, 41]], dtype=torch.float32)
VARIANTS = ('h2_ind_habr_core', 'h2_ind_habr_prior', 'h2_ind_habr_adapt', 'h2_ind_habr')
EXPECTED = {
    'h2_ind_habr_core': (False, False, 1),
    'h2_ind_habr_prior': (True, False, 1),
    'h2_ind_habr_adapt': (True, True, 1),
    'h2_ind_habr': (True, True, 2),
}


def has_gradient(parameter):
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def loss_for(predictions, target, use_prior):
    batch = target.shape[0]
    p3 = predictions['stage3'].view(batch, 9, 5, 64, 64)
    p4 = predictions['stage4'].view(batch, 9, 5, 64, 64)
    geo, cls = two_head_yolo_loss(p3, p4, target, ANCHORS.to(target.device), 1024)
    loss = geo + cls
    if use_prior:
        loss = loss + 0.5 * (coarse_heatmap_loss(predictions['habr_prior3_logits'], target, 1024, 3.0) +
                             coarse_heatmap_loss(predictions['habr_prior4_logits'], target, 1024, 1.5))
    return loss


def check_variant(variant, batch):
    use_prior, use_reliability, rounds = EXPECTED[variant]
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(variant=variant)).cuda().train()
    module, habr = model.module, model.module.habr_former
    if (habr.use_prior, habr.block.use_reliability, habr.rounds) != (use_prior, use_reliability, rounds):
        raise RuntimeError('{} configuration mismatch'.format(variant))
    if hasattr(habr, 'block1') or hasattr(habr, 'block2'):
        raise RuntimeError('{} must use one shared HABR block'.format(variant))
    z3, z4 = torch.randn(2, 384, 64, 64, device='cuda'), torch.randn(2, 768, 32, 32, device='cuda')
    with torch.no_grad():
        out3, out4, prior3, prior4, _ = habr(z3, z4)
    if not torch.equal(out3, z3) or not torch.equal(out4, z4):
        raise RuntimeError('{} zero-residual HABR is not an exact identity'.format(variant))
    if use_prior != (prior3 is not None and prior4 is not None):
        raise RuntimeError('{} prior contract mismatch'.format(variant))
    if use_prior and (prior3.shape != (2, 1, 64, 64) or prior4.shape != (2, 1, 32, 32)):
        raise RuntimeError('{} invalid prior shapes'.format(variant))
    del z3, z4, out3, out4, prior3, prior4
    query = torch.randn(batch, 3, 256, 256, device='cuda')
    reference = torch.randn(batch, 3, 1024, 1024, device='cuda')
    click = torch.rand(batch, 256, 256, device='cuda')
    target = torch.tensor([[256, 256, 512, 512]], dtype=torch.float32, device='cuda').repeat(batch, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    predictions, _ = model(query, reference, click)
    expected_keys = {'stage3', 'stage4'} | ({'habr_prior3_logits', 'habr_prior4_logits'} if use_prior else set())
    if set(predictions) != expected_keys or predictions['stage3'].shape != (batch, 45, 64, 64) or \
            predictions['stage4'].shape != (batch, 45, 64, 64):
        raise RuntimeError('{} violates the E4 dual-head contract'.format(variant))
    first_loss = loss_for(predictions, target, use_prior)
    if not torch.isfinite(first_loss):
        raise RuntimeError('{} first loss is non-finite'.format(variant))
    first_loss.backward()
    if not has_gradient(habr.block.alpha43) or not has_gradient(habr.block.alpha34):
        raise RuntimeError('{} HABR residual alpha has zero gradient'.format(variant))
    if use_prior and (not has_gradient(habr.prior3_head.weight) or not has_gradient(habr.prior4_head.weight)):
        raise RuntimeError('{} HABR prior head has zero gradient'.format(variant))
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    predictions, _ = model(query, reference, click)
    second_loss = loss_for(predictions, target, use_prior)
    second_loss.backward()
    block = habr.block
    required = (block.coarse_to_fine.sampling_predictor[0].weight, block.fine_to_coarse.q_proj.weight,
                block.fine_to_coarse.k_proj.weight, block.fine_to_coarse.v_proj.weight,
                block.fine_to_coarse.out_proj.weight)
    if not all(has_gradient(parameter) for parameter in required):
        raise RuntimeError('{} relation-reasoning branch has zero second-step gradient'.format(variant))
    if use_reliability and (not has_gradient(block.gate3.spatial_gate[0].weight) or
                            not has_gradient(block.gate4.channel_gate[1].weight) or
                            not has_gradient(block.direction_predictor[0].weight)):
        raise RuntimeError('{} adaptive reliability has zero gradient'.format(variant))
    values = first_loss.item(), second_loss.item(), sum(p.numel() for p in model.parameters())
    del model, predictions
    torch.cuda.empty_cache()
    return values


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--batch_size', type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required')
    for variant in VARIANTS:
        loss1, loss2, parameters = check_variant(variant, args.batch_size)
        print('habr_sanity variant={} loss1={:.6f} loss2={:.6f} params={}'.format(
            variant, loss1, loss2, parameters), flush=True)


if __name__ == '__main__':
    main()
