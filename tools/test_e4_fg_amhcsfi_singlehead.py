"""GPU sanity checks for FG-AMHCSFI-Res single-head detection ablations."""

import argparse
import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


VARIANTS = (
    'h2_ind_fg_amhcsfi_res_s3',
    'h2_ind_fg_amhcsfi_res_s4',
    'h2_ind_fg_amhcsfi_res_afuse',
)


def common_state(model):
    excluded = ('det_head_single.', 'stage34_adaptive_fusion.')
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if not key.startswith(excluded)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required')
    reference_state = None
    for variant in VARIANTS:
        torch.manual_seed(2024)
        model = TROGeoMSDetectionAblation(variant=variant).cuda().eval()
        if not hasattr(model, 'det_head_single') or hasattr(model, 'det_head_stage3') or hasattr(model, 'det_head_stage4'):
            raise RuntimeError('{} did not construct exactly one detection head'.format(variant))
        if not all(hasattr(model, name) for name in (
                'cvopm_stage3', 'cvopm_stage4', 'fine_guidance', 'cross_scale_interaction', 'amhcsfi_res_refiner')):
            raise RuntimeError('{} lost a required FG-AMHCSFI module'.format(variant))
        current_state = common_state(model)
        if reference_state is None:
            reference_state = current_state
        else:
            changed = [key for key in reference_state if key not in current_state or
                       not torch.equal(reference_state[key], current_state[key])]
            if changed:
                raise RuntimeError('{} changed public shared initialization: {}'.format(variant, changed[:3]))
        with torch.no_grad():
            query = torch.randn(args.batch_size, 3, 1024, 1024, device='cuda')
            reference = torch.randn(args.batch_size, 3, 1024, 1024, device='cuda')
            click = torch.randn(args.batch_size, 1024, 1024, device='cuda')
            predictions, _ = model(query, reference, click)
        if tuple(predictions['single'].shape) != (args.batch_size, 45, 64, 64):
            raise RuntimeError('{} invalid single-head shape {}'.format(variant, tuple(predictions['single'].shape)))
        if tuple(predictions['fine_logits'].shape) != (args.batch_size, 1, 64, 64):
            raise RuntimeError('{} lost fine guidance'.format(variant))
        if variant.endswith('_afuse'):
            weights = predictions['fusion_weights']
            if tuple(weights.shape) != (args.batch_size, 2) or not torch.allclose(weights.sum(1), torch.ones(args.batch_size, device='cuda')):
                raise RuntimeError('AFuse weights are not normalized')
            if not torch.allclose(weights, torch.full_like(weights, 0.5)):
                raise RuntimeError('AFuse must initialize with equal weights')
        print('{} passed: one_head fine_guidance AMHCSFI shared_init'.format(variant), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
