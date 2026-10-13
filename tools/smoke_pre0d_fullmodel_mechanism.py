"""One synthetic DroneAerial forward/backward for one mechanism variant."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


CASES = {
    'direct_ca': ('direct', 'amhcsfi_res', False, False),
    'coarse_only': ('coarse', 'amhcsfi_res', True, False),
    'fine_only': ('fine', 'amhcsfi_res', False, True),
    'original_csfi': ('bi', 'csfi', True, True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=CASES)
    args = parser.parse_args()
    guidance, cross_scale, expect_coarse, expect_fine = CASES[args.case]
    device = torch.device('cuda')
    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe', gaussian_sigma=25,
        guidance_mode=guidance, cross_scale_module=cross_scale).to(device).train()
    query = torch.rand(1, 3, 256, 256, device=device)
    satellite = torch.rand(1, 3, 1024, 1024, device=device)
    click = torch.rand(1, 256, 256, device=device)
    outputs, _ = model(query, satellite, click)
    assert tuple(outputs['stage3'].shape) == (1, 45, 64, 64)
    assert tuple(outputs['stage4'].shape) == (1, 45, 64, 64)
    assert ('coarse_logits' in outputs) == expect_coarse
    assert ('fine_logits' in outputs) == expect_fine
    loss = sum(value.square().mean() for value in outputs.values() if isinstance(value, torch.Tensor))
    loss.backward()
    grad_count = sum(p.grad is not None for p in model.parameters())
    assert grad_count > 0
    print('SMOKE_OK case={} loss={:.6f} grad_tensors={} peak_cuda_mb={:.1f}'.format(
        args.case, loss.item(), grad_count, torch.cuda.max_memory_allocated() / 1048576), flush=True)


if __name__ == '__main__':
    main()
