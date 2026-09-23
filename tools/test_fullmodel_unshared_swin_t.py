"""Structural and forward checks for the Full Model Swin-T sharing ablation."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def build(unshared_backbone):
    return TROGeoMSDetectionAblation(
        emb_size=768,
        backbone='swin_t',
        variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe',
        dadpe_mode='none',
        amr_pe_mode='none',
        gaussian_sigma=25.0,
        crgpe_outer_sigma=50.0,
        unshared_backbone=unshared_backbone,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward', action='store_true', help='run a batch-one 1024px inference check')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    shared = build(unshared_backbone=False)
    assert hasattr(shared, 'encoder')
    assert not hasattr(shared, 'query_encoder')
    assert not hasattr(shared, 'reference_encoder')

    model = build(unshared_backbone=True)
    assert not hasattr(model, 'encoder')
    assert model.query_encoder is not model.reference_encoder

    query_parameters = list(model.query_encoder.parameters())
    reference_parameters = list(model.reference_encoder.parameters())
    assert len(query_parameters) == len(reference_parameters)
    for query_parameter, reference_parameter in zip(query_parameters, reference_parameters):
        assert query_parameter is not reference_parameter
        assert query_parameter.data_ptr() != reference_parameter.data_ptr()
        assert torch.equal(query_parameter.detach().cpu(), reference_parameter.detach().cpu())

    required = ('cvopm_stage3', 'cvopm_stage4', 'coarse_guidance', 'fine_guidance',
                'cross_scale_interaction', 'amhcsfi_res_refiner',
                'det_head_stage3', 'det_head_stage4')
    assert all(hasattr(model, name) for name in required)

    if args.forward:
        device = torch.device(args.device)
        model = model.to(device).eval()
        with torch.no_grad():
            query = torch.zeros(1, 3, 1024, 1024, device=device)
            reference = torch.zeros(1, 3, 1024, 1024, device=device)
            click_map = torch.zeros(1, 1024, 1024, device=device)
            predictions, _ = model(query, reference, click_map)
        assert tuple(predictions['stage3'].shape) == (1, 45, 64, 64)
        assert tuple(predictions['stage4'].shape) == (1, 45, 64, 64)

    print('Full Model unshared Swin-T structural{} check passed.'.format(' and forward' if args.forward else ''))


if __name__ == '__main__':
    main()
