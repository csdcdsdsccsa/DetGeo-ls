"""Structural and forward checks for the Full Model Swin-T sharing ablation."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def build(unshared_backbone, gaussian_sigma_x=None, crgpe_outer_sigma_x=None):
    return TROGeoMSDetectionAblation(
        emb_size=768,
        backbone='swin_t',
        variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe',
        dadpe_mode='none',
        amr_pe_mode='none',
        gaussian_sigma=25.0,
        gaussian_sigma_x=gaussian_sigma_x,
        crgpe_outer_sigma=50.0,
        crgpe_outer_sigma_x=crgpe_outer_sigma_x,
        unshared_backbone=unshared_backbone,
    )


def run_forward(model, device, query_hw):
    height, width = query_hw
    model = model.to(device).eval()
    with torch.no_grad():
        query = torch.zeros(1, 3, height, width, device=device)
        reference = torch.zeros(1, 3, 1024, 1024, device=device)
        click_map = torch.zeros(1, height, width, device=device)
        predictions, _ = model(query, reference, click_map)
    assert tuple(predictions['stage3'].shape) == (1, 45, 64, 64)
    assert tuple(predictions['stage4'].shape) == (1, 45, 64, 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward', action='store_true',
                        help='run batch-one forward checks using real Drone and SVI query sizes')
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
        run_forward(model, device, (256, 256))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        svi_model = build(unshared_backbone=True, gaussian_sigma_x=50.0,
                          crgpe_outer_sigma_x=100.0)
        run_forward(svi_model, device, (256, 512))

    print('Full Model unshared Swin-T structural{} check passed.'.format(' and forward' if args.forward else ''))


if __name__ == '__main__':
    main()
