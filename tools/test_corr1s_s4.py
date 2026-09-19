"""Structural and shape checks for the strict native-Stage4 Corr1S baseline."""

import argparse

import torch

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', choices=('CVOGL_DroneAerial', 'CVOGL_SVI'), required=True)
    args = parser.parse_args()
    query_hw = (256, 256) if args.data_name == 'CVOGL_DroneAerial' else (256, 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone='swin_t', variant='corr1s_s4', position_mode='current').to(device).eval()

    forbidden = ('cvopm_stage3', 'cvopm_stage4', 'coarse_guidance', 'fine_guidance',
                 'cross_scale_interaction', 'amhcsfi_res_refiner', 'stage4_align',
                 'det_head_stage3', 'det_head_stage4')
    present = [name for name in forbidden if hasattr(model, name)]
    if present:
        raise AssertionError('Corr1S-S4 unexpectedly constructed: {}'.format(present))
    if not hasattr(model, 'det_head_corr1s_s4'):
        raise AssertionError('Corr1S-S4 native Stage4 head is missing')

    encoder_outputs = []
    hook = model.encoder.register_forward_hook(lambda _module, _args, output: encoder_outputs.append(output))
    try:
        with torch.no_grad():
            query = torch.randn(1, 3, *query_hw, device=device)
            satellite = torch.randn(1, 3, 1024, 1024, device=device)
            click_map = torch.rand(1, *query_hw, device=device)
            predictions, _ = model(query, satellite, click_map)
    finally:
        hook.remove()

    if len(encoder_outputs) != 2:
        raise AssertionError('expected shared encoder to run twice, got {}'.format(len(encoder_outputs)))
    (_, query_q4), (_, satellite_r4) = encoder_outputs
    expected_q4 = (1, 768, query_hw[0] // 32, query_hw[1] // 32)
    if tuple(query_q4.shape) != expected_q4 or tuple(satellite_r4.shape) != (1, 768, 32, 32):
        raise AssertionError('unexpected Stage4 shapes: q4={} r4={}'.format(
            tuple(query_q4.shape), tuple(satellite_r4.shape)))
    if set(predictions) != {'single_s4', 'detgeo_attn4'}:
        raise AssertionError('unexpected predictions: {}'.format(sorted(predictions)))
    if tuple(predictions['single_s4'].shape) != (1, 45, 32, 32):
        raise AssertionError('unexpected prediction shape {}'.format(tuple(predictions['single_s4'].shape)))
    if tuple(predictions['detgeo_attn4'].shape) != (1, 32, 32):
        raise AssertionError('unexpected attention shape {}'.format(tuple(predictions['detgeo_attn4'].shape)))
    print('Corr1S-S4 OK: data={} q4={} r4={} pred={} device={}'.format(
        args.data_name, tuple(query_q4.shape), tuple(satellite_r4.shape),
        tuple(predictions['single_s4'].shape), device))


if __name__ == '__main__':
    main()
