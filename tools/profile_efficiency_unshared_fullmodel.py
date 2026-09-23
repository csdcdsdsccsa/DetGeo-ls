"""FP32 efficiency benchmark for the trained Unshared Swin-T Full Model."""

import argparse
import csv
import gc
import math
import os
from collections import OrderedDict

import torch
from torch import nn

from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from tools.profile_efficiency import (
    FullModelProfileWrapper, benchmark_fps, count_flops, format_ops,
    load_checkpoint_strict, make_gaussian_click_map, make_images, write_outputs,
)


CASES = (
    {
        'dataset': 'DroneAerial', 'method': 'Unshared Swin-T Full Model',
        'checkpoint': 'saved_models/fullmodel_unshared_swin_t_drone_naturalrng_seed2024_model_best.pth.tar',
        'query_hw': (256, 256), 'sigma_y': 25.0, 'sigma_x': 25.0,
        'outer_y': 50.0, 'outer_x': 50.0,
    },
    {
        'dataset': 'CVOGL_SVI', 'method': 'Unshared Swin-T Full Model',
        'checkpoint': 'saved_models/fullmodel_unshared_swin_t_svi_naturalrng_seed2024_model_best.pth.tar',
        'query_hw': (256, 512), 'sigma_y': 25.0, 'sigma_x': 50.0,
        'outer_y': 50.0, 'outer_x': 100.0,
    },
)


def build_unshared_full_model(case):
    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe', gaussian_sigma=case['sigma_y'],
        gaussian_sigma_x=case['sigma_x'], crgpe_outer_sigma=case['outer_y'],
        crgpe_outer_sigma_x=case['outer_x'], dadpe_mode='none', amr_pe_mode='none',
        enable_hqs=False, enable_hqs_v2a=False, enable_hqs_v2b=False,
        enable_acr=False, unshared_backbone=True,
    )
    assert model.unshared_backbone is True
    assert not hasattr(model, 'encoder')
    assert hasattr(model, 'query_encoder') and hasattr(model, 'reference_encoder')
    assert model.query_encoder is not model.reference_encoder
    return model


def assert_unshared_encoders(model):
    query_parameters = list(model.query_encoder.parameters())
    reference_parameters = list(model.reference_encoder.parameters())
    assert len(query_parameters) == len(reference_parameters)
    for query_parameter, reference_parameter in zip(query_parameters, reference_parameters):
        assert query_parameter is not reference_parameter
        assert query_parameter.data_ptr() != reference_parameter.data_ptr()
    print('[PASS] Query/Satellite Swin-T are independent parameter sets', flush=True)


def compare_shared_and_unshared(rows, shared_csv, output_md):
    with open(shared_csv, newline='', encoding='utf-8') as handle:
        shared_rows = {(row['Dataset'], row['Method']): row for row in csv.DictReader(handle)}
    lines = [
        '# Shared vs Unshared Swin-T Full Model efficiency', '',
        '| Dataset | Sharing | Params (M) | FLOPs (G) | FPS | Latency (ms) |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        key = (row['Dataset'], 'Full Model')
        if key not in shared_rows:
            raise RuntimeError('missing Shared Full Model row: {}'.format(key))
        shared = shared_rows[key]
        shared_params, shared_flops = float(shared['Params_M']), float(shared['FLOPs_G'])
        if row['Params_M'] <= shared_params:
            raise RuntimeError('Unshared Params must exceed Shared Params for {}'.format(row['Dataset']))
        if not math.isclose(row['FLOPs_G'], shared_flops, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError('Shared/Unshared FLOPs differ for {}: {} vs {}'.format(
                row['Dataset'], shared_flops, row['FLOPs_G']))
        lines.extend((
            '| {dataset} | Shared | {params:.2f} | {flops:.2f} | {fps:.2f} | {latency:.2f} |'.format(
                dataset=row['Dataset'], params=shared_params, flops=shared_flops,
                fps=float(shared['FPS_mean']), latency=float(shared['Latency_ms_mean'])),
            '| {dataset} | Unshared | {params:.2f} | {flops:.2f} | {fps:.2f} | {latency:.2f} |'.format(
                dataset=row['Dataset'], params=row['Params_M'], flops=row['FLOPs_G'],
                fps=row['FPS_mean'], latency=row['Latency_ms_mean']),
        ))
    lines.extend(('', 'Unshared parameters are asserted greater; same-shape Shared/Unshared FLOPs are asserted equal. '
                  'FPS and latency are reported as wall-clock observations only.'))
    with open(output_md, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output_csv', default='results/efficiency_unshared_fullmodel.csv')
    parser.add_argument('--output_md', default='results/efficiency_unshared_fullmodel.md')
    parser.add_argument('--comparison_md', default='results/efficiency_shared_vs_unshared_fullmodel.md')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError('warmup must be non-negative; iterations and repeats must be positive')
    if not torch.cuda.is_available() or args.device != 'cuda:0':
        raise RuntimeError('use one visible CUDA device as cuda:0')
    for case in CASES:
        if not os.path.isfile(case['checkpoint']):
            raise FileNotFoundError(case['checkpoint'])
    shared_csv = 'results/efficiency_fullmodel_vs_detgeo.csv'
    if not os.path.isfile(shared_csv):
        raise FileNotFoundError('missing Shared comparison file: {}'.format(shared_csv))

    import fvcore
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    rows, parameter_counts = [], []
    for case in CASES:
        print('\n=== {} / {} ==='.format(case['dataset'], case['method']), flush=True)
        model = build_unshared_full_model(case)
        assert not isinstance(model, nn.DataParallel)
        load_checkpoint_strict(model, case['checkpoint'])
        assert_unshared_encoders(model)
        model = model.to(device).eval()
        query, satellite = make_images(case['query_hw'], device)
        click = make_gaussian_click_map(*case['query_hw'], case['sigma_y'], case['sigma_x'], device)
        assert tuple(query.shape) == (1, 3, *case['query_hw'])
        assert tuple(satellite.shape) == (1, 3, 1024, 1024)
        assert tuple(click.shape) == (1, *case['query_hw'])
        print('[PASS] shared_encoder=False query={} satellite={} click={}'.format(
            tuple(query.shape), tuple(satellite.shape), tuple(click.shape)), flush=True)

        with torch.inference_mode():
            model(query, satellite, click)
        assert model._logged_sanity
        wrapper = FullModelProfileWrapper(model)
        flops, unsupported, uncalled = count_flops(wrapper, (query, satellite, click))
        speed = benchmark_fps(model, (query, satellite, click), args.warmup, args.iterations, args.repeats)
        params = sum(parameter.numel() for parameter in model.parameters())
        parameter_counts.append(params)
        row = {
            'Dataset': case['dataset'], 'Method': case['method'],
            'Weight_Mode': 'Trained validation-best checkpoint', 'Checkpoint': case['checkpoint'],
            'Params_M': params / 1e6, 'FLOPs_G': flops / 1e9,
            'FPS_mean': speed['fps_mean'], 'FPS_std': speed['fps_std'],
            'Latency_ms_mean': speed['latency_ms_mean'], 'Latency_ms_std': speed['latency_ms_std'],
            'Query_shape': '1x3x{}x{}'.format(*case['query_hw']), 'Satellite_shape': '1x3x1024x1024',
            'Batch': 1, 'Precision': 'FP32 (TF32=False)',
            'Unsupported_ops': format_ops(unsupported), 'Uncalled_modules': ', '.join(uncalled) or 'none',
        }
        rows.append(row)
        print('[PASS] params={:.2f}M FLOPs={:.2f}G FPS={:.2f} latency={:.2f}ms'.format(
            row['Params_M'], row['FLOPs_G'], row['FPS_mean'], row['Latency_ms_mean']), flush=True)
        del wrapper, model, query, satellite, click
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if parameter_counts[0] != parameter_counts[1]:
        raise RuntimeError('Drone/SVI Unshared Full Model parameter counts differ')

    metadata = OrderedDict((
        ('GPU', torch.cuda.get_device_name(0)), ('PyTorch', torch.__version__),
        ('Torchvision', __import__('torchvision').__version__), ('CUDA runtime', torch.version.cuda),
        ('cuDNN', torch.backends.cudnn.version()), ('fvcore', fvcore.__version__),
        ('batch_size', 1), ('precision', 'FP32'), ('TF32', False),
        ('warmup', args.warmup), ('iterations', args.iterations), ('repeats', args.repeats),
        ('FLOPs convention', 'fvcore; 1 MAC = 1 FLOP'),
        ('backbone sharing', 'Unshared: independent trained Query and Satellite Swin-T encoders'),
    ))
    write_outputs(rows, metadata, args.output_csv, args.output_md)
    compare_shared_and_unshared(rows, shared_csv, args.comparison_md)
    print('[PASS] wrote {}, {}, and {}'.format(args.output_csv, args.output_md, args.comparison_md))


if __name__ == '__main__':
    main()
