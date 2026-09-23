"""Unified FP32 efficiency benchmark for original DetGeo and shared Full Model.

This tool intentionally does not alter either model's forward path.  In
particular, original DetGeo's CPU NumPy score min/max remains part of measured
wall-clock latency even though it is outside fvcore's Torch FLOP graph.
"""

import argparse
import csv
import gc
import math
import os
import statistics
import time
from collections import OrderedDict

import torch
from torch import nn

from model.DetGeo import DetGeo
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


CASES = (
    {
        'dataset': 'DroneAerial', 'method': 'DetGeo', 'kind': 'detgeo',
        'checkpoint': 'saved_models/scratch25_worker24_square_seed13_model_best.pth.tar',
        'query_hw': (256, 256),
    },
    {
        'dataset': 'DroneAerial', 'method': 'Full Model', 'kind': 'full',
        'checkpoint': ('saved_models/trogeo_ms_e4_amhcsfi_res_bi_hisym_crgpe_core25_outer50_'
                       'naturalrng_swin_t_drone_seed2024_model_best.pth.tar'),
        'query_hw': (256, 256), 'sigma_y': 25.0, 'sigma_x': 25.0,
        'outer_y': 50.0, 'outer_x': 50.0,
    },
    {
        'dataset': 'CVOGL_SVI', 'method': 'DetGeo', 'kind': 'detgeo',
        'checkpoint': 'saved_models/scratch25_worker24_square_svi_naturalrng_seed13_model_best.pth.tar',
        'query_hw': (256, 512),
    },
    {
        'dataset': 'CVOGL_SVI', 'method': 'Full Model', 'kind': 'full',
        'checkpoint': ('saved_models/trogeo_ms_e4_amhcsfi_res_bi_hisym_crgpe_'
                       'corey25_corex50_outery50_outerx100_naturalrng_swin_t_svi_seed2024_model_best.pth.tar'),
        'query_hw': (256, 512), 'sigma_y': 25.0, 'sigma_x': 50.0,
        'outer_y': 50.0, 'outer_x': 100.0,
    },
)

CRITICAL_OPS = {
    'aten::bmm', 'aten::matmul', 'aten::mm', 'aten::addmm', 'aten::einsum',
    'aten::linear', 'aten::_convolution', 'aten::conv2d',
}


class DetGeoProfileWrapper(nn.Module):
    """Preserve both original DetGeo forward outputs during tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, query, satellite, click):
        prediction, attention = self.model(query, satellite, click)
        return prediction, attention


class FullModelProfileWrapper(nn.Module):
    """Make the Full Model's prediction dict traceable without pruning heads."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, query, satellite, click):
        predictions, _ = self.model(query, satellite, click)
        return (
            predictions['stage3'], predictions['stage4'],
            predictions['coarse_logits'], predictions['fine_logits'],
        )


def build_detgeo():
    return DetGeo(emb_size=512, leaky=True)


def build_full_model(case):
    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe', gaussian_sigma=case['sigma_y'],
        gaussian_sigma_x=case['sigma_x'], crgpe_outer_sigma=case['outer_y'],
        crgpe_outer_sigma_x=case['outer_x'], dadpe_mode='none', amr_pe_mode='none',
        enable_hqs=False, enable_hqs_v2a=False, enable_hqs_v2b=False,
        enable_acr=False, unshared_backbone=False,
    )
    assert model.unshared_backbone is False
    assert hasattr(model, 'encoder')
    assert not hasattr(model, 'query_encoder')
    assert not hasattr(model, 'reference_encoder')
    return model


def load_checkpoint_strict(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location='cpu')
    if 'state_dict' not in checkpoint:
        raise RuntimeError('checkpoint has no state_dict: {}'.format(path))
    state_dict = checkpoint['state_dict']
    keys = list(state_dict)
    has_module = [key.startswith('module.') for key in keys]
    if any(has_module) and not all(has_module):
        raise RuntimeError('mixed module./non-module. checkpoint keys: {}'.format(path))
    if keys and all(has_module):
        state_dict = OrderedDict((key[len('module.'):], value) for key, value in state_dict.items())
    model.load_state_dict(state_dict, strict=True)
    print('[PASS] strict checkpoint load: {}'.format(path), flush=True)


def make_images(query_hw, device):
    height, width = query_hw
    generator = torch.Generator(device='cpu')
    generator.manual_seed(20240923)
    query = torch.randn(1, 3, height, width, generator=generator, dtype=torch.float32).to(device)
    satellite = torch.randn(1, 3, 1024, 1024, generator=generator, dtype=torch.float32).to(device)
    return query, satellite


def make_detgeo_click_map(height, width, device):
    center_y, center_x = height // 2, width // 2
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    distance = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    click = (1.0 - distance / math.sqrt(height * height + width * width)) ** 2
    return click.unsqueeze(0).to(device)


def make_gaussian_click_map(height, width, sigma_y, sigma_x, device):
    center_y, center_x = height // 2, width // 2
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    click = torch.exp(-((yy - center_y) ** 2 / (2.0 * sigma_y ** 2)
                        + (xx - center_x) ** 2 / (2.0 * sigma_x ** 2)))
    return click.unsqueeze(0).to(device)


def bmm_flop_jit(inputs, outputs):
    from fvcore.nn.jit_handles import get_shape
    left, right = get_shape(inputs[0]), get_shape(inputs[1])
    if left is None or right is None or len(left) != 3 or len(right) != 3:
        raise RuntimeError('cannot infer BMM shapes: {} @ {}'.format(left, right))
    if left[0] != right[0] or left[2] != right[1]:
        raise RuntimeError('incompatible BMM shapes: {} @ {}'.format(left, right))
    # fvcore convention: one multiply-accumulate is one FLOP.
    return left[0] * left[1] * left[2] * right[2]


def count_flops(wrapper, inputs):
    from fvcore.nn import FlopCountAnalysis
    analysis = FlopCountAnalysis(wrapper, inputs)
    analysis.set_op_handle('aten::bmm', bmm_flop_jit)
    total = analysis.total()
    unsupported = dict(analysis.unsupported_ops())
    missing = {op: count for op, count in unsupported.items() if op in CRITICAL_OPS}
    if missing:
        raise RuntimeError('critical FLOP ops unsupported: {}'.format(missing))
    return total, unsupported, sorted(analysis.uncalled_modules())


def benchmark_fps(model, inputs, warmup, iterations, repeats):
    query, satellite, click = inputs
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(query, satellite, click)
        torch.cuda.synchronize()
        fps_values, latency_values = [], []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                model(query, satellite, click)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            latency_values.append(elapsed * 1000.0 / iterations)
            fps_values.append(iterations / elapsed)
    return {
        'fps_mean': statistics.mean(fps_values),
        'fps_std': statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0,
        'latency_ms_mean': statistics.mean(latency_values),
        'latency_ms_std': statistics.stdev(latency_values) if len(latency_values) > 1 else 0.0,
    }


def format_ops(ops):
    return ', '.join('{}={}'.format(key, value) for key, value in sorted(ops.items())) or 'none'


def write_outputs(rows, metadata, output_csv, output_md):
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    fieldnames = ('Dataset', 'Method', 'Checkpoint', 'Params_M', 'FLOPs_G', 'FPS_mean',
                  'FPS_std', 'Latency_ms_mean', 'Latency_ms_std', 'Query_shape',
                  'Satellite_shape', 'Batch', 'Precision', 'Unsupported_ops', 'Uncalled_modules')
    with open(output_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(os.path.dirname(output_md) or '.', exist_ok=True)
    with open(output_md, 'w', encoding='utf-8') as handle:
        handle.write('# DetGeo vs Shared-Swin-T Full Model efficiency\n\n')
        handle.write('| Dataset | Method | Params (M) ↓ | FLOPs (G) ↓ | FPS mean±std ↑ | Latency (ms) mean±std ↓ |\n')
        handle.write('|---|---:|---:|---:|---:|---:|\n')
        for row in rows:
            handle.write('| {Dataset} | {Method} | {Params_M:.2f} | {FLOPs_G:.2f} | '
                         '{FPS_mean:.2f} ± {FPS_std:.2f} | {Latency_ms_mean:.2f} ± {Latency_ms_std:.2f} |\n'.format(**row))
        handle.write('\n## Protocol\n\n')
        for key, value in metadata.items():
            handle.write('- {}: {}\n'.format(key, value))
        handle.write('\nDetGeo CPU NumPy score min/max is not represented in fvcore FLOPs; it is included in measured wall-clock FPS.\n')
        handle.write('\n## Per-model tracing details\n\n')
        for row in rows:
            handle.write('### {} — {}\n\n'.format(row['Dataset'], row['Method']))
            handle.write('- checkpoint: `{}`\n'.format(row['Checkpoint']))
            handle.write('- unsupported_ops: {}\n'.format(row['Unsupported_ops']))
            handle.write('- uncalled_modules: {}\n\n'.format(row['Uncalled_modules']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output_csv', default='results/efficiency_fullmodel_vs_detgeo.csv')
    parser.add_argument('--output_md', default='results/efficiency_fullmodel_vs_detgeo.md')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError('warmup must be non-negative; iterations and repeats must be positive')
    if not torch.cuda.is_available() or args.device != 'cuda:0':
        raise RuntimeError('use one visible CUDA device as cuda:0')
    try:
        import fvcore
    except ImportError as error:
        raise RuntimeError('fvcore is required for FLOPs profiling') from error

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    for case in CASES:
        if not os.path.isfile(case['checkpoint']):
            raise FileNotFoundError(case['checkpoint'])
    if not os.path.isfile('saved_models/yolov3.weights'):
        raise FileNotFoundError('saved_models/yolov3.weights')
    print('[PASS] all four checkpoints exist')
    print('[PASS] saved_models/yolov3.weights exists')
    print('[PASS] no DataParallel used; batch size = 1; precision = FP32; TF32 = False')

    rows, parameter_counts = [], {'detgeo': [], 'full': []}
    for case in CASES:
        print('\n=== {} / {} ==='.format(case['dataset'], case['method']), flush=True)
        model = build_detgeo() if case['kind'] == 'detgeo' else build_full_model(case)
        assert not isinstance(model, nn.DataParallel)
        load_checkpoint_strict(model, case['checkpoint'])
        model = model.to(device).eval()
        query, satellite = make_images(case['query_hw'], device)
        if case['kind'] == 'detgeo':
            click = make_detgeo_click_map(*case['query_hw'], device)
        else:
            click = make_gaussian_click_map(*case['query_hw'], case['sigma_y'], case['sigma_x'], device)
            print('[PASS] {} Full Model shared_encoder=True; CRGPE core=({}, {}), outer=({}, {})'.format(
                case['dataset'], case['sigma_y'], case['sigma_x'], case['outer_y'], case['outer_x']))
        assert tuple(query.shape) == (1, 3, *case['query_hw'])
        assert tuple(satellite.shape) == (1, 3, 1024, 1024)
        assert tuple(click.shape) == (1, *case['query_hw'])
        print('[PASS] query={} satellite={} click={}'.format(tuple(query.shape), tuple(satellite.shape), tuple(click.shape)))

        # Trigger one regular eager forward before tracing so model sanity logging
        # (which uses .item()) is not present in the JIT trace.
        with torch.inference_mode():
            model(query, satellite, click)
        if case['kind'] == 'full':
            assert model._logged_sanity
        wrapper = DetGeoProfileWrapper(model) if case['kind'] == 'detgeo' else FullModelProfileWrapper(model)
        flops, unsupported, uncalled = count_flops(wrapper, (query, satellite, click))
        speed = benchmark_fps(model, (query, satellite, click), args.warmup, args.iterations, args.repeats)
        params = sum(parameter.numel() for parameter in model.parameters())
        parameter_counts[case['kind']].append(params)
        row = {
            'Dataset': case['dataset'], 'Method': case['method'], 'Checkpoint': case['checkpoint'],
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

    if parameter_counts['detgeo'][0] != parameter_counts['detgeo'][1]:
        raise RuntimeError('Drone/SVI DetGeo parameter counts differ')
    if parameter_counts['full'][0] != parameter_counts['full'][1]:
        raise RuntimeError('Drone/SVI Full Model parameter counts differ')
    print('[PASS] Drone/SVI parameter counts match within each model family')
    metadata = OrderedDict((
        ('GPU', torch.cuda.get_device_name(0)), ('PyTorch', torch.__version__),
        ('Torchvision', __import__('torchvision').__version__), ('CUDA runtime', torch.version.cuda),
        ('cuDNN', torch.backends.cudnn.version()), ('fvcore', fvcore.__version__),
        ('batch_size', 1), ('precision', 'FP32'), ('TF32', False),
        ('warmup', args.warmup), ('iterations', args.iterations), ('repeats', args.repeats),
        ('FLOPs convention', 'fvcore; 1 MAC = 1 FLOP'),
    ))
    write_outputs(rows, metadata, args.output_csv, args.output_md)
    print('[PASS] wrote {} and {}'.format(args.output_csv, args.output_md))


if __name__ == '__main__':
    main()
