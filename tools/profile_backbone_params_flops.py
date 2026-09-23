"""CPU-only Params/FLOPs benchmark for the ten trained shared Full Models.

The temporary offline encoder replacements only prevent torchvision from
downloading ImageNet weights during construction.  Every replacement retains
the original module attributes/state_dict layout and is immediately followed
by a strict validation-best checkpoint load.
"""

import argparse
import csv
import os
from collections import OrderedDict

import torch
from torch import nn
import torchvision.models as models

import model.TROGeo_ms_detection_ablation as ablation_module
from model.TROGeo_ms_direct_ca_sh import (
    ResNet50MultiStageEncoder as OriginalResNet50MultiStageEncoder,
    SwinBMultiStageEncoder as OriginalSwinBMultiStageEncoder,
    SwinBNativeMultiStageEncoder as OriginalSwinBNativeMultiStageEncoder,
    SwinSMultiStageEncoder as OriginalSwinSMultiStageEncoder,
    SwinTMultiStageEncoder as OriginalSwinTMultiStageEncoder,
)
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from tools.profile_efficiency import (
    FullModelProfileWrapper,
    count_flops,
    format_ops,
    load_checkpoint_strict,
    make_gaussian_click_map,
    make_images,
)


CASES = (
    {
        'dataset': 'DroneAerial', 'backbone_label': 'ResNet-50', 'backbone': 'resnet50',
        'checkpoint': ('saved_models/fullmodel_bires_amhcsfi_res_hisym_crgpe_'
                       'resnet50_drone_naturalrng_seed2024_model_best.pth.tar'),
        'query_hw': (256, 256), 'sigma_y': 25.0, 'sigma_x': 25.0, 'outer_y': 50.0, 'outer_x': 50.0,
    },
    {
        'dataset': 'DroneAerial', 'backbone_label': 'Swin-T', 'backbone': 'swin_t',
        'checkpoint': ('saved_models/trogeo_ms_e4_amhcsfi_res_bi_hisym_crgpe_core25_outer50_'
                       'naturalrng_swin_t_drone_seed2024_model_best.pth.tar'),
        'query_hw': (256, 256), 'sigma_y': 25.0, 'sigma_x': 25.0, 'outer_y': 50.0, 'outer_x': 50.0,
    },
    {
        'dataset': 'DroneAerial', 'backbone_label': 'Swin-S', 'backbone': 'swin_s',
        'checkpoint': ('saved_models/fullmodel_bires_amhcsfi_res_hisym_crgpe_'
                       'swin_s_drone_naturalrng_seed2024_model_best.pth.tar'),
        'query_hw': (256, 256), 'sigma_y': 25.0, 'sigma_x': 25.0, 'outer_y': 50.0, 'outer_x': 50.0,
    },
    {
        'dataset': 'DroneAerial', 'backbone_label': 'Swin-B Projected', 'backbone': 'swin_b',
        'checkpoint': ('saved_models/fullmodel_bires_amhcsfi_res_hisym_crgpe_'
                       'swin_b_drone_naturalrng_seed2024_model_best.pth.tar'),
        'query_hw': (256, 256), 'sigma_y': 25.0, 'sigma_x': 25.0, 'outer_y': 50.0, 'outer_x': 50.0,
    },
    {
        'dataset': 'DroneAerial', 'backbone_label': 'Swin-B Native', 'backbone': 'swin_b_native',
        'checkpoint': ('saved_models/fullmodel_bires_amhcsfi_res_hisym_crgpe_'
                       'swin_b_native_drone_naturalrng_seed2024_model_best.pth.tar'),
        'query_hw': (256, 256), 'sigma_y': 25.0, 'sigma_x': 25.0, 'outer_y': 50.0, 'outer_x': 50.0,
    },
    {
        'dataset': 'CVOGL_SVI', 'backbone_label': 'ResNet-50', 'backbone': 'resnet50',
        'checkpoint': ('saved_models/fullmodel_bires_amhcsfi_res_hisym_crgpe_'
                       'core25x50_outer50x100_resnet50_svi_naturalrng_seed2024_model_best.pth.tar'),
        'query_hw': (256, 512), 'sigma_y': 25.0, 'sigma_x': 50.0, 'outer_y': 50.0, 'outer_x': 100.0,
    },
    {
        'dataset': 'CVOGL_SVI', 'backbone_label': 'Swin-T', 'backbone': 'swin_t',
        'checkpoint': ('saved_models/trogeo_ms_e4_amhcsfi_res_bi_hisym_crgpe_'
                       'corey25_corex50_outery50_outerx100_naturalrng_swin_t_svi_seed2024_model_best.pth.tar'),
        'query_hw': (256, 512), 'sigma_y': 25.0, 'sigma_x': 50.0, 'outer_y': 50.0, 'outer_x': 100.0,
    },
    {
        'dataset': 'CVOGL_SVI', 'backbone_label': 'Swin-S', 'backbone': 'swin_s',
        'checkpoint': ('saved_models/fullmodel_bires_amhcsfi_res_hisym_crgpe_'
                       'core25x50_outer50x100_swin_s_svi_naturalrng_seed2024_model_best.pth.tar'),
        'query_hw': (256, 512), 'sigma_y': 25.0, 'sigma_x': 50.0, 'outer_y': 50.0, 'outer_x': 100.0,
    },
    {
        'dataset': 'CVOGL_SVI', 'backbone_label': 'Swin-B Projected', 'backbone': 'swin_b',
        'checkpoint': ('saved_models/fullmodel_bires_amhcsfi_res_hisym_crgpe_'
                       'core25x50_outer50x100_swin_b_svi_naturalrng_seed2024_model_best.pth.tar'),
        'query_hw': (256, 512), 'sigma_y': 25.0, 'sigma_x': 50.0, 'outer_y': 50.0, 'outer_x': 100.0,
    },
    {
        'dataset': 'CVOGL_SVI', 'backbone_label': 'Swin-B Native', 'backbone': 'swin_b_native',
        'checkpoint': ('saved_models/fullmodel_bires_amhcsfi_res_hisym_crgpe_'
                       'core25x50_outer50x100_swin_b_native_svi_naturalrng_seed2024_model_best.pth.tar'),
        'query_hw': (256, 512), 'sigma_y': 25.0, 'sigma_x': 50.0, 'outer_y': 50.0, 'outer_x': 100.0,
    },
)


class OfflineSwinT(OriginalSwinTMultiStageEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.features = models.swin_t(weights=None).features


class OfflineSwinS(OriginalSwinSMultiStageEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.features = models.swin_s(weights=None).features


class OfflineSwinB(OriginalSwinBMultiStageEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.features = models.swin_b(weights=None).features
        self.stage3_projection = nn.Conv2d(512, 384, kernel_size=1)
        self.stage4_projection = nn.Conv2d(1024, 768, kernel_size=1)
        self._logged_sanity = False


class OfflineSwinBNative(OriginalSwinBNativeMultiStageEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.features = models.swin_b(weights=None).features


class OfflineResNet50(OriginalResNet50MultiStageEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        base = models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.stage3_projection = nn.Conv2d(1024, 384, kernel_size=1)
        self.stage4_projection = nn.Conv2d(2048, 768, kernel_size=1)
        self._logged_sanity = False


def patch_offline_encoders():
    ablation_module.SwinTMultiStageEncoder = OfflineSwinT
    ablation_module.SwinSMultiStageEncoder = OfflineSwinS
    ablation_module.SwinBMultiStageEncoder = OfflineSwinB
    ablation_module.SwinBNativeMultiStageEncoder = OfflineSwinBNative
    ablation_module.ResNet50MultiStageEncoder = OfflineResNet50


def build_model(case):
    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone=case['backbone'], variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe', gaussian_sigma=case['sigma_y'], gaussian_sigma_x=case['sigma_x'],
        crgpe_outer_sigma=case['outer_y'], crgpe_outer_sigma_x=case['outer_x'],
        dadpe_mode='none', amr_pe_mode='none', enable_hqs=False, enable_hqs_v2a=False,
        enable_hqs_v2b=False, enable_acr=False, unshared_backbone=False,
    )
    assert model.unshared_backbone is False
    assert hasattr(model, 'encoder')
    assert not hasattr(model, 'query_encoder') and not hasattr(model, 'reference_encoder')
    return model


def check_backbone(model, case):
    backbone = case['backbone']
    assert model.backbone_name == backbone
    assert model.variant == 'h2_ind_amhcsfi_res_bi'
    assert model.position_mode == 'hisym_crgpe'
    assert model.unshared_backbone is False
    if backbone in ('swin_t', 'swin_s'):
        assert (model.stage3_dim, model.stage4_dim) == (384, 768)
    elif backbone == 'swin_b':
        assert (model.stage3_dim, model.stage4_dim) == (384, 768)
        assert hasattr(model.encoder, 'stage3_projection') and hasattr(model.encoder, 'stage4_projection')
    elif backbone == 'swin_b_native':
        assert (model.stage3_dim, model.stage4_dim) == (512, 1024)
        assert model.native_swin_b is True
    elif backbone == 'resnet50':
        assert (model.stage3_dim, model.stage4_dim) == (384, 768)
        assert hasattr(model.encoder, 'stage3_projection') and hasattr(model.encoder, 'stage4_projection')
    else:
        raise RuntimeError('unexpected backbone: {}'.format(backbone))


def load_shared_swin_t_reference(path):
    if not os.path.isfile(path):
        raise FileNotFoundError('missing Shared Swin-T efficiency reference: {}'.format(path))
    with open(path, newline='', encoding='utf-8') as handle:
        rows = {(row['Dataset'], row['Method']): row for row in csv.DictReader(handle)}
    return {dataset: rows[(dataset, 'Full Model')] for dataset in ('DroneAerial', 'CVOGL_SVI')}


def write_outputs(rows, output_csv, output_md):
    fieldnames = ('Dataset', 'Backbone', 'Params_M', 'FLOPs_G', 'Checkpoint', 'Query_shape',
                  'Satellite_shape', 'Precision', 'Unsupported_ops', 'Uncalled_modules')
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.makedirs(os.path.dirname(output_md) or '.', exist_ok=True)
    with open(output_md, 'w', encoding='utf-8') as handle:
        handle.write('# Full Model backbone Params/FLOPs (CPU-only)\n\n')
        for dataset in ('DroneAerial', 'CVOGL_SVI'):
            handle.write('## {}\n\n| Backbone | Params (M) ↓ | FLOPs (G) ↓ |\n|---|---:|---:|\n'.format(dataset))
            for row in (row for row in rows if row['Dataset'] == dataset):
                handle.write('| {Backbone} | {Params_M:.2f} | {FLOPs_G:.2f} |\n'.format(**row))
            handle.write('\n')
        handle.write('Protocol: CPU-only; FP32; validation-best checkpoints loaded with strict=True; '
                     'fvcore convention uses 1 MAC = 1 FLOP.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_csv', default='results/backbone_params_flops.csv')
    parser.add_argument('--output_md', default='results/backbone_params_flops.md')
    parser.add_argument('--shared_reference_csv', default='results/efficiency_fullmodel_vs_detgeo.csv')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cpu':
        raise RuntimeError('This Params/FLOPs profiler is intended to run on CPU.')
    for case in CASES:
        if not os.path.isfile(case['checkpoint']):
            raise FileNotFoundError(case['checkpoint'])
    shared_reference = load_shared_swin_t_reference(args.shared_reference_csv)
    patch_offline_encoders()
    rows, params_by_backbone, flops_by_case = [], {}, {}
    for case in CASES:
        print('\n=== {} / {} ==='.format(case['dataset'], case['backbone_label']), flush=True)
        model = build_model(case)
        check_backbone(model, case)
        load_checkpoint_strict(model, case['checkpoint'])
        model = model.to(device).eval()
        query, satellite = make_images(case['query_hw'], device)
        click = make_gaussian_click_map(*case['query_hw'], case['sigma_y'], case['sigma_x'], device)
        with torch.inference_mode():
            model(query, satellite, click)
        assert model._logged_sanity
        flops, unsupported, uncalled = count_flops(FullModelProfileWrapper(model), (query, satellite, click))
        params = sum(parameter.numel() for parameter in model.parameters())
        row = {
            'Dataset': case['dataset'], 'Backbone': case['backbone_label'], 'Params_M': params / 1e6,
            'FLOPs_G': flops / 1e9, 'Checkpoint': case['checkpoint'],
            'Query_shape': '1x3x{}x{}'.format(*case['query_hw']), 'Satellite_shape': '1x3x1024x1024',
            'Precision': 'FP32 CPU', 'Unsupported_ops': format_ops(unsupported),
            'Uncalled_modules': ', '.join(uncalled) or 'none',
        }
        rows.append(row)
        params_by_backbone.setdefault(case['backbone'], {})[case['dataset']] = params
        flops_by_case[(case['dataset'], case['backbone'])] = flops
        print('[PASS] Params={:.2f}M FLOPs={:.2f}G'.format(row['Params_M'], row['FLOPs_G']), flush=True)

    for backbone, per_dataset in params_by_backbone.items():
        if per_dataset['DroneAerial'] != per_dataset['CVOGL_SVI']:
            raise RuntimeError('Params differ by dataset for {}'.format(backbone))
        if flops_by_case[('CVOGL_SVI', backbone)] <= flops_by_case[('DroneAerial', backbone)]:
            raise RuntimeError('SVI FLOPs must exceed Drone FLOPs for {}'.format(backbone))
    for row in rows:
        if row['Backbone'] == 'Swin-T':
            reference = shared_reference[row['Dataset']]
            if abs(row['Params_M'] - float(reference['Params_M'])) > 1e-6:
                raise RuntimeError('Swin-T Params mismatch for {}'.format(row['Dataset']))
            if abs(row['FLOPs_G'] - float(reference['FLOPs_G'])) > 1e-6:
                raise RuntimeError('Swin-T FLOPs mismatch for {}'.format(row['Dataset']))
    write_outputs(rows, args.output_csv, args.output_md)
    print('[PASS] wrote {} and {}'.format(args.output_csv, args.output_md), flush=True)


if __name__ == '__main__':
    main()
