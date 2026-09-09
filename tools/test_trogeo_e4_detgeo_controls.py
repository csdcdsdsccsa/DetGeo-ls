"""GPU smoke checks for the two strict E4 DetGeo-control experiments."""

import argparse

import torch

from dataset.trogeo_loader import TROGeoRSDataset
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def check(aug_mode, position_mode, batch):
    dataset = TROGeoRSDataset('data', split_name='train', augment=True, aug_mode=aug_mode)
    if dataset.aug_mode != aug_mode:
        raise RuntimeError('dataset augmentation mode did not persist')
    model = torch.nn.DataParallel(TROGeoMSDetectionAblation(
        variant='h2_ind', position_mode=position_mode)).cuda().eval()
    with torch.no_grad():
        predictions, _ = model(
            torch.randn(batch, 3, 256, 256, device='cuda'),
            torch.randn(batch, 3, 1024, 1024, device='cuda'),
            torch.rand(batch, 256, 256, device='cuda'))
    module = model.module
    if set(predictions) != {'stage3', 'stage4'}:
        raise RuntimeError('{} / {} is not strict two-scale'.format(aug_mode, position_mode))
    if any(value.shape != (batch, 45, 64, 64) for value in predictions.values()):
        raise RuntimeError('{} / {} emitted invalid output shape'.format(aug_mode, position_mode))
    if hasattr(module, 'cvopm_stage2') or hasattr(module, 'det_head_stage2'):
        raise RuntimeError('{} / {} unexpectedly constructed Stage2'.format(aug_mode, position_mode))
    expected = 'DetGeoPositionEmbedding' if position_mode == 'detgeo' else 'Sequential'
    if type(module.position_embedding).__name__ != expected:
        raise RuntimeError('wrong position encoder: {}'.format(type(module.position_embedding).__name__))
    print('e4_detgeo_control_sanity aug_mode={} position_mode={} output=strict_stage3_stage4'.format(
        aug_mode, position_mode), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=7)
    args = parser.parse_args()
    check('detgeo', 'current', args.batch_size)
    check('current', 'detgeo', args.batch_size)


if __name__ == '__main__':
    main()
