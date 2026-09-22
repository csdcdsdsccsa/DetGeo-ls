#!/usr/bin/env python3
"""One real loader and Full Model forward/backward per grouped-CV dataset."""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from dataset.trogeo_loader import TROGeoRSDataset
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation


def run(dataset, gaussian_sigma_x=None, outer_sigma_x=None):
    split = 'data/{}/cv5_grouped/fold1_train.pth'.format(dataset)
    data = TROGeoRSDataset('data', dataset, split_name='train', split_pth=split,
                           img_size=1024, transform=Compose([ToTensor()]), augment=False,
                           aug_mode='current', click_map_mode='gaussian', gaussian_sigma=25.0,
                           gaussian_sigma_x=gaussian_sigma_x)
    query, satellite, click_map, _, _ = next(iter(DataLoader(data, batch_size=1, shuffle=False, num_workers=0)))
    model = TROGeoMSDetectionAblation(
        emb_size=768, backbone='swin_t', variant='h2_ind_amhcsfi_res_bi',
        position_mode='hisym_crgpe', gaussian_sigma=25.0, gaussian_sigma_x=gaussian_sigma_x,
        crgpe_outer_sigma=50.0, crgpe_outer_sigma_x=outer_sigma_x).cuda().train()
    outputs, _ = model(query.cuda(), satellite.cuda(), click_map.cuda())
    loss = outputs['stage3'].mean() + outputs['stage4'].mean()
    loss.backward()
    print('{} loader+forward+backward PASS: q={} r={} click={} p3={} p4={}'.format(
        dataset, tuple(query.shape), tuple(satellite.shape), tuple(click_map.shape),
        tuple(outputs['stage3'].shape), tuple(outputs['stage4'].shape)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('CVOGL_DroneAerial', 'CVOGL_SVI'), required=True)
    args = parser.parse_args()
    if args.dataset == 'CVOGL_SVI':
        run(args.dataset, gaussian_sigma_x=50.0, outer_sigma_x=100.0)
    else:
        run(args.dataset)
