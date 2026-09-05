"""Summarize learned content-adaptive Gaussian-field gates on a fixed split."""
import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from dataset.adaptive_gaussian_field_loader import AdaptiveGaussianFieldDataset
from model.DetGeo_adaptive_gaussian_field import DetGeoAdaptiveGaussianField


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=('msg', 'msg_ag', 'full'), required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--data_name', default='CVOGL_DroneAerial')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gaussian_bank', default='12,20,25,35,50')
    args = parser.parse_args()

    bank = tuple(float(v) for v in args.gaussian_bank.split(','))
    transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = AdaptiveGaussianFieldDataset(args.data_root, args.data_name, args.split, 1024, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    model = torch.nn.DataParallel(DetGeoAdaptiveGaussianField(mode=args.mode, sigma_bank=bank)).cuda()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    records = {key: [] for key in ('alpha', 'gamma', 'a', 'theta', 'beta', 'core', 'context')}
    with torch.no_grad():
        for query, _, _, click_xy, _, _ in loader:
            _, aux = model.module.gaussian_field(query.cuda(), click_xy.cuda(), return_aux=True)
            for key in records:
                if key in aux:
                    records[key].append(aux[key].detach().float().cpu())
    alpha = torch.cat(records['alpha'])
    output = {
        'split': args.split,
        'count': int(alpha.shape[0]),
        'alpha_mean_per_sigma': dict(zip(map(str, bank), alpha.mean(0).tolist())),
        'alpha_argmax_distribution': dict(zip(map(str, bank), torch.bincount(alpha.argmax(1), minlength=len(bank)).div(alpha.shape[0]).tolist())),
    }
    for key in ('gamma', 'beta'):
        if records[key]:
            value = torch.cat(records[key])
            output[key] = {'mean': value.mean().item(), 'std': value.std(unbiased=False).item()}
    if records['a']:
        a, theta = torch.cat(records['a']), torch.cat(records['theta'])
        output['aspect_ratio'] = {'mean': torch.exp(2 * a.abs()).mean().item(), 'std': torch.exp(2 * a.abs()).std(unbiased=False).item()}
        output['theta_degrees_abs'] = {'mean': torch.rad2deg(theta.abs()).mean().item(), 'std': torch.rad2deg(theta.abs()).std(unbiased=False).item()}
    if records['context']:
        core, context = torch.cat(records['core']), torch.cat(records['context'])
        output['context_to_core_mass'] = context.sum().div(core.sum().clamp_min(1e-6)).item()
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
