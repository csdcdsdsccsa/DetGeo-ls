#!/usr/bin/env python3
"""Summarize official-split HiSym-CRGPE Gaussian-scale sensitivity logs."""
import argparse
import csv
import json
import os
import re

TENSOR_RE = re.compile(r'tensor\(\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)')
CASES = (
    ('CVOGL_DroneAerial', 'drone', 15, 15, 15, 30, 30), ('CVOGL_DroneAerial', 'drone', 20, 20, 20, 40, 40), ('CVOGL_DroneAerial', 'drone', 25, 25, 25, 50, 50), ('CVOGL_DroneAerial', 'drone', 30, 30, 30, 60, 60), ('CVOGL_DroneAerial', 'drone', 35, 35, 35, 70, 70),
    ('CVOGL_SVI', 'svi', 15, 15, 30, 30, 60), ('CVOGL_SVI', 'svi', 20, 20, 40, 40, 80), ('CVOGL_SVI', 'svi', 25, 25, 50, 50, 100), ('CVOGL_SVI', 'svi', 30, 30, 60, 60, 120), ('CVOGL_SVI', 'svi', 35, 35, 70, 70, 140),
)
FIELDS = ('dataset', 'scale', 'core_y', 'core_x', 'outer_y', 'outer_x', 'split', 'acc50', 'acc25', 'miou', 'center_acc')

def read_metrics(path):
    with open(path, encoding='utf-8', errors='replace') as handle:
        values = [float(value) for value in TENSOR_RE.findall(handle.read())]
    if len(values) < 4: raise ValueError('missing final metrics: {}'.format(path))
    return values[-4:]

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--log_dir', default='results/gaussian_scale'); parser.add_argument('--output_dir', default='results'); args = parser.parse_args()
    rows = []
    for dataset, short, scale, cy, cx, oy, ox in CASES:
        prefix = 'fullmodel_gausscale_{}_g{}_seed2024'.format(short, scale)
        for split in ('val', 'test'):
            path = os.path.join(args.log_dir, '{}_{}.log'.format(prefix, split))
            if not os.path.isfile(path): continue
            acc50, acc25, miou, center_acc = read_metrics(path)
            rows.append(dict(dataset=dataset, scale='G{}'.format(scale), core_y=cy, core_x=cx, outer_y=oy, outer_x=ox, split=split, acc50=acc50, acc25=acc25, miou=miou, center_acc=center_acc))
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'gaussian_scale_summary.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    json_path = os.path.join(args.output_dir, 'gaussian_scale_summary.json')
    with open(json_path, 'w', encoding='utf-8') as handle: json.dump(rows, handle, ensure_ascii=False, indent=2)
    for dataset in ('CVOGL_DroneAerial', 'CVOGL_SVI'):
        for split in ('val', 'test'):
            selected = [row for row in rows if row['dataset'] == dataset and row['split'] == split]
            if not selected: continue
            print('\n{} {} Gaussian Scale Sensitivity'.format(dataset, split)); print('Scale  Core   Outer  Acc50  Acc25  mIoU   Center')
            for row in selected: print('{scale:<6} {core_y}x{core_x:<3} {outer_y}x{outer_x:<3} {acc50:.4f} {acc25:.4f} {miou:.4f} {center_acc:.4f}'.format(**row))
    print('Wrote {} and {}'.format(csv_path, json_path))

if __name__ == '__main__': main()
