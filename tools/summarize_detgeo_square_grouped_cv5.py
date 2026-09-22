#!/usr/bin/env python3
"""Summarize legacy DetGeo Square grouped-CV results as mean plus sample std."""

import argparse
import csv
import json
import os
import re
import statistics


METRICS = ('acc50', 'acc25', 'miou', 'center_acc')
DATASETS = (('CVOGL_DroneAerial', 'drone'), ('CVOGL_SVI', 'svi'))
SPLITS = ('cvval', 'cvtest')
METRIC_RE = re.compile(r'(?:INFO\s+)?([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)')


def read_metrics(path):
    with open(path, encoding='utf-8', errors='replace') as handle:
        matches = [tuple(map(float, values)) for values in METRIC_RE.findall(handle.read())]
    if not matches:
        raise ValueError('{} does not contain final DetGeo metrics'.format(path))
    return matches[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='results/detgeo_square_grouped_cv5_eval')
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()
    rows, summaries = [], {}
    for dataset, short in DATASETS:
        summaries[dataset] = {}
        for split in SPLITS:
            folds = []
            for fold in range(1, 6):
                name = 'detgeo_square_groupcv5_{}_f{}_seed13_{}.log'.format(short, fold, split)
                metrics = read_metrics(os.path.join(args.log_dir, name))
                folds.append(metrics)
                rows.append({'dataset': dataset, 'split': split, 'fold': fold,
                             **dict(zip(METRICS, metrics))})
            summaries[dataset][split] = {
                metric: {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values),
                         'fold_values': values}
                for metric, values in ((metric, [fold[index] for fold in folds])
                                       for index, metric in enumerate(METRICS))
            }
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'detgeo_square_grouped_cv5_summary.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('dataset', 'split', 'fold', *METRICS))
        writer.writeheader(); writer.writerows(rows)
    json_path = os.path.join(args.output_dir, 'detgeo_square_grouped_cv5_summary.json')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    for dataset, _ in DATASETS:
        print('\n{} grouped CV5 (mean ± sample std, %)'.format(dataset))
        for split in SPLITS:
            print('  {}'.format(split))
            for metric in METRICS:
                value = summaries[dataset][split][metric]
                print('    {:<11} {:.2f} ± {:.2f}'.format(metric, 100 * value['mean'], 100 * value['sample_std']))
    print('\nWrote {} and {}'.format(csv_path, json_path))


if __name__ == '__main__':
    main()
