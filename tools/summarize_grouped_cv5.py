#!/usr/bin/env python3
"""Summarize the ten logged grouped-CV evaluations as mean plus sample std."""

import argparse
import csv
import json
import os
import re
import statistics


METRICS = ('acc50', 'acc25', 'miou', 'center_acc')
DATASETS = (
    ('CVOGL_DroneAerial', 'drone'),
    ('CVOGL_SVI', 'svi'),
)
SPLITS = ('cvval', 'cvtest')
TENSOR_RE = re.compile(r'tensor\(\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)')


def read_metrics(path):
    with open(path, encoding='utf-8', errors='replace') as handle:
        values = [float(value) for value in TENSOR_RE.findall(handle.read())]
    if len(values) < 4:
        raise ValueError('{} does not contain the four final metrics'.format(path))
    return values[-4:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='results/grouped_cv5_eval')
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()

    rows = []
    summaries = {}
    for dataset, short_name in DATASETS:
        summaries[dataset] = {}
        for split in SPLITS:
            fold_metrics = []
            for fold in range(1, 6):
                name = 'fullmodel_swin_t_groupcv5_{}_f{}_seed2024_{}.log'.format(short_name, fold, split)
                path = os.path.join(args.log_dir, name)
                metrics = read_metrics(path)
                fold_metrics.append(metrics)
                rows.append({'dataset': dataset, 'split': split, 'fold': fold,
                             **dict(zip(METRICS, metrics))})
            metric_summary = {}
            for index, metric in enumerate(METRICS):
                values = [fold[index] for fold in fold_metrics]
                metric_summary[metric] = {
                    'mean': statistics.mean(values),
                    'sample_std': statistics.stdev(values),
                    'fold_values': values,
                }
            summaries[dataset][split] = metric_summary

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'grouped_cv5_summary.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('dataset', 'split', 'fold', *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(args.output_dir, 'grouped_cv5_summary.json')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    for dataset, _ in DATASETS:
        print('\n{} Grouped CV5'.format(dataset))
        for split in SPLITS:
            print('  {} (mean ± sample std, %)'.format('Validation' if split == 'cvval' else 'Official Test'))
            for metric in METRICS:
                summary = summaries[dataset][split][metric]
                print('    {:<11} {:.2f} ± {:.2f}'.format(
                    metric, 100 * summary['mean'], 100 * summary['sample_std']))
    print('\nWrote {} and {}'.format(csv_path, json_path))


if __name__ == '__main__':
    main()
