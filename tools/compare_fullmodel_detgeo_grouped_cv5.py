#!/usr/bin/env python3
"""Produce strictly paired Full-Model minus DetGeo deltas for matching CV folds."""

import csv
import json
import os
import statistics


METRICS = ('acc50', 'acc25', 'miou', 'center_acc')
DATASETS = ('CVOGL_DroneAerial', 'CVOGL_SVI')
SPLITS = ('cvval', 'cvtest')


def main():
    with open('results/grouped_cv5_summary.json', encoding='utf-8') as handle:
        full = json.load(handle)
    with open('results/detgeo_square_grouped_cv5_summary.json', encoding='utf-8') as handle:
        detgeo = json.load(handle)
    rows, summary = [], {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for split in SPLITS:
            summary[dataset][split] = {}
            for metric in METRICS:
                deltas = [left - right for left, right in zip(
                    full[dataset][split][metric]['fold_values'],
                    detgeo[dataset][split][metric]['fold_values'])]
                summary[dataset][split][metric] = {
                    'mean_delta': statistics.mean(deltas), 'sample_std': statistics.stdev(deltas),
                    'fold_deltas': deltas,
                }
                for fold, delta in enumerate(deltas, 1):
                    rows.append({'dataset': dataset, 'split': split, 'fold': fold,
                                 'metric': metric, 'full_minus_detgeo': delta})
    os.makedirs('results', exist_ok=True)
    with open('results/fullmodel_vs_detgeo_grouped_cv5_paired.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('dataset', 'split', 'fold', 'metric', 'full_minus_detgeo'))
        writer.writeheader(); writer.writerows(rows)
    with open('results/fullmodel_vs_detgeo_grouped_cv5_paired.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    for dataset in DATASETS:
        for split in SPLITS:
            print('\n{} {}: Full Model - DetGeo (mean ± sample std, %)'.format(dataset, split))
            for metric in METRICS:
                value = summary[dataset][split][metric]
                print('  {:<11} {:+.2f} ± {:.2f}'.format(metric, 100 * value['mean_delta'], 100 * value['sample_std']))


if __name__ == '__main__':
    main()
