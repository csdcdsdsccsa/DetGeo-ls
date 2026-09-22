#!/usr/bin/env python3
"""Create deterministic image-disjoint grouped 5-fold CVOGL development splits.

The indivisible grouping unit is a connected component in the query--satellite
bipartite graph, rather than either image identity by itself.
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
from collections import Counter, defaultdict

import torch


DATASETS = ('CVOGL_DroneAerial', 'CVOGL_SVI')


class DSU:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        if value not in self.parent:
            self.parent[value] = value
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def digest(record):
    return hashlib.sha256(pickle.dumps(record, protocol=4)).hexdigest()


def read_split(data_dir, dataset, split):
    path = os.path.join(data_dir, '{}_{}.pth'.format(dataset, split))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location='cpu')


def identities(records):
    return {record[1] for record in records}, {record[2] for record in records}


def validate_record(record, dataset, index):
    if not isinstance(record, (tuple, list)) or len(record) < 8:
        raise ValueError('{} record {} is not an 8-field CVOGL record'.format(dataset, index))
    if not isinstance(record[1], str) or not isinstance(record[2], str):
        raise ValueError('{} record {} has invalid query/satellite identity'.format(dataset, index))


def create_dataset(data_root, dataset, n_splits, fold_seed, force):
    data_dir = os.path.join(data_root, dataset)
    original_train = read_split(data_dir, dataset, 'train')
    original_val = read_split(data_dir, dataset, 'val')
    official_test = read_split(data_dir, dataset, 'test')
    dev = list(original_train) + list(original_val)
    source = [('train', index) for index in range(len(original_train))] + [
        ('val', index) for index in range(len(original_val))]
    for index, record in enumerate(dev):
        validate_record(record, dataset, index)

    dsu = DSU()
    for record in dev:
        dsu.union('Q::' + record[1], 'S::' + record[2])
    component_indices = defaultdict(list)
    for index, record in enumerate(dev):
        component_indices[dsu.find('Q::' + record[1])].append(index)

    components = list(component_indices.values())
    if len(components) < n_splits:
        raise RuntimeError(
            '{} has only {} connected components; cannot create {} image-disjoint folds'.format(
                dataset, len(components), n_splits))
    random.Random(fold_seed).shuffle(components)
    components.sort(key=len, reverse=True)
    fold_indices = [[] for _ in range(n_splits)]
    fold_components = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    component_id_by_index = {}
    for component_id, indices in enumerate(components):
        fold_id = min(range(n_splits), key=lambda value: (fold_sizes[value], value))
        fold_indices[fold_id].extend(indices)
        fold_components[fold_id].append(component_id)
        fold_sizes[fold_id] += len(indices)
        for index in indices:
            component_id_by_index[index] = component_id

    out_dir = os.path.join(data_dir, 'cv5_grouped')
    if os.path.exists(out_dir) and os.listdir(out_dir) and not force:
        raise FileExistsError('{} exists; use --force only to intentionally regenerate it'.format(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    all_indices = set(range(len(dev)))
    all_val_indices = []
    folds = []
    assignments = []
    for fold_zero, val_unsorted in enumerate(fold_indices):
        val_indices = sorted(val_unsorted)
        train_indices = sorted(all_indices.difference(val_indices))
        train_records = [dev[index] for index in train_indices]
        val_records = [dev[index] for index in val_indices]
        train_query, train_satellite = identities(train_records)
        val_query, val_satellite = identities(val_records)
        assert train_query.isdisjoint(val_query)
        assert train_satellite.isdisjoint(val_satellite)
        assert not set(train_indices).intersection(val_indices)
        assert len(train_indices) + len(val_indices) == len(dev)
        fold = fold_zero + 1
        if not val_indices:
            raise RuntimeError('{} fold {} is empty after connected-component assignment'.format(dataset, fold))
        torch.save(train_records, os.path.join(out_dir, 'fold{}_train.pth'.format(fold)))
        torch.save(val_records, os.path.join(out_dir, 'fold{}_val.pth'.format(fold)))
        folds.append({
            'fold': fold,
            'train_count': len(train_indices), 'val_count': len(val_indices),
            'train_unique_query_count': len(train_query), 'val_unique_query_count': len(val_query),
            'train_unique_satellite_count': len(train_satellite), 'val_unique_satellite_count': len(val_satellite),
            'val_component_count': len(fold_components[fold_zero]),
            'train_dev_indices': train_indices, 'val_dev_indices': val_indices,
        })
        all_val_indices.extend(val_indices)
        for index in val_indices:
            record = dev[index]
            split, source_index = source[index]
            assignments.append({
                'dev_index': index, 'source_split': split, 'source_index': source_index,
                'query_name': record[1], 'satellite_name': record[2], 'class_name': record[7],
                'component_id': component_id_by_index[index], 'fold_id': fold,
            })
    assert sorted(all_val_indices) == list(range(len(dev)))
    duplicate_count = sum(count - 1 for count in Counter(digest(record) for record in dev).values() if count > 1)
    dev_query, dev_satellite = identities(dev)
    test_query, test_satellite = identities(official_test)
    dev_digests = {digest(record) for record in dev}
    test_digests = {digest(record) for record in official_test}
    manifest = {
        'dataset': dataset, 'fold_seed': fold_seed, 'n_splits': n_splits,
        'train_count_original': len(original_train), 'val_count_original': len(original_val),
        'test_count_original': len(official_test), 'development_count': len(dev),
        'unique_query_count': len(dev_query), 'unique_satellite_count': len(dev_satellite),
        'component_count': len(components), 'largest_component_size': max(map(len, components)),
        'duplicate_record_count': duplicate_count,
        'development_test_exact_record_overlap_count': len(dev_digests.intersection(test_digests)),
        'development_test_query_overlap_count': len(dev_query.intersection(test_query)),
        'development_test_satellite_overlap_count': len(dev_satellite.intersection(test_satellite)),
        'folds': folds,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'assignments.csv'), 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)
    print('{}: dev={} components={} largest_component={} folds={} sizes={} duplicates={} dev/test(records,Q,S)=({},{},{})'.format(
        dataset, len(dev), len(components), manifest['largest_component_size'], n_splits, fold_sizes, duplicate_count,
        manifest['development_test_exact_record_overlap_count'],
        manifest['development_test_query_overlap_count'], manifest['development_test_satellite_overlap_count']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--fold_seed', type=int, default=2024)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.n_splits != 5:
        raise ValueError('this protocol is fixed to grouped 5-fold')
    for dataset in args.datasets:
        create_dataset(args.data_root, dataset, args.n_splits, args.fold_seed, args.force)


if __name__ == '__main__':
    main()
