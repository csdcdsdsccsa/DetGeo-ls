#!/usr/bin/env python3
"""Assert the image-disjoint grouped CVOGL CV5 protocol and print its audit."""

import argparse
import hashlib
import json
import os
import pickle

import torch

from make_cvogl_grouped_cv5 import DATASETS, DSU, identities, read_split, validate_record


def digest(record):
    return hashlib.sha256(pickle.dumps(record, protocol=4)).hexdigest()


def check_dataset(data_root, dataset):
    data_dir = os.path.join(data_root, dataset)
    out_dir = os.path.join(data_dir, 'cv5_grouped')
    with open(os.path.join(out_dir, 'manifest.json'), encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest['dataset'] != dataset or manifest['n_splits'] != 5:
        raise AssertionError('invalid manifest for {}'.format(dataset))
    original_train = read_split(data_dir, dataset, 'train')
    original_val = read_split(data_dir, dataset, 'val')
    official_test = read_split(data_dir, dataset, 'test')
    dev = list(original_train) + list(original_val)
    for index, record in enumerate(dev):
        validate_record(record, dataset, index)
    dsu = DSU()
    for record in dev:
        dsu.union('Q::' + record[1], 'S::' + record[2])
    component_by_index = {index: dsu.find('Q::' + record[1]) for index, record in enumerate(dev)}
    all_val = []
    val_component_fold = {}
    for fold in manifest['folds']:
        number = fold['fold']
        train_path = os.path.join(out_dir, 'fold{}_train.pth'.format(number))
        val_path = os.path.join(out_dir, 'fold{}_val.pth'.format(number))
        assert os.path.isfile(train_path) and os.path.isfile(val_path)
        train_indices, val_indices = fold['train_dev_indices'], fold['val_dev_indices']
        assert not set(train_indices).intersection(val_indices)
        assert set(train_indices).union(val_indices) == set(range(len(dev)))
        expected_train = [dev[index] for index in train_indices]
        expected_val = [dev[index] for index in val_indices]
        actual_train = torch.load(train_path, map_location='cpu')
        actual_val = torch.load(val_path, map_location='cpu')
        assert [digest(record) for record in actual_train] == [digest(record) for record in expected_train]
        assert [digest(record) for record in actual_val] == [digest(record) for record in expected_val]
        train_query, train_satellite = identities(actual_train)
        val_query, val_satellite = identities(actual_val)
        assert train_query.isdisjoint(val_query)
        assert train_satellite.isdisjoint(val_satellite)
        for index in val_indices:
            component = component_by_index[index]
            previous = val_component_fold.setdefault(component, number)
            assert previous == number
        all_val.extend(val_indices)
        print('{} fold {}: train={} val={} query_overlap=0 satellite_overlap=0'.format(
            dataset, number, len(actual_train), len(actual_val)))
    assert sorted(all_val) == list(range(len(dev)))
    dev_query, dev_satellite = identities(dev)
    test_query, test_satellite = identities(official_test)
    print('{} PASS: val coverage={} component_leakage=0 official_test_records=0 dev/test(Q,S)=({},{})'.format(
        dataset, len(all_val), len(dev_query.intersection(test_query)), len(dev_satellite.intersection(test_satellite))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for dataset in args.datasets:
        check_dataset(args.data_root, dataset)


if __name__ == '__main__':
    main()
