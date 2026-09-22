#!/usr/bin/env python3
"""Prove that legacy RSDataset reads the existing grouped-CV split files."""

import json
import os

from torchvision.transforms import Compose, ToTensor

from dataset.data_loader import RSDataset


DATASETS = ('CVOGL_DroneAerial', 'CVOGL_SVI')


def check_dataset(data_root, dataset):
    split_dir = os.path.join(data_root, dataset, 'cv5_grouped')
    with open(os.path.join(split_dir, 'manifest.json'), encoding='utf-8') as handle:
        manifest = json.load(handle)
    fold = manifest['folds'][0]
    train_path = os.path.join(split_dir, 'fold1_train.pth')
    val_path = os.path.join(split_dir, 'fold1_val.pth')
    transform = Compose([ToTensor()])
    train = RSDataset(data_root, dataset, split_name='train', split_pth=train_path,
                      img_size=1024, transform=transform, augment=True)
    val = RSDataset(data_root, dataset, split_name='val', split_pth=val_path,
                    img_size=1024, transform=transform, augment=False)
    legacy = RSDataset(data_root, dataset, split_name='train', split_pth=None,
                       img_size=1024, transform=transform, augment=False)
    assert len(train) == fold['train_count']
    assert len(val) == fold['val_count']
    assert len(legacy) == manifest['train_count_original']
    assert train.data_list[0] != val.data_list[0]
    print('{} RSDataset custom split PASS: train={} val={} legacy_train={} first_pair=({}, {})'.format(
        dataset, len(train), len(val), len(legacy), train.data_list[0][1], train.data_list[0][2]))
    print('  train_path={}'.format(train_path))
    print('  val_path={}'.format(val_path))


if __name__ == '__main__':
    for dataset_name in DATASETS:
        check_dataset('data', dataset_name)
