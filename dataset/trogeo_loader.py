"""TROGeo geometry-only data loader, without CVOGL-Seg masks or OST targets."""

import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)


class TROGeoRSDataset(Dataset):
    def __init__(self, data_root, data_name='CVOGL_DroneAerial', split_name='train', img_size=1024,
                 transform=None, augment=False):
        if data_name not in ('CVOGL_DroneAerial', 'CVOGL_SVI'):
            raise ValueError('unsupported data_name: {}'.format(data_name))
        data_dir = os.path.join(data_root, data_name)
        self.data_list = torch.load(os.path.join(data_dir, '{}_{}.pth'.format(data_name, split_name)))
        self.queryimg_dir = os.path.join(data_dir, 'query')
        self.rsimg_dir = os.path.join(data_dir, 'satellite')
        self.img_size = img_size
        self.split_name = split_name
        self.transform = transform
        self.augment = augment
        self.query_featuremap_hw = (256, 256) if data_name == 'CVOGL_DroneAerial' else (256, 512)
        self.rs_transform = A.Compose([
            A.RandomSizedBBoxSafeCrop(width=img_size, height=img_size, erosion_rate=0.2, p=0.4),
            A.OneOf([
                A.RandomRotate90(p=1.0),
                A.Rotate(limit=[180, 180], p=1.0),
                A.Rotate(limit=[270, 270], p=1.0),
            ], p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        _, query_name, satellite_name, _, click_xy, bbox, _, class_name = self.data_list[index]
        bbox = (np.asarray(bbox, dtype=np.float32) / (1024.0 / self.img_size)).astype(np.float32)
        query = cv2.cvtColor(cv2.imread(os.path.join(self.queryimg_dir, query_name)), cv2.COLOR_BGR2RGB)
        satellite = cv2.cvtColor(cv2.imread(os.path.join(self.rsimg_dir, satellite_name)), cv2.COLOR_BGR2RGB)
        if self.augment:
            transformed = self.rs_transform(image=satellite, bboxes=[list(bbox) + [class_name]])
            satellite = transformed['image']
            bbox = np.asarray(transformed['bboxes'][0][:4], dtype=np.float32)
        if self.transform is not None:
            query = self.transform(query.copy())
            satellite = self.transform(satellite.copy())
        click_h, click_w = int(click_xy[1]), int(click_xy[0])
        if self.split_name == 'train' and random.choice([True, False]):
            query = torch.flip(query, dims=[-1])
            click_w = query.shape[-1] - click_w - 1
        height, width = self.query_featuremap_hw
        rows = np.arange(height, dtype=np.float32)[:, None]
        cols = np.arange(width, dtype=np.float32)[None, :]
        norm = float((height * height + width * width) ** 0.5)
        click_map = (1.0 - np.sqrt((rows - click_h) ** 2 + (cols - click_w) ** 2) / norm) ** 2
        return query, satellite, click_map.astype(np.float32), bbox.astype(np.float32), index
