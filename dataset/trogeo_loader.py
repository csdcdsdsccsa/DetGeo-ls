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
                 transform=None, augment=False, aug_mode='current', click_map_mode='distance', gaussian_sigma=25.0,
                 gaussian_sigma_x=None):
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
        if aug_mode not in ('current', 'detgeo'):
            raise ValueError('aug_mode must be current or detgeo, got {}'.format(aug_mode))
        self.aug_mode = aug_mode
        if click_map_mode not in ('distance', 'gaussian'):
            raise ValueError('click_map_mode must be distance or gaussian, got {}'.format(click_map_mode))
        if gaussian_sigma <= 0:
            raise ValueError('gaussian_sigma must be > 0')
        if gaussian_sigma_x is not None and gaussian_sigma_x <= 0:
            raise ValueError('gaussian_sigma_x must be > 0')
        self.click_map_mode = click_map_mode
        self.gaussian_sigma = float(gaussian_sigma)
        self.gaussian_sigma_x = self.gaussian_sigma if gaussian_sigma_x is None else float(gaussian_sigma_x)
        self.query_featuremap_hw = (256, 256) if data_name == 'CVOGL_DroneAerial' else (256, 512)
        if aug_mode == 'current':
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
        else:
            # Copied parameter-for-parameter from original DetGeo
            # RSDataset.rs_transform. Original DetGeo does not apply the
            # TROGeo query-side random horizontal flip below.
            self.rs_transform = A.Compose([
                A.RandomSizedBBoxSafeCrop(width=img_size, height=img_size, erosion_rate=0.2, p=0.2),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.OneOf([A.Blur(p=0.4), A.MedianBlur(p=0.3)], p=0.5),
                A.OneOf([A.RandomBrightnessContrast(p=0.4), A.CLAHE(p=0.3)], p=0.5),
                A.ToGray(p=0.2),
                A.RandomGamma(p=0.3),
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
        if self.aug_mode == 'current' and self.split_name == 'train' and random.choice([True, False]):
            query = torch.flip(query, dims=[-1])
            click_w = query.shape[-1] - click_w - 1
        click_map = self.make_click_map(self.query_featuremap_hw, click_h, click_w,
                                        self.click_map_mode, self.gaussian_sigma, self.gaussian_sigma_x)
        return query, satellite, click_map, bbox.astype(np.float32), index

    @staticmethod
    def make_click_map(shape, click_h, click_w, mode='distance', gaussian_sigma=25.0, gaussian_sigma_x=None):
        """Return the single float32 click channel used by the TROGeo front end."""
        height, width = shape
        rows = np.arange(height, dtype=np.float32)[:, None]
        cols = np.arange(width, dtype=np.float32)[None, :]
        dy2 = (rows - click_h) ** 2
        dx2 = (cols - click_w) ** 2
        if mode == 'gaussian':
            sigma_y = float(gaussian_sigma)
            sigma_x = sigma_y if gaussian_sigma_x is None else float(gaussian_sigma_x)
            click_map = np.exp(-(dy2 / (2.0 * sigma_y ** 2) + dx2 / (2.0 * sigma_x ** 2)))
        elif mode == 'distance':
            # Preserve the original isotropic distance-decay formulation.
            dist2 = dy2 + dx2
            norm = float((height * height + width * width) ** 0.5)
            click_map = (1.0 - np.sqrt(dist2) / norm) ** 2
        else:
            raise ValueError('click map mode must be distance or gaussian, got {}'.format(mode))
        return click_map.astype(np.float32)
