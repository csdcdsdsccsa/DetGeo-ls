"""VIGOR-Building loader following the supplied HiSymGeo split protocol."""

import os
import xml.etree.ElementTree as ET
from numbers import Integral

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)


class VigorBuildingDataset(Dataset):
    """One target building per split entry; no random split construction."""

    def __init__(self, data_root, split_pth, img_size=640, ground_size=(512, 256),
                 transform=None, augment=False, gaussian_sigma=25.0, gaussian_sigma_x=50.0):
        if not os.path.isfile(split_pth):
            raise FileNotFoundError('VIGOR-Building split does not exist: {}'.format(split_pth))
        if img_size != 640 or tuple(ground_size) != (512, 256):
            raise ValueError('VIGOR-Building protocol requires satellite=640 and ground=(512,256)')
        self.data_root, self.data_list = data_root, torch.load(split_pth, map_location='cpu')
        self.sat_size, self.ground_w, self.ground_h = img_size, ground_size[0], ground_size[1]
        self.transform, self.augment = transform, augment
        self.gaussian_sigma, self.gaussian_sigma_x = float(gaussian_sigma), float(gaussian_sigma_x)
        if self.gaussian_sigma <= 0 or self.gaussian_sigma_x <= 0:
            raise ValueError('Gaussian sigmas must be positive')
        self.rs_transform = A.Compose([
            A.RandomSizedBBoxSafeCrop(width=640, height=640, erosion_rate=0.2, p=0.2),
            A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.OneOf([A.Blur(p=0.4), A.MedianBlur(p=0.3)], p=0.5),
            A.OneOf([A.RandomBrightnessContrast(p=0.4), A.CLAHE(p=0.3)], p=0.5),
            A.ToGray(p=0.2), A.RandomGamma(p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc'))

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def _read_rgb(path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _parse_voc_xml(path):
        root = ET.parse(path).getroot()
        size = root.find('size')
        if size is None:
            raise ValueError('VOC annotation has no size: {}'.format(path))
        objects = []
        for obj in root.findall('object'):
            name, box = obj.findtext('name'), obj.find('bndbox')
            if name is None or box is None:
                continue
            objects.append((name.strip(), np.array([
                float(box.findtext('xmin')), float(box.findtext('ymin')),
                float(box.findtext('xmax')), float(box.findtext('ymax'))], dtype=np.float32)))
        return int(size.findtext('width')), int(size.findtext('height')), objects

    @staticmethod
    def _scale_box(box, src_w, src_h, dst_w, dst_h):
        scale = np.array([dst_w / float(src_w), dst_h / float(src_h)] * 2, dtype=np.float32)
        return box.astype(np.float32) * scale

    @staticmethod
    def _find_box(objects, name):
        for candidate, box in objects:
            if candidate.strip() == name:
                return box.copy()
        raise KeyError('target building {!r} is absent from paired satellite annotation'.format(name))

    @staticmethod
    def _resolve_target_name(target_idx, ground_objects, satellite_objects):
        """Match HiSymGeo's name -> 1-based -> 0-based target resolution."""
        target_name = str(target_idx).strip()
        ground_names = {str(name).strip() for name, _ in ground_objects}
        satellite_names = {str(name).strip() for name, _ in satellite_objects}
        common_names = ground_names & satellite_names
        if target_name in common_names:
            return target_name

        if isinstance(target_idx, Integral):
            index = int(target_idx)
            if 1 <= index <= len(ground_objects):
                candidate = str(ground_objects[index - 1][0]).strip()
                if candidate in common_names:
                    return candidate
            if 0 <= index < len(ground_objects):
                candidate = str(ground_objects[index][0]).strip()
                if candidate in common_names:
                    return candidate
        raise KeyError('cannot resolve target_idx={!r} against paired VOC objects'.format(target_idx))

    def _click_map(self, cx, cy):
        yy = np.arange(self.ground_h, dtype=np.float32)[:, None]
        xx = np.arange(self.ground_w, dtype=np.float32)[None, :]
        return np.exp(-((yy - cy) ** 2 / (2.0 * self.gaussian_sigma ** 2) +
                        (xx - cx) ** 2 / (2.0 * self.gaussian_sigma_x ** 2))).astype(np.float32)

    def __getitem__(self, index):
        city, ground_img, ground_xml, satellite_img, satellite_xml, target_idx = self.data_list[index]
        ground_base = os.path.join(self.data_root, city, 'query')
        sat_base = os.path.join(self.data_root, city, 'satellite')
        ground = self._read_rgb(os.path.join(ground_base, 'images', ground_img))
        satellite = self._read_rgb(os.path.join(sat_base, 'images', satellite_img))
        gw, gh, ground_objects = self._parse_voc_xml(os.path.join(ground_base, 'labels', ground_xml))
        sw, sh, satellite_objects = self._parse_voc_xml(os.path.join(sat_base, 'labels', satellite_xml))
        target_name = self._resolve_target_name(target_idx, ground_objects, satellite_objects)
        ground_box = self._find_box(ground_objects, target_name)
        sat_box = self._find_box(satellite_objects, target_name)
        ground = cv2.resize(ground, (self.ground_w, self.ground_h), interpolation=cv2.INTER_LINEAR)
        satellite = cv2.resize(satellite, (self.sat_size, self.sat_size), interpolation=cv2.INTER_LINEAR)
        ground_box = self._scale_box(ground_box, gw, gh, self.ground_w, self.ground_h)
        sat_box = self._scale_box(sat_box, sw, sh, self.sat_size, self.sat_size)
        if self.augment:
            transformed = self.rs_transform(image=satellite, bboxes=[sat_box.tolist()])
            satellite = transformed['image']
            if not transformed['bboxes']:
                raise RuntimeError('VIGOR augmentation removed the target box')
            sat_box = np.asarray(transformed['bboxes'][0][:4], dtype=np.float32)
        cx = int(np.clip(np.rint(0.5 * (ground_box[0] + ground_box[2])), 0, self.ground_w - 1))
        cy = int(np.clip(np.rint(0.5 * (ground_box[1] + ground_box[3])), 0, self.ground_h - 1))
        click = self._click_map(cx, cy)
        if self.transform is not None:
            ground, satellite = self.transform(ground.copy()), self.transform(satellite.copy())
        return ground, satellite, torch.tensor(click), torch.tensor(sat_box), index
