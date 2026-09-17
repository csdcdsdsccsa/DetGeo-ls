import os
import cv2
import numpy as np
from .trogeo_loader import TROGeoRSDataset


class TROGeoSAMMaskDataset(TROGeoRSDataset):
    """TROGeo data with an offline SAM mask aligned to query-side flipping."""
    def __init__(self, *args, sam_mask_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        data_dir = os.path.join(kwargs['data_root'], kwargs.get('data_name', 'CVOGL_DroneAerial'))
        candidate = os.path.join(sam_mask_root, self.split_name) if sam_mask_root else ''
        self.sam_mask_dir = candidate if candidate and os.path.isdir(candidate) else (sam_mask_root or os.path.join(data_dir, 'sam_mask', self.split_name))

    def __getitem__(self, index):
        query, satellite, click_map, bbox, sample_index = super().__getitem__(index)
        mask = cv2.imread(os.path.join(self.sam_mask_dir, '{:06d}.png'.format(index)), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError('missing SAM mask: {}'.format(os.path.join(self.sam_mask_dir, '{:06d}.png'.format(index))))
        height, width = click_map.shape
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        current_x = int(np.argmax(click_map)) % width
        original_x = int(self.data_list[index][4][0]); flipped_x = width - original_x - 1
        if current_x == flipped_x:
            mask = np.flip(mask, axis=1).copy()
        elif current_x != original_x:
            raise RuntimeError('cannot align SAM mask')
        return query, satellite, click_map, mask.astype(np.float32), bbox, sample_index
