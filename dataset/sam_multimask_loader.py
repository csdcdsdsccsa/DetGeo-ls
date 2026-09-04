import os

import cv2
import numpy as np

from .data_loader import RSDataset


class SAMMultiMaskDataset(RSDataset):
    """Keep RSDataset's augmentation sequence, then append offline SAM candidates."""

    def __init__(self, *args, sam_multimask_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        data_dir = os.path.join(self.data_root, self.data_name)
        self.sam_multimask_dir = sam_multimask_root or os.path.join(data_dir, 'sam_multimask', self.split_name)

    def __getitem__(self, idx):
        queryimg, rsimg, original_position_map, bbox, sample_index = super().__getitem__(idx)
        path = os.path.join(self.sam_multimask_dir, '{:06d}.npz'.format(idx))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            masks = data['masks'].copy()
            scores = data['scores'].copy()
        if masks.ndim != 3 or scores.ndim != 1 or masks.shape[0] != scores.shape[0]:
            raise ValueError('invalid SAM multi-mask file: {}'.format(path))
        feature_h, feature_w = self.query_featuremap_hw
        resized = [cv2.resize(mask.astype(np.uint8), (feature_w, feature_h), interpolation=cv2.INTER_NEAREST)
                   for mask in masks]
        sam_masks = np.stack(resized, axis=0).astype(np.float32)
        if sam_masks.max() > 1.0:
            sam_masks /= 255.0
        return queryimg, rsimg, original_position_map, sam_masks, scores.astype(np.float32), np.asarray(bbox, dtype=np.float32), sample_index
