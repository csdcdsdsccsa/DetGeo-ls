import os
import cv2
import numpy as np

from .data_loader import RSDataset


class AdaptiveSAMPromptDataset(RSDataset):
    def __init__(self, *args, sam_multimask_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        data_dir = os.path.join(self.data_root, self.data_name)
        self.sam_multimask_dir = sam_multimask_root or os.path.join(data_dir, 'sam_multimask', self.split_name)

    def __getitem__(self, idx):
        queryimg, rsimg, original_click_map, bbox, sample_index = super().__getitem__(idx)
        path = os.path.join(self.sam_multimask_dir, '{:06d}.npz'.format(idx))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            masks, scores = data['masks'], data['scores']
        if masks.ndim != 3 or scores.ndim != 1 or masks.shape[0] != scores.shape[0]:
            raise ValueError('invalid SAM multimask file: {}'.format(path))
        height, width = self.query_featuremap_hw
        masks = np.stack([cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                          for mask in masks]).astype(np.float32)
        if masks.max() > 1:
            masks /= 255.0
        click_xy = np.asarray(self.data_list[idx][4], dtype=np.float32)
        return queryimg, rsimg, original_click_map, click_xy, masks, scores.astype(np.float32), bbox, sample_index
