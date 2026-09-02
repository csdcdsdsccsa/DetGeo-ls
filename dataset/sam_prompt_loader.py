import os

import cv2
import numpy as np

from .data_loader import RSDataset


class SAMPromptDataset(RSDataset):
    """Original DetGeo samples plus offline SAM and Gaussian prompt maps."""

    def __init__(self, *args, sam_mask_root=None, gaussian_sigma=25.0, **kwargs):
        super().__init__(*args, **kwargs)
        data_dir = os.path.join(self.data_root, self.data_name)
        self.sam_mask_dir = sam_mask_root or os.path.join(data_dir, "sam_mask", self.split_name)
        self.gaussian_sigma = float(gaussian_sigma)

    def __getitem__(self, idx):
        queryimg, rsimg, original_click_map, bbox, sample_index = super().__getitem__(idx)
        mask_path = os.path.join(self.sam_mask_dir, "{0:06d}.png".format(idx))
        sam_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if sam_mask is None:
            raise FileNotFoundError("missing SAM mask for {0}/{1}: {2}".format(self.split_name, idx, mask_path))

        feature_h, feature_w = self.query_featuremap_hw
        sam_mask = cv2.resize(sam_mask, (feature_w, feature_h), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        _, _, _, _, click_xy, _, _, _ = self.data_list[idx]
        click_x, click_y = float(click_xy[0]), float(click_xy[1])
        yy, xx = np.meshgrid(np.arange(feature_h, dtype=np.float32), np.arange(feature_w, dtype=np.float32), indexing="ij")
        gaussian_map = np.exp(-((xx - click_x) ** 2 + (yy - click_y) ** 2) / (2.0 * self.gaussian_sigma ** 2)).astype(np.float32)
        masked_gaussian = gaussian_map * sam_mask
        return queryimg, rsimg, original_click_map, gaussian_map, sam_mask, masked_gaussian, bbox, sample_index
