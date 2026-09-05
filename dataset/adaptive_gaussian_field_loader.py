import numpy as np

from .data_loader import RSDataset


class AdaptiveGaussianFieldDataset(RSDataset):
    """Preserve RSDataset augmentation, then provide only the click coordinate."""

    def __getitem__(self, idx):
        queryimg, rsimg, original_click_map, bbox, sample_index = super().__getitem__(idx)
        click_xy = np.asarray(self.data_list[idx][4], dtype=np.float32)
        return queryimg, rsimg, original_click_map, click_xy, bbox, sample_index
