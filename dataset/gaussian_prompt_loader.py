import numpy as np

from .data_loader import RSDataset


class GaussianPromptDataset(RSDataset):
    """Original DetGeo samples with a Gaussian click map and no SAM input."""

    def __init__(self, *args, gaussian_sigma=25.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian_sigma = float(gaussian_sigma)

    def __getitem__(self, idx):
        queryimg, rsimg, original_click_map, bbox, sample_index = super().__getitem__(idx)
        feature_h, feature_w = self.query_featuremap_hw
        _, _, _, _, click_xy, _, _, _ = self.data_list[idx]
        click_x, click_y = float(click_xy[0]), float(click_xy[1])
        yy, xx = np.meshgrid(
            np.arange(feature_h, dtype=np.float32),
            np.arange(feature_w, dtype=np.float32),
            indexing="ij",
        )
        gaussian_map = np.exp(
            -((xx - click_x) ** 2 + (yy - click_y) ** 2)
            / (2.0 * self.gaussian_sigma ** 2)
        ).astype(np.float32)
        return queryimg, rsimg, original_click_map, gaussian_map, bbox, sample_index
