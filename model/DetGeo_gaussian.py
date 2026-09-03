from .DetGeo import DetGeo


class DetGeoGaussian(DetGeo):
    """DetGeo with only its square click map replaced by a Gaussian map."""

    def forward(self, query_imgs, reference_imgs, original_click_map, gaussian_map):
        del original_click_map
        return super().forward(query_imgs, reference_imgs, gaussian_map)
