from .DetGeo import DetGeo
from .adaptive_gaussian_field import AdaptiveGaussianField


class DetGeoAdaptiveGaussianField(DetGeo):
    """Only replace P generation; retain DetGeo's original click fusion and backend."""

    def __init__(self, emb_size=512, leaky=True, mode='msg', sigma_bank=(12, 20, 25, 35, 50),
                 base_sigma=25, context_scale=2.0, gamma_init=0.05, beta_init=0.05):
        super().__init__(emb_size=emb_size, leaky=leaky)
        self.gaussian_field = AdaptiveGaussianField(mode=mode, sigma_bank=sigma_bank, base_sigma=base_sigma,
                                                    context_scale=context_scale, gamma_init=gamma_init, beta_init=beta_init)

    def forward(self, query_imgs, reference_imgs, original_click_map, click_xy):
        del original_click_map
        position_map = self.gaussian_field(query_imgs, click_xy)
        return super().forward(query_imgs, reference_imgs, position_map.squeeze(1))
