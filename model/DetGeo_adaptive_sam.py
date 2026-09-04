import torch

from .DetGeo import DetGeo
from .adaptive_sam_prompt import AdaptiveSAMPrompt


class DetGeoAdaptiveSAM(DetGeo):
    def __init__(self, emb_size=512, leaky=True, preserve_downstream_rng=False,
                 base_sigma=25.0, sigma_min=8.0, sigma_max=50.0):
        super().__init__(emb_size=emb_size, leaky=leaky)
        state = torch.get_rng_state() if preserve_downstream_rng else None
        self.adaptive_prompt = AdaptiveSAMPrompt(base_sigma=base_sigma, sigma_min=sigma_min, sigma_max=sigma_max)
        if state is not None:
            torch.set_rng_state(state)

    def forward(self, query_imgs, reference_imgs, original_click_map, click_xy, sam_masks, sam_scores):
        del original_click_map
        position_map = self.adaptive_prompt(click_xy, sam_masks, sam_scores)
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, position_map), 1))
        query_fvisu = self.query_mapping_visu(self.query_resnet(query_imgs))
        reference_fvisu = self.reference_mapping_visu(self.reference_darknet(reference_imgs)[1])
        batch, channels, query_h, query_w = query_fvisu.shape
        query_gvisu = query_fvisu.view(batch, channels, query_h * query_w).mean(2)
        fused, attn = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
        return self.fcn_out(fused), attn.squeeze(1)
