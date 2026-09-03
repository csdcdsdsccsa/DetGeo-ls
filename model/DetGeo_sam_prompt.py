import torch

from .DetGeo import DetGeo
from .prompt_fusion import PromptFusion


class DetGeoSAMPrompt(DetGeo):
    """Replace only DetGeo's square position map; retain the original YOLO path."""

    def __init__(self, emb_size=512, leaky=True, preserve_downstream_rng=False):
        super().__init__(emb_size=emb_size, leaky=leaky)
        torch_rng_state = torch.get_rng_state() if preserve_downstream_rng else None
        self.prompt_fusion = PromptFusion()
        if torch_rng_state is not None:
            # PromptFusion retains its initialized weights, while the global
            # CPU RNG resumes at the same post-DetGeo state as Square/Gaussian.
            torch.set_rng_state(torch_rng_state)

    def forward(self, query_imgs, reference_imgs, original_click_map, gaussian_map, sam_mask, masked_gaussian):
        del original_click_map
        new_position_map = self.prompt_fusion(
            gaussian_map.unsqueeze(1), sam_mask.unsqueeze(1), masked_gaussian.unsqueeze(1)
        )
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, new_position_map), dim=1))
        query_fvisu = self.query_resnet(query_imgs)
        reference_fvisu = self.reference_darknet(reference_imgs)[1]
        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)
        batch, channels, query_h, query_w = query_fvisu.shape
        query_gvisu = query_fvisu.view(batch, channels, query_h * query_w).mean(dim=2)
        fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
        return self.fcn_out(fused_features), attn_score.squeeze(1)
