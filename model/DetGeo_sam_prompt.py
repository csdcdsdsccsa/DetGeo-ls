from .DetGeo import DetGeo
from .prompt_fusion import PromptFusion


class DetGeoSAMPrompt(DetGeo):
    """Keep the original DetGeo YOLO head; add only a residual prompt adapter."""

    def __init__(self, emb_size=512, leaky=True):
        super().__init__(emb_size=emb_size, leaky=leaky)
        self.prompt_fusion = PromptFusion()

    def forward(self, query_imgs, reference_imgs, original_click_map, gaussian_map, sam_mask, masked_gaussian):
        original_click_map = original_click_map.unsqueeze(1)
        base_query_rgb = self.combine_clickptns_conv(self._cat_query_prompt(query_imgs, original_click_map))
        query_imgs = self.prompt_fusion(base_query_rgb, gaussian_map.unsqueeze(1), sam_mask.unsqueeze(1), masked_gaussian.unsqueeze(1))
        query_fvisu = self.query_resnet(query_imgs)
        reference_fvisu = self.reference_darknet(reference_imgs)[1]
        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)
        batch, channels, query_h, query_w = query_fvisu.shape
        _, _, reference_h, reference_w = reference_fvisu.shape
        query_gvisu = query_fvisu.view(batch, channels, query_h * query_w).mean(dim=2)
        fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
        return self.fcn_out(fused_features), attn_score.squeeze(1)

    @staticmethod
    def _cat_query_prompt(query_imgs, original_click_map):
        import torch
        return torch.cat((query_imgs, original_click_map), dim=1)
