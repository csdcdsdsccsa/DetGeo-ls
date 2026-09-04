import torch

from .DetGeo import DetGeo
from .rgb_position_fusion import RGBPositionFusion
from .sam_position_refiner import SAMPositionRefiner


class DetGeoPromptInteraction(DetGeo):
    """P10-compatible SAM P0 refiner and/or RGB-position interaction ablation."""

    def __init__(self, emb_size=512, leaky=True, use_sam_refinement=False,
                 use_rgbp_interaction=False, preserve_downstream_rng=False):
        super().__init__(emb_size=emb_size, leaky=leaky)
        self.use_sam_refinement = bool(use_sam_refinement)
        self.use_rgbp_interaction = bool(use_rgbp_interaction)
        rng_state = torch.get_rng_state().clone() if preserve_downstream_rng else None
        if preserve_downstream_rng:
            # Construct candidates in the same order in B/C/D, then retain only needed modules.
            sam_candidate = SAMPositionRefiner()
            rgbp_candidate = RGBPositionFusion()
            if self.use_sam_refinement:
                self.sam_refiner = sam_candidate
            if self.use_rgbp_interaction:
                self.rgbp_interaction = rgbp_candidate
            torch.set_rng_state(rng_state)
        else:
            if self.use_sam_refinement:
                self.sam_refiner = SAMPositionRefiner()
            if self.use_rgbp_interaction:
                self.rgbp_interaction = RGBPositionFusion()

    def forward(self, query_imgs, reference_imgs, original_position_map,
                sam_masks=None, sam_scores=None):
        position_map = original_position_map.unsqueeze(1)
        if self.use_sam_refinement:
            if sam_masks is None or sam_scores is None:
                raise ValueError('SAM refinement requires masks and scores')
            position_map = self.sam_refiner(position_map, sam_masks, sam_scores)
        base_query = self.combine_clickptns_conv(torch.cat((query_imgs, position_map), dim=1))
        query_input = self.rgbp_interaction(query_imgs, position_map, base_query) if self.use_rgbp_interaction else base_query
        query_fvisu = self.query_mapping_visu(self.query_resnet(query_input))
        reference_fvisu = self.reference_mapping_visu(self.reference_darknet(reference_imgs)[1])
        batch, channels, query_h, query_w = query_fvisu.shape
        query_gvisu = query_fvisu.view(batch, channels, query_h * query_w).mean(dim=2)
        fused, attn = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
        return self.fcn_out(fused), attn.squeeze(1)
