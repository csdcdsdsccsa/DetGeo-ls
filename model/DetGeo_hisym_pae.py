import torch

from .DetGeo import DetGeo
from .hisym_pae_fusion import HiSymPAEFusion
from .sam_position_refiner import SAMPositionRefiner


class DetGeoHiSymPAE(DetGeo):
    """DetGeo with HiSymGeo-style residual PAE, optionally on SAM-refined P0."""

    def __init__(self, emb_size=512, leaky=True, use_sam_refinement=False,
                 preserve_downstream_rng=False):
        # Complete original initialization first so its RNG consumption is unchanged.
        super().__init__(emb_size=emb_size, leaky=leaky)
        self.use_sam_refinement = bool(use_sam_refinement)
        # The old click 1x1 fusion is deliberately not part of C-new/D-new.
        del self.combine_clickptns_conv
        rng_state = torch.get_rng_state().clone() if preserve_downstream_rng else None
        if preserve_downstream_rng:
            # Fixed candidate order makes D's SAM match B and C/D's PAE match.
            sam_candidate = SAMPositionRefiner()
            pae_candidate = HiSymPAEFusion()
            if self.use_sam_refinement:
                self.sam_refiner = sam_candidate
            self.pae_fusion = pae_candidate
            torch.set_rng_state(rng_state)
        else:
            if self.use_sam_refinement:
                self.sam_refiner = SAMPositionRefiner()
            self.pae_fusion = HiSymPAEFusion()

    def forward(self, query_imgs, reference_imgs, original_position_map,
                sam_masks=None, sam_scores=None):
        position_map = original_position_map.unsqueeze(1)
        if self.use_sam_refinement:
            if sam_masks is None or sam_scores is None:
                raise ValueError('SAM refinement requires masks and scores')
            position_map = self.sam_refiner(position_map, sam_masks, sam_scores)
        query_input = self.pae_fusion(query_imgs, position_map)
        query_fvisu = self.query_mapping_visu(self.query_resnet(query_input))
        reference_fvisu = self.reference_mapping_visu(self.reference_darknet(reference_imgs)[1])
        batch, channels, query_h, query_w = query_fvisu.shape
        query_gvisu = query_fvisu.view(batch, channels, query_h * query_w).mean(dim=2)
        fused, attn = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
        return self.fcn_out(fused), attn.squeeze(1)
