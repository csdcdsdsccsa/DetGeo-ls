# -*- coding:utf8 -*-
"""PAE-only DetGeo with PAE constructed at the original click-fusion RNG slot."""

import torch
import torch.nn as nn

from .DetGeo import MyResnet, CrossViewFusionModule
from .darknet import Darknet, ConvBatchNormReLU
from .hisym_pae_fusion import HiSymPAEFusion


class DetGeoHiSymPAEInline(nn.Module):
    """Original DetGeo construction order, replacing only its click fusion in place."""

    def __init__(self, emb_size=512, leaky=True):
        super().__init__()
        self.query_resnet = MyResnet()
        self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('./saved_models/yolov3.weights')

        use_instnorm = False
        # This exactly occupies DetGeo's former combine_clickptns_conv slot.
        self.pae_fusion = HiSymPAEFusion()
        self.crossview_fusionmodule = CrossViewFusionModule()
        self.query_visudim = 512
        self.reference_visudim = 512
        self.query_mapping_visu = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1,
                                                     leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1,
                                                         leaky=leaky, instance=use_instnorm)
        self.fcn_out = nn.Sequential(
            ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1),
        )

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        position_map = mat_clickptns.unsqueeze(1)
        query_fvisu = self.query_resnet(self.pae_fusion(query_imgs, position_map))
        reference_fvisu = self.reference_darknet(reference_imgs)[1]
        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)
        batch, channels, query_h, query_w = query_fvisu.shape
        query_gvisu = torch.mean(query_fvisu.view(batch, channels, query_h * query_w), dim=2).view(batch, channels)
        fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
        return self.fcn_out(fused_features), attn_score.squeeze(1)
