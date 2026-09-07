# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import *
import torchvision.models as models

class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        #print(('x1', x.shape), flush=True)
        x = self.base_model.layer2(x)
        #print(('x2', x.shape), flush=True)
        x = self.base_model.layer3(x)
        #print(('x3', x.shape), flush=True)
        x = self.base_model.layer4(x)
        #print(('x4', x.shape), flush=True)
        return x


class CrossViewFusionModule(nn.Module):
    """Single-scale spatial cross-view attention.

    This replaces only the original DetGeo GAP + cosine-similarity fusion:
      - reference/satellite spatial tokens are Q
      - query-image spatial tokens are K/V
      - output keeps the reference 64x64 spatial layout
      - standard residual + LayerNorm + FFN + residual + LayerNorm

    No positional encoding, no self-attention, and no learnable residual gate
    are introduced in this first single-scale experiment.
    """

    def __init__(self, emb_size=512, num_heads=8, ffn_dim=1024):
        super(CrossViewFusionModule, self).__init__()
        if emb_size % num_heads != 0:
            raise ValueError('emb_size must be divisible by num_heads')

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, emb_size),
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, query_fvisu, reference_fvisu):
        # query_fvisu:     B, D, Hq, Wq
        # reference_fvisu: B, D, Hr, Wr
        B, D, Hq, Wq = query_fvisu.shape
        _, _, Hr, Wr = reference_fvisu.shape

        # Keep all query spatial positions instead of applying GAP.
        # Drone: 8x8 -> 64 tokens; Ground/SVI: 8x16 -> 128 tokens.
        query_tokens = query_fvisu.flatten(2).transpose(1, 2).contiguous()

        # Satellite/reference middle-scale feature: 64x64 -> 4096 tokens.
        reference_tokens = reference_fvisu.flatten(2).transpose(1, 2).contiguous()

        # Q = satellite/reference, K/V = query image.
        # The output length is therefore Hr*Wr and preserves the satellite layout.
        cross_features, attn_weights = self.cross_attn(
            reference_tokens,
            query_tokens,
            query_tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        # Standard Transformer residual block. No gamma gate in this experiment.
        x = self.norm1(reference_tokens + cross_features)
        x = self.norm2(x + self.ffn(x))

        context = x.transpose(1, 2).contiguous().view(B, D, Hr, Wr)

        # Keep DetGeo's original two-output interface for train.py/test code.
        # Each satellite location receives a diagnostic score equal to its
        # strongest attention weight over query spatial tokens.
        attn = attn_weights.max(dim=-1).values.view(B, Hr, Wr)
        return context, attn


class DetGeo(nn.Module):
    def __init__(self, emb_size=512, leaky=True):
        super(DetGeo, self).__init__()
        ## Visual model
        self.query_resnet = MyResnet()
        
        self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('./saved_models/yolov3.weights')
        
        use_instnorm=False

        self.combine_clickptns_conv = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        
        self.query_visudim = 512 
        self.reference_visudim = 512

        self.query_mapping_visu = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)

        ## output head -- unchanged original DetGeo head
        self.fcn_out = torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
                nn.Conv2d(emb_size//2, 9*5, kernel_size=1))

        # Construct the new module after all original trainable layers so the
        # shared original DetGeo layers keep their original initialization order.
        # There is no RNG isolation/reset for the new module itself.
        self.crossview_fusionmodule = CrossViewFusionModule(
            emb_size=emb_size,
            num_heads=8,
            ffn_dim=1024,
        )

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        # Original DetGeo square click-point encoding is kept unchanged.
        mat_clickptns = mat_clickptns.unsqueeze(1)
        
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1))
        query_fvisu = self.query_resnet(query_imgs)
        
        reference_raw_fvisu = self.reference_darknet(reference_imgs)
        reference_fvisu = reference_raw_fvisu[1]

        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)

        # Only this part differs from original DetGeo:
        #   original: Query GAP -> global vector -> dot-product/min-max with Satellite
        #   current:  Query Flatten -> K/V; Satellite Flatten -> Q -> MHCA
        #             -> Residual + FFN -> restore Bx512x64x64
        fused_features, attn_score = self.crossview_fusionmodule(
            query_fvisu,
            reference_fvisu,
        )

        outbox = self.fcn_out(fused_features)

        return outbox, attn_score
