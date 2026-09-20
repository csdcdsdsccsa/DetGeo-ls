"""Two-scale, separate-head TROGeo Direct-CA detector for DroneAerial.

Stage 3 and stage 4 each use direct satellite-query cross-attention.  They
never fuse features; only their anchor-ordered predictions are concatenated.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .trogeo_attention import SpatialTransformer
from .TROGeo_wo_ost import double_conv


class SwinTMultiStageEncoder(nn.Module):
    """One shared Swin-T forward which exposes its stage-3 and stage-4 maps."""

    def __init__(self):
        super().__init__()
        self.features = models.swin_t(
            weights=models.Swin_T_Weights.IMAGENET1K_V1
        ).features

    def forward(self, x):
        stage3 = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            # torchvision Swin-T features[5] is the 384-channel stage-3 output.
            if index == 5:
                stage3 = x.permute(0, 3, 1, 2).contiguous()
        if stage3 is None:
            raise RuntimeError('Swin-T stage-3 feature was not produced')
        stage4 = x.permute(0, 3, 1, 2).contiguous()
        return stage3, stage4


class SwinSMultiStageEncoder(nn.Module):
    """One shared Swin-S forward exposing the same Stage3/Stage4 interface."""

    def __init__(self):
        super().__init__()
        self.features = models.swin_s(
            weights=models.Swin_S_Weights.IMAGENET1K_V1
        ).features

    def forward(self, x):
        stage3 = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            # torchvision Swin-S shares Swin-T's C=384 Stage3 interface.
            if index == 5:
                stage3 = x.permute(0, 3, 1, 2).contiguous()
        if stage3 is None:
            raise RuntimeError('Swin-S stage-3 feature was not produced')
        stage4 = x.permute(0, 3, 1, 2).contiguous()
        if stage3.shape[1] != 384 or stage4.shape[1] != 768:
            raise RuntimeError('Swin-S requires Stage3/Stage4 channels 384/768, got {}/{}'.format(
                stage3.shape[1], stage4.shape[1]))
        return stage3, stage4


class TROGeoMSDirectCASH(nn.Module):
    """Swin-T stage-3/stage-4 Direct-CA with separate 6/3-anchor heads."""

    def __init__(self, emb_size=768, backbone='swin_t'):
        super().__init__()
        if emb_size != 768:
            raise ValueError('TROGeoMSDirectCASH is fixed to emb_size=768')
        if backbone != 'swin_t':
            raise ValueError('TROGeoMSDirectCASH requires backbone=swin_t')
        self.encoder = SwinTMultiStageEncoder()  # The same instance serves both views.
        self.position_embedding = double_conv(4, 3)
        self.cvopm_stage3 = SpatialTransformer(
            in_channels=384, n_heads=6, d_head=64, depth=1, context_dim=384,
            use_self_attention=False,
        )
        self.cvopm_stage4 = SpatialTransformer(
            in_channels=768, n_heads=12, d_head=64, depth=1, context_dim=768,
            use_self_attention=False,
        )
        # A1-A6, then A7-A9: the unchanged 9-anchor target/loss routes gradients.
        self.det_head_stage3 = nn.Conv2d(384, 6 * 5, kernel_size=1)
        # This is the existing single-scale stage-4 upsampling path, with only
        # its final 9-anchor projection reduced to A7-A9.
        self.det_head_stage4_up = nn.ConvTranspose2d(
            768, 384, kernel_size=4, stride=2, padding=1
        )
        self.det_head_stage4_relu = nn.ReLU(inplace=True)
        self.det_head_stage4_prediction = nn.Conv2d(384, 3 * 5, kernel_size=1)
        self._logged_sanity = False

    @staticmethod
    def _assert_shape(name, tensor, expected):
        if tensor.shape[1:] != expected:
            raise RuntimeError('expected {} [B,{}], got {}'.format(
                name, ','.join(str(value) for value in expected), tuple(tensor.shape)
            ))

    def forward(self, query_imgs, reference_imgs, click_map):
        query_input = self.position_embedding(
            torch.cat((query_imgs, click_map.unsqueeze(1)), dim=1)
        )
        query_stage3, query_stage4 = self.encoder(query_input)
        reference_stage3, reference_stage4 = self.encoder(reference_imgs)
        self._assert_shape('Drone query stage3', query_stage3, (384, 16, 16))
        self._assert_shape('Drone query stage4', query_stage4, (768, 8, 8))
        self._assert_shape('satellite stage3', reference_stage3, (384, 64, 64))
        self._assert_shape('satellite stage4', reference_stage4, (768, 32, 32))

        context3 = query_stage3.flatten(2).transpose(1, 2).contiguous()
        context4 = query_stage4.flatten(2).transpose(1, 2).contiguous()
        fused_stage3 = self.cvopm_stage3(reference_stage3, context=context3)
        fused_stage4 = self.cvopm_stage4(reference_stage4, context=context4)
        prediction_stage3 = self.det_head_stage3(fused_stage3)
        stage4_up = self.det_head_stage4_relu(self.det_head_stage4_up(fused_stage4))
        prediction_stage4 = self.det_head_stage4_prediction(stage4_up)
        self._assert_shape('stage3 prediction', prediction_stage3, (30, 64, 64))
        self._assert_shape('stage4 prediction', prediction_stage4, (15, 64, 64))
        outbox = torch.cat((prediction_stage3, prediction_stage4), dim=1)
        self._assert_shape('detector output', outbox, (45, 64, 64))

        if not self._logged_sanity:
            print(
                '[TROGeo MS-Direct-CA-SH sanity] backbone=swin_t shared_encoder=True '
                'self_attention_stage3=False self_attention_stage4=False '
                'Stage3:Q=satellite KV=query Fq3={} Fr3={} Z3={} P3={} anchors=A1-A6; '
                'Stage4:Q=satellite KV=query Fq4={} Fr4={} Z4={} Stage4Up={} P4={} anchors=A7-A9; '
                'feature_fusion=False prediction_concat=True outbox={} OST=False'.format(
                    tuple(query_stage3.shape), tuple(reference_stage3.shape), tuple(fused_stage3.shape),
                    tuple(prediction_stage3.shape), tuple(query_stage4.shape), tuple(reference_stage4.shape),
                    tuple(fused_stage4.shape), tuple(stage4_up.shape), tuple(prediction_stage4.shape),
                    tuple(outbox.shape),
                ), flush=True,
            )
            self._logged_sanity = True
        return outbox, None
