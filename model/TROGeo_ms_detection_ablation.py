"""Controlled multi-scale detection ablations on the shared Swin-T Direct-CA front end."""

import torch
import torch.nn as nn
import torchvision.models as models

from .TROGeo_ms_direct_ca_sh import SwinTMultiStageEncoder
from .TROGeo_wo_ost import double_conv
from .trogeo_attention import SpatialTransformer


class SwinTThreeStageEncoder(nn.Module):
    """Shared Swin-T encoder exposing stage-2, stage-3, and stage-4 feature maps."""

    def __init__(self):
        super().__init__()
        self.features = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1).features

    def forward(self, x):
        stage2 = stage3 = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 3:
                stage2 = x.permute(0, 3, 1, 2).contiguous()
            elif index == 5:
                stage3 = x.permute(0, 3, 1, 2).contiguous()
        if stage2 is None or stage3 is None:
            raise RuntimeError('Swin-T stage-2/stage-3 features were not produced')
        return stage2, stage3, x.permute(0, 3, 1, 2).contiguous()


class TROGeoMSDetectionAblation(nn.Module):
    """E1--E8 heads; E7/E8 expand independent Direct-CA to three scales."""

    VALID_VARIANTS = ('correct63', 'b_multigrid', 'h2_shared', 'h2_ind', 'h3_ind', 'h3_adaptive',
                      'h2_ind_3scale', 'h2_ind_3scale_stage2cls05')

    def __init__(self, emb_size=768, backbone='swin_t', variant='correct63'):
        super().__init__()
        if emb_size != 768 or backbone != 'swin_t' or variant not in self.VALID_VARIANTS:
            raise ValueError('requires emb_size=768, backbone=swin_t, and a valid MS variant')
        self.variant = variant
        self.three_scale = variant in ('h2_ind_3scale', 'h2_ind_3scale_stage2cls05')
        self.encoder = SwinTThreeStageEncoder() if self.three_scale else SwinTMultiStageEncoder()
        self.position_embedding = double_conv(4, 3)
        if self.three_scale:
            self.cvopm_stage2 = SpatialTransformer(192, 3, 64, depth=1, context_dim=192,
                                                    use_self_attention=False, query_chunk_size=512)
        self.cvopm_stage3 = SpatialTransformer(384, 6, 64, depth=1, context_dim=384,
                                                use_self_attention=False)
        self.cvopm_stage4 = SpatialTransformer(768, 12, 64, depth=1, context_dim=768,
                                                use_self_attention=False)
        self._logged_sanity = False

        if variant == 'correct63':
            self.det_head_stage3 = nn.Conv2d(384, 30, kernel_size=1)
            self.stage4_align = nn.Sequential(
                nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.det_head_stage4 = nn.Conv2d(384, 15, kernel_size=1)
        elif variant == 'b_multigrid':
            self.det_head_stage3 = nn.Conv2d(384, 30, kernel_size=1)
            self.stage4_native_reduce = nn.Sequential(
                nn.Conv2d(768, 384, kernel_size=1), nn.ReLU(inplace=True),
            )
            self.det_head_stage4 = nn.Conv2d(384, 15, kernel_size=1)
        else:
            self.stage4_align = nn.Sequential(
                nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            if variant == 'h2_shared':
                self.det_head_shared = nn.Conv2d(384, 45, kernel_size=1)
            else:  # H2/H3 variants deliberately share the independent-head state layout.
                self.det_head_stage3 = nn.Conv2d(384, 45, kernel_size=1)
                self.det_head_stage4 = nn.Conv2d(384, 45, kernel_size=1)
                if self.three_scale:
                    self.stage2_align = nn.Sequential(
                        nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                    )
                    self.det_head_stage2 = nn.Conv2d(384, 45, kernel_size=1)

    @staticmethod
    def _expect(name, tensor, channels, height, width):
        if tensor.shape[1:] != (channels, height, width):
            raise RuntimeError('expected {} [B,{},{},{}], got {}'.format(
                name, channels, height, width, tuple(tensor.shape)))

    def forward(self, query_imgs, reference_imgs, click_map):
        query_input = self.position_embedding(torch.cat((query_imgs, click_map.unsqueeze(1)), dim=1))
        if self.three_scale:
            q2, q3, q4 = self.encoder(query_input)
            r2, r3, r4 = self.encoder(reference_imgs)
            self._expect('query stage2', q2, 192, 32, 32)
            self._expect('satellite stage2', r2, 192, 128, 128)
            z2 = self.cvopm_stage2(r2, context=q2.flatten(2).transpose(1, 2).contiguous())
        else:
            q3, q4 = self.encoder(query_input)
            r3, r4 = self.encoder(reference_imgs)
        self._expect('query stage3', q3, 384, 16, 16)
        self._expect('query stage4', q4, 768, 8, 8)
        self._expect('satellite stage3', r3, 384, 64, 64)
        self._expect('satellite stage4', r4, 768, 32, 32)
        z3 = self.cvopm_stage3(r3, context=q3.flatten(2).transpose(1, 2).contiguous())
        z4 = self.cvopm_stage4(r4, context=q4.flatten(2).transpose(1, 2).contiguous())

        if self.three_scale:
            p2 = self.det_head_stage2(self.stage2_align(z2))
            p3 = self.det_head_stage3(z3)
            p4 = self.det_head_stage4(self.stage4_align(z4))
            self._expect('E7 p2', p2, 45, 64, 64)
            self._expect('E7 p3', p3, 45, 64, 64)
            self._expect('E7 p4', p4, 45, 64, 64)
            predictions = {'stage2': p2, 'stage3': p3, 'stage4': p4}
        elif self.variant == 'correct63':
            p3 = self.det_head_stage3(z3)  # A3-A8
            p4 = self.det_head_stage4(self.stage4_align(z4))  # A0-A2
            self._expect('E1 p3', p3, 30, 64, 64)
            self._expect('E1 p4', p4, 15, 64, 64)
            predictions = {'joint': torch.cat((p4, p3), dim=1)}  # reversed-anchor order is vital.
        elif self.variant == 'b_multigrid':
            p3 = self.det_head_stage3(z3)
            p4 = self.det_head_stage4(self.stage4_native_reduce(z4))
            self._expect('E2 p3', p3, 30, 64, 64)
            self._expect('E2 p4', p4, 15, 32, 32)
            predictions = {'stage3': p3, 'stage4': p4}
        else:
            aligned4 = self.stage4_align(z4)
            if self.variant == 'h2_shared':
                p3, p4 = self.det_head_shared(z3), self.det_head_shared(aligned4)
            else:
                p3, p4 = self.det_head_stage3(z3), self.det_head_stage4(aligned4)
            self._expect('H p3', p3, 45, 64, 64)
            self._expect('H p4', p4, 45, 64, 64)
            predictions = {'stage3': p3, 'stage4': p4}

        if not self._logged_sanity:
            shapes = {name: tuple(value.shape) for name, value in predictions.items()}
            if self.three_scale:
                print('[TROGeo MS detection sanity] variant={} shared_encoder=True '
                      'self_attention_stage2=False self_attention_stage3=False self_attention_stage4=False '
                      'stage2_query_chunk=512 q2={} q3={} q4={} r2={} r3={} r4={} z2={} z3={} z4={} '
                      'predictions={} feature_fusion=False'.format(
                          self.variant, tuple(q2.shape), tuple(q3.shape), tuple(q4.shape), tuple(r2.shape),
                          tuple(r3.shape), tuple(r4.shape), tuple(z2.shape), tuple(z3.shape), tuple(z4.shape),
                          shapes), flush=True)
            else:
                print('[TROGeo MS detection sanity] variant={} shared_encoder=True self_attention_stage3=False '
                      'self_attention_stage4=False q3={} q4={} r3={} r4={} z3={} z4={} predictions={} '
                      'feature_fusion=False'.format(self.variant, tuple(q3.shape), tuple(q4.shape),
                      tuple(r3.shape), tuple(r4.shape), tuple(z3.shape), tuple(z4.shape), shapes), flush=True)
            self._logged_sanity = True
        return predictions, None
