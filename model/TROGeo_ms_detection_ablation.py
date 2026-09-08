"""Controlled multi-scale detection ablations on the shared Swin-T Direct-CA front end."""

import torch
import torch.nn as nn

from .TROGeo_ms_direct_ca_sh import SwinTMultiStageEncoder
from .TROGeo_wo_ost import double_conv
from .trogeo_attention import SpatialTransformer


class TROGeoMSDetectionAblation(nn.Module):
    """E1/E2/E3/E4/E5 heads; the encoder and two Direct-CA blocks are fixed."""

    VALID_VARIANTS = ('correct63', 'b_multigrid', 'h2_shared', 'h2_ind', 'h3_ind')

    def __init__(self, emb_size=768, backbone='swin_t', variant='correct63'):
        super().__init__()
        if emb_size != 768 or backbone != 'swin_t' or variant not in self.VALID_VARIANTS:
            raise ValueError('requires emb_size=768, backbone=swin_t, and a valid MS variant')
        self.variant = variant
        self.encoder = SwinTMultiStageEncoder()
        self.position_embedding = double_conv(4, 3)
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
            else:  # h2_ind and inference-only h3_ind deliberately share names/state layout.
                self.det_head_stage3 = nn.Conv2d(384, 45, kernel_size=1)
                self.det_head_stage4 = nn.Conv2d(384, 45, kernel_size=1)

    @staticmethod
    def _expect(name, tensor, channels, height, width):
        if tensor.shape[1:] != (channels, height, width):
            raise RuntimeError('expected {} [B,{},{},{}], got {}'.format(
                name, channels, height, width, tuple(tensor.shape)))

    def forward(self, query_imgs, reference_imgs, click_map):
        query_input = self.position_embedding(torch.cat((query_imgs, click_map.unsqueeze(1)), dim=1))
        q3, q4 = self.encoder(query_input)
        r3, r4 = self.encoder(reference_imgs)
        self._expect('query stage3', q3, 384, 16, 16)
        self._expect('query stage4', q4, 768, 8, 8)
        self._expect('satellite stage3', r3, 384, 64, 64)
        self._expect('satellite stage4', r4, 768, 32, 32)
        z3 = self.cvopm_stage3(r3, context=q3.flatten(2).transpose(1, 2).contiguous())
        z4 = self.cvopm_stage4(r4, context=q4.flatten(2).transpose(1, 2).contiguous())

        if self.variant == 'correct63':
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
            print('[TROGeo MS detection sanity] variant={} shared_encoder=True self_attention_stage3=False '
                  'self_attention_stage4=False q3={} q4={} r3={} r4={} z3={} z4={} predictions={} '
                  'feature_fusion=False'.format(self.variant, tuple(q3.shape), tuple(q4.shape),
                  tuple(r3.shape), tuple(r4.shape), tuple(z3.shape), tuple(z4.shape), shapes), flush=True)
            self._logged_sanity = True
        return predictions, None
