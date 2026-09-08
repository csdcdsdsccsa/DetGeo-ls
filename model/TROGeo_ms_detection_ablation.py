"""Controlled multi-scale detection ablations on the shared Swin-T Direct-CA front end."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .TROGeo_ms_direct_ca_sh import SwinTMultiStageEncoder
from .TROGeo_wo_ost import double_conv
from .trogeo_attention import SpatialTransformer


def _make_pe_mlp(out_dim):
    return nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, out_dim))


def _make_le_block():
    return nn.Sequential(
        nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Sigmoid(),
    )


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

    QUERY_PE_VARIANTS = (
        'h2_ind_3scale_pe_ln_amp',
        'h2_ind_3scale_pe_all_add',
        'h2_ind_3scale_pe_all_key',
        'h2_ind_3scale_le_ind_res',
        'h2_ind_3scale_le_stage2_res',
    )
    THREE_SCALE_VARIANTS = ('h2_ind_3scale', 'h2_ind_3scale_stage2cls05') + QUERY_PE_VARIANTS
    VALID_VARIANTS = ('correct63', 'b_multigrid', 'h2_shared', 'h2_ind', 'h3_ind', 'h3_adaptive') + \
                     THREE_SCALE_VARIANTS

    def __init__(self, emb_size=768, backbone='swin_t', variant='correct63'):
        super().__init__()
        if emb_size != 768 or backbone != 'swin_t' or variant not in self.VALID_VARIANTS:
            raise ValueError('requires emb_size=768, backbone=swin_t, and a valid MS variant')
        self.variant = variant
        self.three_scale = variant in self.THREE_SCALE_VARIANTS
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

        # Construct optional PE/LE modules after all E7 base modules, then
        # restore RNG so their initialization cannot perturb E7's trajectory.
        _pos_rng_state = torch.get_rng_state()
        try:
            if self.variant in self.QUERY_PE_VARIANTS[:3]:
                self.pe_proj2 = _make_pe_mlp(192)
                self.pe_proj3 = _make_pe_mlp(384)
                self.pe_proj4 = _make_pe_mlp(768)
                self.pe_alpha2 = nn.Parameter(torch.tensor(0.1))
                self.pe_alpha3 = nn.Parameter(torch.tensor(0.1))
                self.pe_alpha4 = nn.Parameter(torch.tensor(0.1))
                if self.variant == 'h2_ind_3scale_pe_ln_amp':
                    self.pe_norm2 = nn.LayerNorm(192)
                    self.pe_norm3 = nn.LayerNorm(384)
                    self.pe_norm4 = nn.LayerNorm(768)
            elif self.variant == 'h2_ind_3scale_le_ind_res':
                self.le_stage2 = _make_le_block()
                self.le_stage3 = _make_le_block()
                self.le_stage4 = _make_le_block()
                self.le_beta2_logit = nn.Parameter(torch.tensor(-2.1972246))
                self.le_beta3_logit = nn.Parameter(torch.tensor(-2.1972246))
                self.le_beta4_logit = nn.Parameter(torch.tensor(-2.1972246))
            elif self.variant == 'h2_ind_3scale_le_stage2_res':
                self.le_stage2 = _make_le_block()
                self.le_beta2_logit = nn.Parameter(torch.tensor(-2.1972246))
                self.le_beta3_logit = nn.Parameter(torch.tensor(-2.1972246))
                self.le_beta4_logit = nn.Parameter(torch.tensor(-2.1972246))
        finally:
            torch.set_rng_state(_pos_rng_state)

    @staticmethod
    def _expect(name, tensor, channels, height, width):
        if tensor.shape[1:] != (channels, height, width):
            raise RuntimeError('expected {} [B,{},{},{}], got {}'.format(
                name, channels, height, width, tuple(tensor.shape)))

    @staticmethod
    def _to_tokens(feature):
        return feature.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _position_pyramid(click_map, q2, q3, q4):
        position = click_map.unsqueeze(1).to(device=q2.device, dtype=q2.dtype)
        resize = lambda target: F.interpolate(position, size=target.shape[-2:], mode='bilinear', align_corners=False)
        return resize(q2), resize(q3), resize(q4)

    def _query_contexts(self, click_map, q2, q3, q4):
        """Return standard contexts plus optional separate K/V contexts for Query-PE variants."""
        q2_tokens, q3_tokens, q4_tokens = self._to_tokens(q2), self._to_tokens(q3), self._to_tokens(q4)
        contexts = [q2_tokens, q3_tokens, q4_tokens]
        key_contexts = value_contexts = [None, None, None]
        if self.variant in self.QUERY_PE_VARIANTS[:3]:
            p2, p3, p4 = self._position_pyramid(click_map, q2, q3, q4)
            p_tokens = [self._to_tokens(p2), self._to_tokens(p3), self._to_tokens(p4)]
            pe = [self.pe_proj2(p_tokens[0]), self.pe_proj3(p_tokens[1]), self.pe_proj4(p_tokens[2])]
            if self.variant == 'h2_ind_3scale_pe_ln_amp':
                pe = [self.pe_norm2(pe[0]) * p_tokens[0], self.pe_norm3(pe[1]) * p_tokens[1],
                      self.pe_norm4(pe[2]) * p_tokens[2]]
            additive = [self.pe_alpha2 * pe[0], self.pe_alpha3 * pe[1], self.pe_alpha4 * pe[2]]
            if self.variant == 'h2_ind_3scale_pe_all_key':
                key_contexts = [q2_tokens + additive[0], q3_tokens + additive[1], q4_tokens + additive[2]]
                value_contexts = [q2_tokens, q3_tokens, q4_tokens]
            else:
                contexts = [q2_tokens + additive[0], q3_tokens + additive[1], q4_tokens + additive[2]]
        elif self.variant in ('h2_ind_3scale_le_ind_res', 'h2_ind_3scale_le_stage2_res'):
            p2, p3, p4 = self._position_pyramid(click_map, q2, q3, q4)
            if self.variant == 'h2_ind_3scale_le_ind_res':
                gates = [
                    self.le_stage2(torch.cat((p2, q2.mean(dim=1, keepdim=True)), dim=1)),
                    self.le_stage3(torch.cat((p3, q3.mean(dim=1, keepdim=True)), dim=1)),
                    self.le_stage4(torch.cat((p4, q4.mean(dim=1, keepdim=True)), dim=1)),
                ]
            else:
                gate2 = self.le_stage2(torch.cat((p2, q2.mean(dim=1, keepdim=True)), dim=1))
                gates = [gate2, F.interpolate(gate2, size=q3.shape[-2:], mode='bilinear', align_corners=False),
                         F.interpolate(gate2, size=q4.shape[-2:], mode='bilinear', align_corners=False)]
            betas = [torch.sigmoid(self.le_beta2_logit), torch.sigmoid(self.le_beta3_logit),
                     torch.sigmoid(self.le_beta4_logit)]
            contexts = [self._to_tokens(q2 + betas[0] * gates[0] * q2),
                        self._to_tokens(q3 + betas[1] * gates[1] * q3),
                        self._to_tokens(q4 + betas[2] * gates[2] * q4)]
        return contexts, key_contexts, value_contexts

    def _position_mode(self):
        return {
            'h2_ind_3scale_pe_ln_amp': 'PE=P*LN(MLP(P)); add_to_KV=True',
            'h2_ind_3scale_pe_all_add': 'PE=MLP(P); add_to_KV=True',
            'h2_ind_3scale_pe_all_key': 'PE=MLP(P); add_to_K_only=True',
            'h2_ind_3scale_le_ind_res': 'Independent-LE residual gating',
            'h2_ind_3scale_le_stage2_res': 'Stage2-LE shared residual gating',
        }.get(self.variant, 'none')

    def forward(self, query_imgs, reference_imgs, click_map):
        query_input = self.position_embedding(torch.cat((query_imgs, click_map.unsqueeze(1)), dim=1))
        if self.three_scale:
            q2, q3, q4 = self.encoder(query_input)
            r2, r3, r4 = self.encoder(reference_imgs)
            self._expect('query stage2', q2, 192, 32, 32)
            self._expect('satellite stage2', r2, 192, 128, 128)
            contexts, key_contexts, value_contexts = self._query_contexts(click_map, q2, q3, q4)
            z2 = self.cvopm_stage2(r2, context=contexts[0], context_key=key_contexts[0],
                                   context_value=value_contexts[0])
        else:
            q3, q4 = self.encoder(query_input)
            r3, r4 = self.encoder(reference_imgs)
        self._expect('query stage3', q3, 384, 16, 16)
        self._expect('query stage4', q4, 768, 8, 8)
        self._expect('satellite stage3', r3, 384, 64, 64)
        self._expect('satellite stage4', r4, 768, 32, 32)
        if self.three_scale:
            z3 = self.cvopm_stage3(r3, context=contexts[1], context_key=key_contexts[1],
                                   context_value=value_contexts[1])
            z4 = self.cvopm_stage4(r4, context=contexts[2], context_key=key_contexts[2],
                                   context_value=value_contexts[2])
        else:
            z3 = self.cvopm_stage3(r3, context=self._to_tokens(q3))
            z4 = self.cvopm_stage4(r4, context=self._to_tokens(q4))

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
                      'predictions={} feature_fusion=False position_mode={}'.format(
                          self.variant, tuple(q2.shape), tuple(q3.shape), tuple(q4.shape), tuple(r2.shape),
                          tuple(r3.shape), tuple(r4.shape), tuple(z2.shape), tuple(z3.shape), tuple(z4.shape),
                          shapes, self._position_mode()), flush=True)
            else:
                print('[TROGeo MS detection sanity] variant={} shared_encoder=True self_attention_stage3=False '
                      'self_attention_stage4=False q3={} q4={} r3={} r4={} z3={} z4={} predictions={} '
                      'feature_fusion=False'.format(self.variant, tuple(q3.shape), tuple(q4.shape),
                      tuple(r3.shape), tuple(r4.shape), tuple(z3.shape), tuple(z4.shape), shapes), flush=True)
            self._logged_sanity = True
        return predictions, None
