"""Controlled multi-scale detection ablations on the shared Swin-T Direct-CA front end."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .TROGeo_ms_direct_ca_sh import SwinTMultiStageEncoder
from .TROGeo_wo_ost import double_conv
from .detgeo_position_embedding import DetGeoPositionEmbedding
from .trogeo_attention import CrossAttention, SpatialTransformer
from .vit_multistage import TiledViTMultiStageEncoder
from .habr_former import HABRFormer


def build_directional_geometry(distance_map):
    """Recover click-relative ``[D, X, Y]`` geometry from a DetGeo map."""
    if distance_map.dim() != 3:
        raise RuntimeError('distance_map must be [B,H,W], got {}'.format(tuple(distance_map.shape)))
    batch, height, width = distance_map.shape
    with torch.no_grad():
        flat_index = distance_map.reshape(batch, -1).argmax(dim=1)
        click_y = torch.div(flat_index, width, rounding_mode='floor').to(distance_map.dtype)
        click_x = (flat_index % width).to(distance_map.dtype)
        rows = torch.arange(height, device=distance_map.device, dtype=distance_map.dtype).view(1, height, 1)
        cols = torch.arange(width, device=distance_map.device, dtype=distance_map.dtype).view(1, 1, width)
        x_map = ((cols - click_x.view(batch, 1, 1)) / float(width)).expand(batch, height, width)
        y_map = ((rows - click_y.view(batch, 1, 1)) / float(height)).expand(batch, height, width)
    return torch.stack((distance_map, x_map, y_map), dim=1)


class AdaptiveMultiRangePositionField(nn.Module):
    """Refine a DetGeo distance field with fixed or query-adaptive ranges.

    This module is deliberately limited to the fourth input channel of the
    original DetGeo position encoder.  It never replaces the click map used by
    downstream geometry-aware code.
    """

    def __init__(self, mode='adaptive'):
        super().__init__()
        if mode not in ('fixed', 'adaptive'):
            raise ValueError('AMR-PE mode must be fixed or adaptive')
        self.mode = mode
        # Both modes instantiate exactly the same parameter layout.  Fixed mode
        # simply does not use the predictor in forward, making fixed/adaptive a
        # clean comparison under ordinary standard RNG.
        self.content_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2, bias=False),
            nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.weight_predictor = nn.Sequential(
            nn.Flatten(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 3),
        )
        nn.init.zeros_(self.weight_predictor[-1].weight)
        nn.init.zeros_(self.weight_predictor[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, query_imgs, distance_map):
        if distance_map.dim() != 3:
            raise RuntimeError('AMR-PE distance_map must be [B,H,W], got {}'.format(
                tuple(distance_map.shape)))
        d = distance_map.clamp(min=0.0, max=1.0)
        fields = torch.stack((d.square(), d, torch.sqrt(d.clamp_min(1e-6))), dim=1)
        batch_size = d.shape[0]
        if self.mode == 'fixed':
            weights = d.new_full((batch_size, 3), 1.0 / 3.0)
        else:
            weights = torch.softmax(self.weight_predictor(self.content_encoder(query_imgs)), dim=1)
        d_multi = (fields * weights[:, :, None, None]).sum(dim=1)
        alpha = torch.tanh(self.gamma)
        return d + alpha * (d_multi - d), weights, d_multi, alpha


class InputDirectionResidual(nn.Module):
    """Direction-aware residual on top of the unchanged DetGeo input PE."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, base_feature, geometry):
        delta = self.encoder(geometry)
        return base_feature + torch.tanh(self.gamma) * delta, delta


class MultiScaleDirectionResidual(nn.Module):
    """Scale-specific directional residuals before the unchanged Direct-CA."""

    def __init__(self):
        super().__init__()
        self.stage3_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(), nn.Conv2d(32, 384, kernel_size=1, bias=True),
        )
        self.stage4_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(), nn.Conv2d(32, 768, kernel_size=1, bias=True),
        )
        self.beta3 = nn.Parameter(torch.tensor(0.0))
        self.beta4 = nn.Parameter(torch.tensor(0.0))

    def forward(self, q3, q4, geometry):
        g3 = F.interpolate(geometry, size=q3.shape[-2:], mode='bilinear', align_corners=False)
        g4 = F.interpolate(geometry, size=q4.shape[-2:], mode='bilinear', align_corners=False)
        p3, p4 = self.stage3_encoder(g3), self.stage4_encoder(g4)
        return q3 + torch.tanh(self.beta3) * p3, q4 + torch.tanh(self.beta4) * p4, p3, p4


def _make_pe_mlp(out_dim):
    return nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, out_dim))


class PositionQueryRefinement(nn.Module):
    """Position-conditioned residual Query refinement before Direct-CA."""

    def __init__(self, dim, heads, dim_head=64):
        super().__init__()
        self.attn = CrossAttention(query_dim=dim, context_dim=dim, heads=heads, dim_head=dim_head)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, query_tokens, position_tokens):
        if query_tokens.shape != position_tokens.shape:
            raise RuntimeError('PQRA requires equal Query/Position shapes, got {} and {}'.format(
                tuple(query_tokens.shape), tuple(position_tokens.shape)))
        delta = self.attn(query_tokens, context=query_tokens,
                          context_key=query_tokens + position_tokens,
                          context_value=query_tokens)
        return query_tokens + torch.tanh(self.alpha) * delta, delta


def _make_le_block():
    return nn.Sequential(
        nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Sigmoid(),
    )


class CrossScaleFeatureInteraction(nn.Module):
    """Bidirectional, gated residual interaction after E4's Direct-CA maps.

    ``alpha3`` and ``alpha4`` start at zero, so this module is exactly an
    identity mapping at initialization.  The two residual updates are computed
    from the original (not sequentially updated) maps.
    """

    def __init__(self):
        super().__init__()
        self.proj_4to3 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1, bias=False),
            nn.GroupNorm(32, 384),
            nn.GELU(),
        )
        self.gate3 = nn.Sequential(
            nn.Conv2d(768, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.proj_3to4 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 768),
            nn.GELU(),
        )
        self.gate4 = nn.Sequential(
            nn.Conv2d(1536, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.alpha3 = nn.Parameter(torch.tensor(0.0))
        self.alpha4 = nn.Parameter(torch.tensor(0.0))

    def forward(self, z3, z4):
        z4_to3, z3_to4, _, _, gate3, gate4 = self.components(z3, z4)
        z3_out = z3 + torch.tanh(self.alpha3) * gate3 * z4_to3
        z4_out = z4 + torch.tanh(self.alpha4) * gate4 * z3_to4
        return z3_out, z4_out, gate3, gate4

    def components(self, z3, z4):
        """Return the original CSFI maps and spatial gates without updating features."""
        z4_to3 = F.interpolate(self.proj_4to3(z4), size=z3.shape[-2:],
                                mode='bilinear', align_corners=False)
        z3_to4 = self.proj_3to4(z3)
        fused3 = torch.cat((z3, z4_to3), dim=1)
        fused4 = torch.cat((z4, z3_to4), dim=1)
        return z4_to3, z3_to4, fused3, fused4, self.gate3(fused3), self.gate4(fused4)


class MultiHeadSpatialRelationMask(nn.Module):
    """Three-receptive-field spatial mask for a paired cross-scale feature."""

    def __init__(self, in_channels, relation_dim=64, adaptive=False):
        super().__init__()
        self.adaptive = adaptive
        self.relation_encoder = nn.Sequential(
            nn.Conv2d(in_channels, relation_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, relation_dim),
            nn.GELU(),
        )
        self.head1 = self._make_head(relation_dim, 1)
        self.head3 = self._make_head(relation_dim, 3)
        self.head5 = self._make_head(relation_dim, 5)
        if adaptive:
            hidden_dim = max(relation_dim // 2, 16)
            self.scale_selector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(relation_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
            nn.init.zeros_(self.scale_selector[-1].weight)
            nn.init.zeros_(self.scale_selector[-1].bias)

    @staticmethod
    def _make_head(channels, kernel_size):
        # A valid Conv followed by the same-size transposed Conv restores HxW exactly.
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=0, bias=False),
            nn.GELU(),
            nn.ConvTranspose2d(channels, 1, kernel_size=kernel_size, padding=0, bias=True),
        )

    def forward(self, pair_feature):
        relation = self.relation_encoder(pair_feature)
        heads = torch.stack((self.head1(relation), self.head3(relation), self.head5(relation)), dim=1)
        batch_size = relation.shape[0]
        if self.adaptive:
            weights = torch.softmax(self.scale_selector(relation), dim=1)
        else:
            weights = relation.new_full((batch_size, 3), 1.0 / 3.0)
        # 3 * weighted mean gives the equal-sum formulation at uniform weights.
        logits = torch.sum(heads * (3.0 * weights.view(batch_size, 3, 1, 1, 1)), dim=1)
        return torch.sigmoid(logits), weights


class MultiHeadCrossScaleFeatureInteraction(CrossScaleFeatureInteraction):
    """CSFI with MHSAM-inspired 1/3/5 relation masks inside each residual."""

    def __init__(self, adaptive=False):
        super().__init__()
        self.adaptive = adaptive
        self.relation_mask3 = MultiHeadSpatialRelationMask(768, relation_dim=64, adaptive=adaptive)
        self.relation_mask4 = MultiHeadSpatialRelationMask(1536, relation_dim=64, adaptive=adaptive)

    def forward(self, z3, z4):
        z4_to3, z3_to4, pair3, pair4, gate3, gate4 = self.components(z3, z4)
        mask3, weights3 = self.relation_mask3(pair3)
        mask4, weights4 = self.relation_mask4(pair4)
        z3_out = z3 + torch.tanh(self.alpha3) * gate3 * mask3 * z4_to3
        z4_out = z4 + torch.tanh(self.alpha4) * gate4 * mask4 * z3_to4
        return z3_out, z4_out, {'gate3': gate3, 'gate4': gate4, 'mask3': mask3, 'mask4': mask4,
                                'weights3': weights3, 'weights4': weights4}


class AdaptiveMultiReceptiveMask(nn.Module):
    """Sample-adaptive 1/3/5-receptive-field residual gate refinement."""

    def __init__(self, in_channels, relation_dim=64):
        super().__init__()
        self.relation_encoder = nn.Sequential(
            nn.Conv2d(in_channels, relation_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, relation_dim), nn.GELU())
        self.head1 = MultiHeadSpatialRelationMask._make_head(relation_dim, 1)
        self.head3 = MultiHeadSpatialRelationMask._make_head(relation_dim, 3)
        self.head5 = MultiHeadSpatialRelationMask._make_head(relation_dim, 5)
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(relation_dim, 32), nn.GELU(), nn.Linear(32, 3))
        nn.init.zeros_(self.scale_selector[-1].weight)
        nn.init.zeros_(self.scale_selector[-1].bias)

    def forward(self, pair_feature):
        relation = self.relation_encoder(pair_feature)
        heads = torch.stack((self.head1(relation), self.head3(relation), self.head5(relation)), dim=1)
        weights = torch.softmax(self.scale_selector(relation), dim=1)
        logits = (heads * (3.0 * weights[:, :, None, None, None])).sum(dim=1)
        return torch.sigmoid(logits), weights


class ResidualAdaptiveMultiReceptiveCSFI(nn.Module):
    """Refine original CSFI gates without duplicating original CSFI parameters."""

    def __init__(self):
        super().__init__()
        self.mask3 = AdaptiveMultiReceptiveMask(768, relation_dim=64)
        self.mask4 = AdaptiveMultiReceptiveMask(1536, relation_dim=64)
        self.refine_beta3 = nn.Parameter(torch.tensor(0.0))
        self.refine_beta4 = nn.Parameter(torch.tensor(0.0))

    def forward(self, z3, z4, csfi):
        z4_to3, z3_to4, pair3, pair4, gate3, gate4 = csfi.components(z3, z4)
        mask3, weights3 = self.mask3(pair3)
        mask4, weights4 = self.mask4(pair4)
        modulation3 = 1.0 + torch.tanh(self.refine_beta3) * (2.0 * mask3 - 1.0)
        modulation4 = 1.0 + torch.tanh(self.refine_beta4) * (2.0 * mask4 - 1.0)
        z3_out = z3 + torch.tanh(csfi.alpha3) * gate3 * modulation3 * z4_to3
        z4_out = z4 + torch.tanh(csfi.alpha4) * gate4 * modulation4 * z3_to4
        return z3_out, z4_out, {'gate3': gate3, 'gate4': gate4, 'mask3': mask3, 'mask4': mask4,
                                'modulation3': modulation3, 'modulation4': modulation4,
                                'weights3': weights3, 'weights4': weights4}


class AdaptiveStage34Fusion(nn.Module):
    """Sample-adaptive fusion of AMHCSFI-refined Stage3 and aligned Stage4."""

    def __init__(self, channels=384, hidden_dim=128):
        super().__init__()
        self.weight_predictor = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.weight_predictor[-1].weight)
        nn.init.zeros_(self.weight_predictor[-1].bias)

    def forward(self, z3, z4_aligned):
        if z3.shape != z4_aligned.shape:
            raise RuntimeError('AdaptiveStage34Fusion requires equal feature shapes, got {} and {}'.format(
                tuple(z3.shape), tuple(z4_aligned.shape)))
        descriptor = torch.cat((z3.mean(dim=(2, 3)), z4_aligned.mean(dim=(2, 3))), dim=1)
        weights = torch.softmax(self.weight_predictor(descriptor), dim=1)
        fused = weights[:, 0].view(-1, 1, 1, 1) * z3 + weights[:, 1].view(-1, 1, 1, 1) * z4_aligned
        return fused, weights


class QCCHeadRanker(nn.Module):
    """Detached prediction-state ranker used by Bi-Res QCC variants."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, stats):
        if stats.ndim != 2 or stats.shape[1] != 9:
            raise RuntimeError('QCC rank stats must be [B,9], got {}'.format(tuple(stats.shape)))
        return self.mlp(stats).squeeze(1)


class AdaptiveCSFIReliability(nn.Module):
    """Optional channel and sample-adaptive directional reliability for CSFI."""

    def __init__(self, use_channel_gate=False, use_direction_weight=False):
        super().__init__()
        self.use_channel_gate = use_channel_gate
        self.use_direction_weight = use_direction_weight
        if use_channel_gate:
            self.channel3 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(768, 96, 1), nn.GELU(),
                                          nn.Conv2d(96, 384, 1))
            self.channel4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(1536, 192, 1), nn.GELU(),
                                          nn.Conv2d(192, 768, 1))
            nn.init.zeros_(self.channel3[-1].weight)
            nn.init.zeros_(self.channel3[-1].bias)
            nn.init.zeros_(self.channel4[-1].weight)
            nn.init.zeros_(self.channel4[-1].bias)
        if use_direction_weight:
            self.dir_proj3 = nn.Sequential(nn.Linear(384, 128), nn.GELU())
            self.dir_proj4 = nn.Sequential(nn.Linear(768, 128), nn.GELU())
            self.direction_predictor = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 2))
            nn.init.zeros_(self.direction_predictor[-1].weight)
            nn.init.zeros_(self.direction_predictor[-1].bias)

    def forward(self, z3, z4, base_csfi):
        z4_to3, z3_to4, fused3, fused4, spatial3, spatial4 = base_csfi.components(z3, z4)
        if self.use_channel_gate:
            channel3 = 2.0 * torch.sigmoid(self.channel3(fused3))
            channel4 = 2.0 * torch.sigmoid(self.channel4(fused4))
        else:
            channel3 = channel4 = 1.0
        batch = z3.shape[0]
        if self.use_direction_weight:
            v3 = self.dir_proj3(z3.mean(dim=(2, 3)))
            v4 = self.dir_proj4(z4.mean(dim=(2, 3)))
            direction = torch.softmax(self.direction_predictor(torch.cat((v3, v4), dim=1)), dim=1)
            lambda43, lambda34 = direction[:, 0].view(batch, 1, 1, 1), direction[:, 1].view(batch, 1, 1, 1)
        else:
            lambda43 = z3.new_full((batch, 1, 1, 1), 0.5)
            lambda34 = z3.new_full((batch, 1, 1, 1), 0.5)
        gate3, gate4 = spatial3 * channel3, spatial4 * channel4
        z3_out = z3 + 2.0 * lambda43 * torch.tanh(base_csfi.alpha3) * gate3 * z4_to3
        z4_out = z4 + 2.0 * lambda34 * torch.tanh(base_csfi.alpha4) * gate4 * z3_to4
        diagnostics = {
            'spatial3_mean': spatial3.mean(), 'spatial4_mean': spatial4.mean(),
            'channel3_mean': channel3.mean() if torch.is_tensor(channel3) else z3.new_tensor(1.0),
            'channel4_mean': channel4.mean() if torch.is_tensor(channel4) else z3.new_tensor(1.0),
            'lambda43': lambda43.mean(), 'lambda34': lambda34.mean(),
        }
        return z3_out, z4_out, diagnostics


class CoarseGuidance(nn.Module):
    """Use Stage4's cross-view response as a residual Stage3 search prior."""

    def __init__(self):
        super().__init__()
        self.coarse_head = nn.Conv2d(768, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, z4, r3):
        coarse_logits = self.coarse_head(z4)
        coarse_up = F.interpolate(torch.sigmoid(coarse_logits), size=r3.shape[-2:],
                                  mode='bilinear', align_corners=False)
        r3_guided = r3 * (1.0 + torch.tanh(self.gamma) * coarse_up)
        return r3_guided, coarse_logits, coarse_up


class FineGuidance(nn.Module):
    """Use Stage3's cross-view response as a residual Stage4 search prior."""

    def __init__(self):
        super().__init__()
        self.fine_head = nn.Conv2d(384, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, z3, r4):
        fine_logits = self.fine_head(z3)
        fine_down = F.interpolate(torch.sigmoid(fine_logits), size=r4.shape[-2:],
                                  mode='bilinear', align_corners=False)
        r4_guided = r4 * (1.0 + torch.tanh(self.gamma) * fine_down)
        return r4_guided, fine_logits, fine_down


class HeadQualitySelector(nn.Module):
    """Predict the more reliable frozen FG-Res detection head."""

    def __init__(self, feature_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 2),
        )

    def forward(self, feat3, feat4, score3, score4):
        return self.mlp(torch.cat((feat3, feat4, score3[:, None], score4[:, None]), dim=1))


class HeadPairwiseRanker(nn.Module):
    """HQS-v2a: balanced pairwise Head3-vs-Head4 ranker."""

    def __init__(self, feature_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, feat3, feat4, score3, score4):
        x = torch.cat((feat3, feat4, score3[:, None], score4[:, None]), dim=1)
        return self.mlp(x).squeeze(1)


class HeadPredictionQualityRanker(nn.Module):
    """HQS-v2b pairwise ranker with detached prediction-quality statistics."""

    def __init__(self, feature_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 7, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, feat3, feat4, quality_stats):
        if quality_stats.ndim != 2 or quality_stats.shape[1] != 7:
            raise RuntimeError('HQS-v2b quality_stats must be [B,7], got {}'.format(tuple(quality_stats.shape)))
        return self.mlp(torch.cat((feat3, feat4, quality_stats), dim=1)).squeeze(1)


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
    # Strict E4 two-scale Query-position ablations.  These intentionally do
    # not create a Stage2 CA/head or enter the three-head loss path.
    TWO_SCALE_EMBED_VARIANTS = (
        'h2_ind_pe_ln_amp_kv',
        'h2_ind_pe_ln_amp_key',
        'h2_ind_pe_all_add',
        'h2_ind_pe_all_key',
    )
    TWO_SCALE_LN_AMP_VARIANTS = (
        'h2_ind_pe_ln_amp_kv',
        'h2_ind_pe_ln_amp_key',
    )
    TWO_SCALE_KEY_ONLY_VARIANTS = (
        'h2_ind_pe_ln_amp_key',
        'h2_ind_pe_all_key',
    )
    TWO_SCALE_QUERY_PE_VARIANTS = TWO_SCALE_EMBED_VARIANTS + (
        'h2_ind_le_stage2_res',
    )
    # Strict E4 two-scale position-guided cross-attention variants.  Position
    # information changes only the pre-softmax attention logits; Query K/V are
    # left untouched and no Stage2 CA/head is constructed.
    TWO_SCALE_PGCA_VARIANTS = (
        'h2_ind_pgca_c_direct',
        'h2_ind_pgca_a_conv',
        'h2_ind_pgca_b_dynamic',
    )
    QUERY_REFINE_VARIANTS = ('h2_ind_pqra',)
    # Strict E4 two-scale post-matching interaction.  This must remain
    # separate from all PE/LE and PGCA variants so that h2_ind is untouched.
    TWO_SCALE_COLLAB_VARIANTS = (
        'h2_ind_csfi', 'h2_ind_cg', 'h2_ind_csfi_cg', 'h2_ind_hier',
        'h2_ind_csfi_fg', 'h2_ind_csfi_bi')
    ADAPTIVE_CSFI_VARIANTS = ('h2_ind_csfi_cg_channel', 'h2_ind_csfi_cg_dir', 'h2_ind_csfi_cg_ar')
    MHCSFI_VARIANTS = ('h2_ind_mhcsfi_bi', 'h2_ind_amhcsfi_bi')
    QCC_A_VARIANTS = ('h2_ind_amhcsfi_res_bi_qcc_a',)
    QCC_RANK_VARIANTS = ('h2_ind_amhcsfi_res_bi_qcc_b', 'h2_ind_amhcsfi_res_bi_qcc_full')
    QCC_FULL_VARIANTS = ('h2_ind_amhcsfi_res_bi_qcc_full',)
    QCC_VARIANTS = QCC_A_VARIANTS + QCC_RANK_VARIANTS
    AMHCSFI_RES_BI_VARIANTS = ('h2_ind_amhcsfi_res_bi',) + QCC_VARIANTS
    # Bi-Res keeps its bidirectional guidance and auxiliary heatmap losses;
    # this sibling changes only the final detector from two heads to an
    # adaptive Stage-3/4 fused single head.
    AMHCSFI_RES_BI_SINGLE_VARIANTS = ('h2_ind_amhcsfi_res_bi_afuse',)
    FG_AMHCSFI_RES_SINGLE_VARIANTS = (
        'h2_ind_fg_amhcsfi_res_s3', 'h2_ind_fg_amhcsfi_res_s4', 'h2_ind_fg_amhcsfi_res_afuse')
    AMHCSFI_RES_GUIDE_VARIANTS = ('h2_ind_cg_amhcsfi_res', 'h2_ind_fg_amhcsfi_res') + \
                                  FG_AMHCSFI_RES_SINGLE_VARIANTS
    AMHCSFI_RES_VARIANTS = (AMHCSFI_RES_BI_VARIANTS + AMHCSFI_RES_BI_SINGLE_VARIANTS +
                            AMHCSFI_RES_GUIDE_VARIANTS)
    ADAPTIVE_CSFI_CHANNEL_VARIANTS = ('h2_ind_csfi_cg_channel', 'h2_ind_csfi_cg_ar')
    ADAPTIVE_CSFI_DIRECTION_VARIANTS = ('h2_ind_csfi_cg_dir', 'h2_ind_csfi_cg_ar')
    NO_CSFI_GUIDE_VARIANTS = ('h2_ind_fg_nocsfi', 'h2_ind_bi_nocsfi')
    DADPE_SUPPORTED_VARIANTS = ('h2_ind_csfi_bi', 'h2_ind_fg_amhcsfi_res', 'h2_ind_amhcsfi_res_bi')
    CG_HABR_PRIOR_VARIANTS = ('h2_ind_cg_habr_prior',)
    TWO_SCALE_COLLAB_VARIANTS = (TWO_SCALE_COLLAB_VARIANTS + ADAPTIVE_CSFI_VARIANTS + MHCSFI_VARIANTS + AMHCSFI_RES_VARIANTS +
                                 NO_CSFI_GUIDE_VARIANTS + CG_HABR_PRIOR_VARIANTS)
    CSFI_VARIANTS = (
        'h2_ind_csfi', 'h2_ind_csfi_cg', 'h2_ind_hier',
        'h2_ind_csfi_fg', 'h2_ind_csfi_bi') + AMHCSFI_RES_VARIANTS
    COARSE_GUIDE_VARIANTS = ('h2_ind_cg', 'h2_ind_csfi_cg', 'h2_ind_hier', 'h2_ind_csfi_bi',
                             'h2_ind_bi_nocsfi', 'h2_ind_cg_amhcsfi_res') + MHCSFI_VARIANTS + AMHCSFI_RES_BI_VARIANTS + AMHCSFI_RES_BI_SINGLE_VARIANTS + CG_HABR_PRIOR_VARIANTS + \
                            ADAPTIVE_CSFI_VARIANTS
    FINE_GUIDE_VARIANTS = ('h2_ind_csfi_fg', 'h2_ind_csfi_bi', 'h2_ind_fg_nocsfi', 'h2_ind_bi_nocsfi',
                           'h2_ind_fg_amhcsfi_res') + FG_AMHCSFI_RES_SINGLE_VARIANTS + \
                          MHCSFI_VARIANTS + AMHCSFI_RES_BI_VARIANTS + AMHCSFI_RES_BI_SINGLE_VARIANTS
    FINE_ONLY_GUIDE_VARIANTS = ('h2_ind_csfi_fg', 'h2_ind_fg_nocsfi', 'h2_ind_fg_amhcsfi_res') + \
                               FG_AMHCSFI_RES_SINGLE_VARIANTS
    BIDIR_GUIDE_VARIANTS = ('h2_ind_csfi_bi', 'h2_ind_bi_nocsfi') + MHCSFI_VARIANTS + AMHCSFI_RES_BI_VARIANTS + AMHCSFI_RES_BI_SINGLE_VARIANTS
    HABR_VARIANTS = ('h2_ind_habr_core', 'h2_ind_habr_prior', 'h2_ind_habr_adapt', 'h2_ind_habr') + \
                    CG_HABR_PRIOR_VARIANTS
    HABR_MODE = {
        'h2_ind_habr_core': 'core', 'h2_ind_habr_prior': 'prior',
        'h2_ind_habr_adapt': 'adapt', 'h2_ind_habr': 'full',
        'h2_ind_cg_habr_prior': 'prior'}
    HIER_VARIANTS = ('h2_ind_hier',)
    THREE_SCALE_VARIANTS = ('h2_ind_3scale', 'h2_ind_3scale_stage2cls05') + QUERY_PE_VARIANTS
    VALID_VARIANTS = ('correct63', 'b_multigrid', 'h2_shared', 'h2_ind', 'h3_ind', 'h3_adaptive') + \
                     THREE_SCALE_VARIANTS + TWO_SCALE_QUERY_PE_VARIANTS + TWO_SCALE_PGCA_VARIANTS + \
                     TWO_SCALE_COLLAB_VARIANTS + HABR_VARIANTS + QUERY_REFINE_VARIANTS

    def __init__(self, emb_size=768, backbone='swin_t', variant='correct63', position_mode='current', dadpe_mode='none',
                 amr_pe_mode='none',
                 enable_hqs=False, enable_hqs_v2a=False, enable_hqs_v2b=False):
        super().__init__()
        if (emb_size != 768 or backbone not in ('swin_t', 'vit_t', 'vit_s') or variant not in self.VALID_VARIANTS
                or position_mode not in ('current', 'detgeo')):
            raise ValueError('requires emb_size=768, a supported backbone, and a valid MS variant')
        if backbone in ('vit_t', 'vit_s') and variant != 'h2_ind':
            raise ValueError('ViT backbones are restricted to the strict two-scale E4 h2_ind experiment')
        if dadpe_mode not in ('none', 'input', 'multiscale'):
            raise ValueError('dadpe_mode must be none/input/multiscale')
        if dadpe_mode != 'none' and (variant not in self.DADPE_SUPPORTED_VARIANTS
                                     or position_mode != 'detgeo' or backbone != 'swin_t'):
            raise ValueError('DADPE requires A3-Bi, FG-Res, or Bi-Res, original DetGeo PE, and Swin-T')
        if amr_pe_mode not in ('none', 'fixed', 'adaptive'):
            raise ValueError('amr_pe_mode must be none/fixed/adaptive')
        if amr_pe_mode != 'none' and (variant != 'h2_ind_amhcsfi_res_bi' or position_mode != 'detgeo'
                                      or backbone != 'swin_t' or dadpe_mode != 'none'):
            raise ValueError('AMR-PE requires Bi-Res, original DetGeo PE, Swin-T, and dadpe_mode=none')
        self.variant = variant
        self.position_mode = position_mode
        self.backbone_name = backbone
        self.dadpe_mode = dadpe_mode
        self.amr_pe_mode = amr_pe_mode
        self.enable_hqs = bool(enable_hqs)
        self.enable_hqs_v2a = bool(enable_hqs_v2a)
        self.enable_hqs_v2b = bool(enable_hqs_v2b)
        if sum((self.enable_hqs, self.enable_hqs_v2a, self.enable_hqs_v2b)) > 1:
            raise ValueError('HQS-v1/v2a/v2b are mutually exclusive')
        if (self.enable_hqs or self.enable_hqs_v2a or self.enable_hqs_v2b) and variant != 'h2_ind_fg_amhcsfi_res':
            raise ValueError('HQS requires h2_ind_fg_amhcsfi_res')
        self.three_scale = variant in self.THREE_SCALE_VARIANTS
        # q2 is exposed only to construct the propagated LE gate; detection
        # remains strictly two-scale for this variant.
        self.need_query_stage2 = variant == 'h2_ind_le_stage2_res'
        if backbone in ('vit_t', 'vit_s'):
            self.encoder = TiledViTMultiStageEncoder(backbone)
        else:
            self.encoder = (SwinTThreeStageEncoder() if (self.three_scale or self.need_query_stage2)
                            else SwinTMultiStageEncoder())
        self.position_embedding = (double_conv(4, 3) if position_mode == 'current'
                                   else DetGeoPositionEmbedding())
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
            elif variant in self.FG_AMHCSFI_RES_SINGLE_VARIANTS + self.AMHCSFI_RES_BI_SINGLE_VARIANTS:
                self.det_head_single = nn.Conv2d(384, 45, kernel_size=1)
                if variant in self.FG_AMHCSFI_RES_SINGLE_VARIANTS:
                    _rng_pad_stage4_head = nn.Conv2d(384, 45, kernel_size=1)
                    del _rng_pad_stage4_head
            else:  # H2/H3 variants deliberately share the independent-head state layout.
                self.det_head_stage3 = nn.Conv2d(384, 45, kernel_size=1)
                self.det_head_stage4 = nn.Conv2d(384, 45, kernel_size=1)
                if self.three_scale:
                    self.stage2_align = nn.Sequential(
                        nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                    )
                    self.det_head_stage2 = nn.Conv2d(384, 45, kernel_size=1)

        # PE/LE modules are initialized on the ordinary RNG trajectory.  This
        # intentionally lets the added module consume RNG, matching the
        # standard training protocol used for this rerun.
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
        elif self.variant in self.TWO_SCALE_EMBED_VARIANTS:
            self.pe_proj3 = _make_pe_mlp(384)
            self.pe_proj4 = _make_pe_mlp(768)
            self.pe_alpha3 = nn.Parameter(torch.tensor(0.1))
            self.pe_alpha4 = nn.Parameter(torch.tensor(0.1))
            if self.variant in self.TWO_SCALE_LN_AMP_VARIANTS:
                self.pe_norm3 = nn.LayerNorm(384)
                self.pe_norm4 = nn.LayerNorm(768)
        elif self.variant == 'h2_ind_le_stage2_res':
            self.le_stage2 = _make_le_block()
            self.le_beta3_logit = nn.Parameter(torch.tensor(-2.1972246))
            self.le_beta4_logit = nn.Parameter(torch.tensor(-2.1972246))
        elif self.variant in self.TWO_SCALE_PGCA_VARIANTS:
            if self.variant != 'h2_ind_pgca_c_direct':
                self.pe_bias3 = nn.Conv2d(3, 1, kernel_size=1)
                self.pe_bias4 = nn.Conv2d(3, 1, kernel_size=1)
            if self.variant == 'h2_ind_pgca_b_dynamic':
                self.lambda_predictor3 = nn.Sequential(
                    nn.Linear(384, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Sigmoid())
                self.lambda_predictor4 = nn.Sequential(
                    nn.Linear(768, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Sigmoid())
            else:
                # A/C use one learned global strength per scale; unlike B it is
                # fixed across samples and starts as a small residual (0.05).
                self.lambda3 = nn.Parameter(torch.tensor(0.05))
                self.lambda4 = nn.Parameter(torch.tensor(0.05))

        if self.variant in self.CSFI_VARIANTS or self.variant in self.ADAPTIVE_CSFI_VARIANTS:
            # AMHCSFI-Res reuses this original module.  Construct it at the
            # exact ordinary-CSFI RNG position, before the matching guidance.
            self.cross_scale_interaction = CrossScaleFeatureInteraction()
        if self.variant in self.MHCSFI_VARIANTS:
            self.mh_cross_scale_interaction = MultiHeadCrossScaleFeatureInteraction(
                adaptive=self.variant == 'h2_ind_amhcsfi_bi')
        # NoCSFI variants are standalone natural-RNG models: they neither
        # construct CSFI nor synthesize its RNG consumption.
        if self.variant in self.COARSE_GUIDE_VARIANTS:
            self.coarse_guidance = CoarseGuidance()
        if self.variant in self.FINE_GUIDE_VARIANTS:
            self.fine_guidance = FineGuidance()
        # Must be after every corresponding ordinary-CSFI public module.
        if self.variant in self.AMHCSFI_RES_VARIANTS:
            self.amhcsfi_res_refiner = ResidualAdaptiveMultiReceptiveCSFI()
        if self.variant in self.QCC_VARIANTS:
            # Every QCC variant has the identical parameter layout.  QCC-A
            # simply does not consume the ranker at train/inference time.
            # Deliberately ordinary --standard_rng: no RNG-state restoration.
            self.qcc_quality_head_stage3 = nn.Conv2d(384, 9, kernel_size=1)
            self.qcc_quality_head_stage4 = nn.Conv2d(384, 9, kernel_size=1)
            self.qcc_ranker = QCCHeadRanker()
            nn.init.zeros_(self.qcc_quality_head_stage3.weight)
            nn.init.zeros_(self.qcc_quality_head_stage3.bias)
            nn.init.zeros_(self.qcc_quality_head_stage4.weight)
            nn.init.zeros_(self.qcc_quality_head_stage4.bias)
        if self.variant == 'h2_ind_fg_amhcsfi_res_afuse':
            rng_state = torch.get_rng_state()
            self.stage34_adaptive_fusion = AdaptiveStage34Fusion(channels=384, hidden_dim=128)
            torch.set_rng_state(rng_state)
        elif self.variant in self.AMHCSFI_RES_BI_SINGLE_VARIANTS:
            # New experiments intentionally consume ordinary --standard_rng;
            # no state restore or synthetic parameter padding is used here.
            self.stage34_adaptive_fusion = AdaptiveStage34Fusion(channels=384, hidden_dim=128)
        if self.variant in self.HABR_VARIANTS:
            self.habr_former = HABRFormer(mode=self.HABR_MODE[self.variant], relation_dim=256,
                                          num_samples=4, num_heads=4, window_size=4, offset_scale=2.0)
        # Intentionally last: shared CSFI and coarse-guidance initialization
        # keep the same ordinary --standard_rng trajectory as A3.
        if self.variant in self.ADAPTIVE_CSFI_VARIANTS:
            self.adaptive_csfi_reliability = AdaptiveCSFIReliability(
                use_channel_gate=self.variant in self.ADAPTIVE_CSFI_CHANNEL_VARIANTS,
                use_direction_weight=self.variant in self.ADAPTIVE_CSFI_DIRECTION_VARIANTS)
        if self.variant in self.HIER_VARIANTS:
            self.hier_conf_alpha = nn.Parameter(torch.tensor(0.0))
        # Created after every A0 shared module so --standard_rng leaves their
        # initialization trajectory intact; PQRA alone consumes the new RNG.
        if self.variant in self.QUERY_REFINE_VARIANTS:
            self.pqra_pos_proj3 = _make_pe_mlp(384)
            self.pqra_pos_proj4 = _make_pe_mlp(768)
            self.query_refine3 = PositionQueryRefinement(dim=384, heads=6, dim_head=64)
            self.query_refine4 = PositionQueryRefinement(dim=768, heads=12, dim_head=64)
        # DADPE is deliberately constructed under ordinary standard RNG.  Its
        # parameters therefore naturally advance the model/DataLoader RNG path.
        if self.dadpe_mode != 'none':
            self.input_direction_residual = InputDirectionResidual()
            if self.dadpe_mode == 'multiscale':
                self.multiscale_direction_residual = MultiScaleDirectionResidual()
        # AMR-PE follows the ordinary --standard_rng protocol.  Constructing it
        # after the Bi-Res public modules keeps their initial tensors identical
        # to the DetGeo-PE baseline while naturally advancing later RNG state.
        if self.amr_pe_mode != 'none':
            self.amr_position_field = AdaptiveMultiRangePositionField(mode=self.amr_pe_mode)
        # v1 follows ordinary standard RNG; v2a pads/restores only to reproduce
        # v1's exact post-construction RNG state for the controlled comparison.
        if self.enable_hqs:
            self.head_quality_selector = HeadQualitySelector(feature_dim=384)
        if self.enable_hqs_v2a:
            pre_hqs_rng = torch.get_rng_state()
            rng_pad_v1 = HeadQualitySelector(feature_dim=384)
            post_v1_rng = torch.get_rng_state()
            del rng_pad_v1
            torch.set_rng_state(pre_hqs_rng)
            self.head_pairwise_ranker = HeadPairwiseRanker(feature_dim=384)
            torch.set_rng_state(post_v1_rng)
        if self.enable_hqs_v2b:
            pre_hqs_rng = torch.get_rng_state()
            rng_pad_v1 = HeadQualitySelector(feature_dim=384)
            post_v1_rng = torch.get_rng_state()
            del rng_pad_v1
            torch.set_rng_state(pre_hqs_rng)
            self.head_prediction_quality_ranker = HeadPredictionQualityRanker(feature_dim=384)
            torch.set_rng_state(post_v1_rng)

    @staticmethod
    def _expect(name, tensor, channels, height, width):
        if tensor.shape[1:] != (channels, height, width):
            raise RuntimeError('expected {} [B,{},{},{}], got {}'.format(
                name, channels, height, width, tuple(tensor.shape)))

    @staticmethod
    def _to_tokens(feature):
        return feature.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _resize_position(click_map, target):
        position = click_map.unsqueeze(1).to(device=target.device, dtype=target.dtype)
        return F.interpolate(position, size=target.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _position_pyramid(click_map, q2, q3, q4):
        return (TROGeoMSDetectionAblation._resize_position(click_map, q2),
                TROGeoMSDetectionAblation._resize_position(click_map, q3),
                TROGeoMSDetectionAblation._resize_position(click_map, q4))

    def _two_scale_pe_contexts(self, click_map, q3, q4):
        """Return E4 two-scale contexts, with optional separate K/V inputs."""
        q3_tokens, q4_tokens = self._to_tokens(q3), self._to_tokens(q4)
        p3_tokens = self._to_tokens(self._resize_position(click_map, q3))
        p4_tokens = self._to_tokens(self._resize_position(click_map, q4))
        pe3, pe4 = self.pe_proj3(p3_tokens), self.pe_proj4(p4_tokens)
        if self.variant in self.TWO_SCALE_LN_AMP_VARIANTS:
            pe3, pe4 = self.pe_norm3(pe3) * p3_tokens, self.pe_norm4(pe4) * p4_tokens
        additive3, additive4 = self.pe_alpha3 * pe3, self.pe_alpha4 * pe4
        if self.variant in self.TWO_SCALE_KEY_ONLY_VARIANTS:
            return (q3_tokens, q4_tokens, q3_tokens + additive3, q4_tokens + additive4,
                    q3_tokens, q4_tokens)
        return (q3_tokens + additive3, q4_tokens + additive4, None, None, None, None)

    def _two_scale_stage2_le_contexts(self, click_map, q2, q3, q4):
        p2 = self._resize_position(click_map, q2)
        gate2 = self.le_stage2(torch.cat((p2, q2.mean(dim=1, keepdim=True)), dim=1))
        gate3 = F.interpolate(gate2, size=q3.shape[-2:], mode='bilinear', align_corners=False)
        gate4 = F.interpolate(gate2, size=q4.shape[-2:], mode='bilinear', align_corners=False)
        beta3, beta4 = torch.sigmoid(self.le_beta3_logit), torch.sigmoid(self.le_beta4_logit)
        return (self._to_tokens(q3 + beta3 * gate3 * q3),
                self._to_tokens(q4 + beta4 * gate4 * q4))

    def _two_scale_position_biases(self, position_feature, q3, q4):
        """Build Stage3/4 logit biases without modifying Query K or V."""
        p3 = F.interpolate(position_feature, size=q3.shape[-2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(position_feature, size=q4.shape[-2:], mode='bilinear', align_corners=False)
        if self.variant == 'h2_ind_pgca_c_direct':
            bias3, bias4 = p3.mean(dim=1, keepdim=True), p4.mean(dim=1, keepdim=True)
        else:
            bias3, bias4 = self.pe_bias3(p3), self.pe_bias4(p4)
        if self.variant == 'h2_ind_pgca_b_dynamic':
            lambda3 = self.lambda_predictor3(q3.mean(dim=(2, 3)))
            lambda4 = self.lambda_predictor4(q4.mean(dim=(2, 3)))
        else:
            lambda3, lambda4 = self.lambda3, self.lambda4
        return bias3.flatten(2), bias4.flatten(2), lambda3, lambda4

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
            'h2_ind_pe_ln_amp_kv': 'PE=P*LN(MLP(P)); add_to_KV=True',
            'h2_ind_pe_ln_amp_key': 'PE=P*LN(MLP(P)); add_to_K_only=True',
            'h2_ind_pe_all_add': 'PE=MLP(P); add_to_KV=True',
            'h2_ind_pe_all_key': 'PE=MLP(P); add_to_K_only=True',
            'h2_ind_le_stage2_res': 'Stage2-LE propagated residual gating',
            'h2_ind_pgca_c_direct': 'PGCA-C direct original-PE logit bias; global lambda',
            'h2_ind_pgca_a_conv': 'PGCA-A learned 1x1-conv logit bias; global lambda',
            'h2_ind_pgca_b_dynamic': 'PGCA-B learned 1x1-conv logit bias; sample-adaptive lambda',
        }.get(self.variant, 'none')

    def forward(self, query_imgs, reference_imgs, click_map):
        amr_weights = amr_d_multi = amr_alpha = amr_identity_error = None
        if self.amr_pe_mode != 'none':
            effective_click_map, amr_weights, amr_d_multi, amr_alpha = self.amr_position_field(
                query_imgs, click_map)
            amr_identity_error = (effective_click_map - click_map).abs().max()
        else:
            effective_click_map = click_map
        position_feature = self.position_embedding(
            torch.cat((query_imgs, effective_click_map.unsqueeze(1)), dim=1))
        geometry = input_dir_delta = dadpe_p3 = dadpe_p4 = None
        if self.dadpe_mode != 'none':
            geometry = build_directional_geometry(click_map)
            query_input, input_dir_delta = self.input_direction_residual(position_feature, geometry)
        else:
            query_input = position_feature
        csfi_gate3 = csfi_gate4 = None
        coarse_logits = coarse_up = None
        fine_logits = fine_down = None
        habr_prior3_logits = habr_prior4_logits = habr_diagnostics = None
        adaptive_csfi_diagnostics = mhcsfi_diagnostics = None
        pqra_delta3 = pqra_delta4 = pqra_position3 = pqra_position4 = None
        if self.three_scale:
            q2, q3, q4 = self.encoder(query_input)
            r2, r3, r4 = self.encoder(reference_imgs)
            self._expect('query stage2', q2, 192, 32, 32)
            self._expect('satellite stage2', r2, 192, 128, 128)
            contexts, key_contexts, value_contexts = self._query_contexts(click_map, q2, q3, q4)
            z2 = self.cvopm_stage2(r2, context=contexts[0], context_key=key_contexts[0],
                                   context_value=value_contexts[0])
        elif self.need_query_stage2:
            q2, q3, q4 = self.encoder(query_input)
            _, r3, r4 = self.encoder(reference_imgs)
            self._expect('LE query stage2', q2, 192, 32, 32)
        else:
            q3, q4 = self.encoder(query_input)
            r3, r4 = self.encoder(reference_imgs)
        self._expect('query stage3', q3, 384, 16, 16)
        self._expect('query stage4', q4, 768, 8, 8)
        self._expect('satellite stage3', r3, 384, 64, 64)
        self._expect('satellite stage4', r4, 768, 32, 32)
        if self.dadpe_mode == 'multiscale':
            q3, q4, dadpe_p3, dadpe_p4 = self.multiscale_direction_residual(q3, q4, geometry)
        if self.variant in self.BIDIR_GUIDE_VARIANTS:
            # Reuse the same CA4 parameters for both passes: first to derive
            # the coarse Stage4 prior, then to refine the final Stage4 map.
            z4_seed = self.cvopm_stage4(r4, context=self._to_tokens(q4))
            r3_guided, coarse_logits, coarse_up = self.coarse_guidance(z4_seed, r3)
            z3 = self.cvopm_stage3(r3_guided, context=self._to_tokens(q3))
            r4_guided, fine_logits, fine_down = self.fine_guidance(z3, r4)
            z4 = self.cvopm_stage4(r4_guided, context=self._to_tokens(q4))
        elif self.variant in self.FINE_ONLY_GUIDE_VARIANTS:
            z3 = self.cvopm_stage3(r3, context=self._to_tokens(q3))
            r4_guided, fine_logits, fine_down = self.fine_guidance(z3, r4)
            z4 = self.cvopm_stage4(r4_guided, context=self._to_tokens(q4))
        elif self.variant in self.COARSE_GUIDE_VARIANTS:
            # Coarse-to-fine order is intentional: Stage4 response guides the
            # Stage3 satellite search before Stage3 Direct-CA is evaluated.
            z4 = self.cvopm_stage4(r4, context=self._to_tokens(q4))
            r3_guided, coarse_logits, coarse_up = self.coarse_guidance(z4, r3)
            z3 = self.cvopm_stage3(r3_guided, context=self._to_tokens(q3))
        elif self.three_scale:
            z3 = self.cvopm_stage3(r3, context=contexts[1], context_key=key_contexts[1],
                                   context_value=value_contexts[1])
            z4 = self.cvopm_stage4(r4, context=contexts[2], context_key=key_contexts[2],
                                   context_value=value_contexts[2])
        elif self.variant in self.TWO_SCALE_EMBED_VARIANTS:
            context3, context4, key3, key4, value3, value4 = self._two_scale_pe_contexts(click_map, q3, q4)
            z3 = self.cvopm_stage3(r3, context=context3, context_key=key3, context_value=value3)
            z4 = self.cvopm_stage4(r4, context=context4, context_key=key4, context_value=value4)
        elif self.need_query_stage2:
            context3, context4 = self._two_scale_stage2_le_contexts(click_map, q2, q3, q4)
            z3 = self.cvopm_stage3(r3, context=context3)
            z4 = self.cvopm_stage4(r4, context=context4)
        elif self.variant in self.TWO_SCALE_PGCA_VARIANTS:
            bias3, bias4, lambda3, lambda4 = self._two_scale_position_biases(position_feature, q3, q4)
            z3 = self.cvopm_stage3(r3, context=self._to_tokens(q3),
                                   position_bias=bias3, position_lambda=lambda3)
            z4 = self.cvopm_stage4(r4, context=self._to_tokens(q4),
                                   position_bias=bias4, position_lambda=lambda4)
        elif self.variant in self.QUERY_REFINE_VARIANTS:
            q3_tokens, q4_tokens = self._to_tokens(q3), self._to_tokens(q4)
            pqra_position3 = self.pqra_pos_proj3(self._to_tokens(self._resize_position(click_map, q3)))
            pqra_position4 = self.pqra_pos_proj4(self._to_tokens(self._resize_position(click_map, q4)))
            q3_refined, pqra_delta3 = self.query_refine3(q3_tokens, pqra_position3)
            q4_refined, pqra_delta4 = self.query_refine4(q4_tokens, pqra_position4)
            z3 = self.cvopm_stage3(r3, context=q3_refined)
            z4 = self.cvopm_stage4(r4, context=q4_refined)
        else:
            z3 = self.cvopm_stage3(r3, context=self._to_tokens(q3))
            z4 = self.cvopm_stage4(r4, context=self._to_tokens(q4))

        if self.variant in self.HABR_VARIANTS:
            z3, z4, habr_prior3_logits, habr_prior4_logits, habr_diagnostics = self.habr_former(z3, z4)
        elif self.variant in self.ADAPTIVE_CSFI_VARIANTS:
            z3, z4, adaptive_csfi_diagnostics = self.adaptive_csfi_reliability(
                z3, z4, self.cross_scale_interaction)
        elif self.variant in self.MHCSFI_VARIANTS:
            z3, z4, mhcsfi_diagnostics = self.mh_cross_scale_interaction(z3, z4)
        elif self.variant in self.AMHCSFI_RES_VARIANTS:
            z3, z4, amhcsfi_res_diagnostics = self.amhcsfi_res_refiner(z3, z4, self.cross_scale_interaction)
            csfi_gate3, csfi_gate4 = amhcsfi_res_diagnostics['gate3'], amhcsfi_res_diagnostics['gate4']
        elif self.variant in self.CSFI_VARIANTS:
            z3, z4, csfi_gate3, csfi_gate4 = self.cross_scale_interaction(z3, z4)

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
        elif self.variant in self.FG_AMHCSFI_RES_SINGLE_VARIANTS:
            aligned4 = self.stage4_align(z4)
            fusion_weights = None
            if self.variant == 'h2_ind_fg_amhcsfi_res_s3':
                detection_feature = z3
            elif self.variant == 'h2_ind_fg_amhcsfi_res_s4':
                detection_feature = aligned4
            elif self.variant == 'h2_ind_fg_amhcsfi_res_afuse':
                detection_feature, fusion_weights = self.stage34_adaptive_fusion(z3, aligned4)
            else:
                raise RuntimeError('Unsupported FG-AMHCSFI single-head variant')
            p = self.det_head_single(detection_feature)
            self._expect('FG-AMHCSFI single head', p, 45, 64, 64)
            predictions = {'single': p, 'fine_logits': fine_logits}
            if fusion_weights is not None:
                predictions['fusion_weights'] = fusion_weights
        elif self.variant in self.AMHCSFI_RES_BI_SINGLE_VARIANTS:
            aligned4 = self.stage4_align(z4)
            detection_feature, fusion_weights = self.stage34_adaptive_fusion(z3, aligned4)
            p = self.det_head_single(detection_feature)
            self._expect('Bi-Res AFuse single head', p, 45, 64, 64)
            predictions = {
                'single': p,
                'coarse_logits': coarse_logits,
                'fine_logits': fine_logits,
                'fusion_weights': fusion_weights,
            }
        elif self.variant in self.HIER_VARIANTS:
            p3 = self.det_head_stage3(z3)
            prior = F.interpolate(torch.sigmoid(coarse_logits), size=p3.shape[-2:],
                                  mode='bilinear', align_corners=False)
            p3_view = p3.view(p3.shape[0], 9, 5, 64, 64)
            p3_view[:, :, 4] = p3_view[:, :, 4] + torch.tanh(self.hier_conf_alpha) * \
                                (2.0 * prior - 1.0).squeeze(1).unsqueeze(1)
            p3 = p3_view.view_as(p3)
            self._expect('Hier p3', p3, 45, 64, 64)
            predictions = {'stage3': p3, 'coarse_logits': coarse_logits}
        else:
            aligned4 = self.stage4_align(z4)
            if self.variant == 'h2_shared':
                p3, p4 = self.det_head_shared(z3), self.det_head_shared(aligned4)
            else:
                p3, p4 = self.det_head_stage3(z3), self.det_head_stage4(aligned4)
            self._expect('H p3', p3, 45, 64, 64)
            self._expect('H p4', p4, 45, 64, 64)
            predictions = {'stage3': p3, 'stage4': p4}
            if self.variant in self.QCC_VARIANTS:
                q3_logits = self.qcc_quality_head_stage3(z3)
                q4_logits = self.qcc_quality_head_stage4(aligned4)
                self._expect('QCC quality3', q3_logits, 9, 64, 64)
                self._expect('QCC quality4', q4_logits, 9, 64, 64)
                predictions['qcc_quality3'] = q3_logits
                predictions['qcc_quality4'] = q4_logits
            if self.enable_hqs or self.enable_hqs_v2a or self.enable_hqs_v2b:
                batch_size = p3.shape[0]
                feat3, feat4 = z3.mean(dim=(2, 3)).detach(), aligned4.mean(dim=(2, 3)).detach()
                if self.enable_hqs:
                    p3_conf = p3.view(batch_size, 9, 5, 64, 64)[:, :, 4].reshape(batch_size, -1)
                    p4_conf = p4.view(batch_size, 9, 5, 64, 64)[:, :, 4].reshape(batch_size, -1)
                    score3 = torch.softmax(p3_conf, dim=1).max(dim=1).values.detach()
                    score4 = torch.softmax(p4_conf, dim=1).max(dim=1).values.detach()
                    predictions['hqs_logits'] = self.head_quality_selector(feat3, feat4, score3, score4)
                elif self.enable_hqs_v2a:
                    p3_conf = p3.view(batch_size, 9, 5, 64, 64)[:, :, 4].reshape(batch_size, -1)
                    p4_conf = p4.view(batch_size, 9, 5, 64, 64)[:, :, 4].reshape(batch_size, -1)
                    score3 = torch.softmax(p3_conf, dim=1).max(dim=1).values.detach()
                    score4 = torch.softmax(p4_conf, dim=1).max(dim=1).values.detach()
                    predictions['hqs_rank_logit'] = self.head_pairwise_ranker(feat3, feat4, score3, score4)
                else:
                    # Decoding-dependent quality statistics are intentionally
                    # constructed in train.py, where anchors are available.
                    predictions['hqs_feat3'] = feat3
                    predictions['hqs_feat4'] = feat4
            if coarse_logits is not None:
                predictions['coarse_logits'] = coarse_logits
            if fine_logits is not None:
                predictions['fine_logits'] = fine_logits
            if habr_prior3_logits is not None:
                predictions['habr_prior3_logits'] = habr_prior3_logits
                predictions['habr_prior4_logits'] = habr_prior4_logits

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
                      'feature_fusion={} position_injection={} position_encoder={}'.format(
                      self.variant, tuple(q3.shape), tuple(q4.shape),
                      tuple(r3.shape), tuple(r4.shape), tuple(z3.shape), tuple(z4.shape), shapes,
                       self.variant in self.CSFI_VARIANTS or self.variant in self.ADAPTIVE_CSFI_VARIANTS or self.variant in self.MHCSFI_VARIANTS,
                       self._position_mode(), self.position_mode), flush=True)
                if self.dadpe_mode != 'none':
                    print('[DADPE sanity] mode={} geometry={} gamma={:.6f}'.format(
                        self.dadpe_mode, tuple(geometry.shape), self.input_direction_residual.gamma.item()), flush=True)
                if self.dadpe_mode == 'multiscale':
                    print('[MS-DADPE sanity] p3={} p4={} beta3={:.6f} beta4={:.6f}'.format(
                        tuple(dadpe_p3.shape), tuple(dadpe_p4.shape),
                        self.multiscale_direction_residual.beta3.item(),
                        self.multiscale_direction_residual.beta4.item()), flush=True)
                if self.amr_pe_mode != 'none':
                    mean_weights = amr_weights.mean(dim=0)
                    print('[AMR-PE sanity] mode={} gamma={:.6f} alpha={:.6f} '
                          'w_f={:.6f} w_m={:.6f} w_c={:.6f} identity_error={:.8f}'.format(
                              self.amr_pe_mode, self.amr_position_field.gamma.item(), amr_alpha.item(),
                              mean_weights[0].item(), mean_weights[1].item(), mean_weights[2].item(),
                              amr_identity_error.item()), flush=True)
                if self.variant in self.CSFI_VARIANTS:
                    print('[E4-CSFI sanity] gate3_mean={:.6f} gate4_mean={:.6f} '
                          'alpha3={:.6f} alpha4={:.6f}'.format(
                              csfi_gate3.mean().item(), csfi_gate4.mean().item(),
                              self.cross_scale_interaction.alpha3.item(),
                               self.cross_scale_interaction.alpha4.item()), flush=True)
                if self.variant in self.ADAPTIVE_CSFI_VARIANTS:
                    d = adaptive_csfi_diagnostics
                    print('[A3-AR-CSFI sanity] channel={} direction={} spatial3={:.6f} spatial4={:.6f} '
                          'channel3={:.6f} channel4={:.6f} lambda43={:.6f} lambda34={:.6f} '
                          'alpha3={:.6f} alpha4={:.6f}'.format(
                              self.variant in self.ADAPTIVE_CSFI_CHANNEL_VARIANTS,
                              self.variant in self.ADAPTIVE_CSFI_DIRECTION_VARIANTS,
                              d['spatial3_mean'].item(), d['spatial4_mean'].item(),
                              d['channel3_mean'].item(), d['channel4_mean'].item(),
                              d['lambda43'].item(), d['lambda34'].item(),
                              self.cross_scale_interaction.alpha3.item(),
                              self.cross_scale_interaction.alpha4.item()), flush=True)
                if self.variant in self.MHCSFI_VARIANTS:
                    d = mhcsfi_diagnostics
                    w3, w4 = d['weights3'].mean(0), d['weights4'].mean(0)
                    print('[E4-MHCSFI sanity] adaptive={} gate3_mean={:.6f} gate4_mean={:.6f} '
                          'mask3_mean={:.6f} mask4_mean={:.6f} alpha3={:.6f} alpha4={:.6f} '
                          'w3=[{:.4f},{:.4f},{:.4f}] w4=[{:.4f},{:.4f},{:.4f}]'.format(
                              self.variant == 'h2_ind_amhcsfi_bi', d['gate3'].mean().item(), d['gate4'].mean().item(),
                              d['mask3'].mean().item(), d['mask4'].mean().item(),
                              self.mh_cross_scale_interaction.alpha3.item(), self.mh_cross_scale_interaction.alpha4.item(),
                              w3[0].item(), w3[1].item(), w3[2].item(), w4[0].item(), w4[1].item(), w4[2].item()), flush=True)
                if self.variant in self.AMHCSFI_RES_VARIANTS:
                    d = amhcsfi_res_diagnostics
                    w3, w4 = d['weights3'].mean(0), d['weights4'].mean(0)
                    print('[E4-AMHCSFI-RES sanity] alpha3={:.6f} alpha4={:.6f} beta3={:.6f} beta4={:.6f} '
                          'mod3={:.6f} mod4={:.6f} w3=[{:.4f},{:.4f},{:.4f}] w4=[{:.4f},{:.4f},{:.4f}]'.format(
                              self.cross_scale_interaction.alpha3.item(), self.cross_scale_interaction.alpha4.item(),
                              self.amhcsfi_res_refiner.refine_beta3.item(), self.amhcsfi_res_refiner.refine_beta4.item(),
                              d['modulation3'].mean().item(), d['modulation4'].mean().item(),
                              w3[0].item(), w3[1].item(), w3[2].item(), w4[0].item(), w4[1].item(), w4[2].item()), flush=True)
                if self.variant == 'h2_ind_fg_amhcsfi_res_afuse':
                    mean_weights = fusion_weights.mean(dim=0)
                    print('[E4-FG-AMHCSFI-AFuse sanity] w3={:.6f} w4={:.6f} sum={:.6f}'.format(
                        mean_weights[0].item(), mean_weights[1].item(), mean_weights.sum().item()), flush=True)
                if self.variant in self.COARSE_GUIDE_VARIANTS:
                    print('[E4-CG sanity] coarse={} coarse_up={} gamma={:.6f} hierarchical={}'.format(
                        tuple(coarse_logits.shape), tuple(coarse_up.shape), self.coarse_guidance.gamma.item(),
                        self.variant in self.HIER_VARIANTS), flush=True)
                if self.variant in self.FINE_GUIDE_VARIANTS:
                    print('[E4-FG sanity] fine={} fine_down={} gamma={:.6f} bidirectional={}'.format(
                        tuple(fine_logits.shape), tuple(fine_down.shape), self.fine_guidance.gamma.item(),
                        self.variant in self.BIDIR_GUIDE_VARIANTS), flush=True)
                if self.variant in self.HABR_VARIANTS:
                    print('[E4-HABR sanity] mode={} prior={} rounds={} lambda43={:.6f} lambda34={:.6f} '
                          'offset_mean={:.6f}'.format(
                              self.habr_former.mode, self.habr_former.use_prior, self.habr_former.rounds,
                               habr_diagnostics['lambda43'].item(), habr_diagnostics['lambda34'].item(),
                               habr_diagnostics['offset_mean'].item()), flush=True)
                if self.variant in self.QUERY_REFINE_VARIANTS:
                    print('[E4-PQRA sanity] q3_tokens={} q4_tokens={} p3_tokens={} p4_tokens={} '
                          'alpha3={:.6f} alpha4={:.6f} delta3_norm={:.6f} delta4_norm={:.6f}'.format(
                              tuple(q3_tokens.shape), tuple(q4_tokens.shape),
                              tuple(pqra_position3.shape), tuple(pqra_position4.shape),
                              self.query_refine3.alpha.item(), self.query_refine4.alpha.item(),
                              pqra_delta3.norm().item(), pqra_delta4.norm().item()), flush=True)
                if self.need_query_stage2:
                    print('[Stage2-LE sanity] q2={} no_stage2_ca=True no_stage2_head=True'.format(
                        tuple(q2.shape)), flush=True)
            self._logged_sanity = True
        return predictions, None
