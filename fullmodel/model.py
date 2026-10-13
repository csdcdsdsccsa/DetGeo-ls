"""Only the published Full Model path; module names/order match the source checkpoint."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SpatialTransformer
from .backbone import SwinTMultiStageEncoder
from .position import HiSymCoreRingGPE


class CrossScaleFeatureInteraction(nn.Module):
    def __init__(self, c3=384, c4=768):
        super().__init__()
        self.proj_4to3 = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=1, bias=False), nn.GroupNorm(32, c3), nn.GELU())
        self.gate3 = nn.Sequential(
            nn.Conv2d(2 * c3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid())
        self.proj_3to4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, c4), nn.GELU())
        self.gate4 = nn.Sequential(
            nn.Conv2d(2 * c4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid())
        self.alpha3 = nn.Parameter(torch.tensor(0.0))
        self.alpha4 = nn.Parameter(torch.tensor(0.0))

    def components(self, z3, z4):
        z4_to3 = F.interpolate(self.proj_4to3(z4), size=z3.shape[-2:],
                               mode='bilinear', align_corners=False)
        z3_to4 = self.proj_3to4(z3)
        fused3 = torch.cat((z3, z4_to3), dim=1)
        fused4 = torch.cat((z4, z3_to4), dim=1)
        return z4_to3, z3_to4, fused3, fused4, self.gate3(fused3), self.gate4(fused4)


class AdaptiveMultiReceptiveMask(nn.Module):
    def __init__(self, in_channels, relation_dim=64):
        super().__init__()
        self.relation_encoder = nn.Sequential(
            nn.Conv2d(in_channels, relation_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, relation_dim), nn.GELU())
        self.head1 = self._make_head(relation_dim, 1)
        self.head3 = self._make_head(relation_dim, 3)
        self.head5 = self._make_head(relation_dim, 5)
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(relation_dim, 32),
            nn.GELU(), nn.Linear(32, 3))
        nn.init.zeros_(self.scale_selector[-1].weight)
        nn.init.zeros_(self.scale_selector[-1].bias)

    @staticmethod
    def _make_head(channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=0, bias=False),
            nn.GELU(),
            nn.ConvTranspose2d(channels, 1, kernel_size=kernel_size, padding=0, bias=True))

    def forward(self, pair_feature):
        relation = self.relation_encoder(pair_feature)
        heads = torch.stack((self.head1(relation), self.head3(relation), self.head5(relation)), dim=1)
        weights = torch.softmax(self.scale_selector(relation), dim=1)
        logits = (heads * (3.0 * weights[:, :, None, None, None])).sum(dim=1)
        return torch.sigmoid(logits), weights


class ResidualAdaptiveMultiReceptiveCSFI(nn.Module):
    def __init__(self, c3=384, c4=768):
        super().__init__()
        self.mask3 = AdaptiveMultiReceptiveMask(2 * c3, relation_dim=64)
        self.mask4 = AdaptiveMultiReceptiveMask(2 * c4, relation_dim=64)
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


class CoarseGuidance(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.coarse_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, z4, r3):
        coarse_logits = self.coarse_head(z4)
        coarse_up = F.interpolate(torch.sigmoid(coarse_logits), size=r3.shape[-2:],
                                  mode='bilinear', align_corners=False)
        r3_guided = r3 * (1.0 + torch.tanh(self.gamma) * coarse_up)
        return r3_guided, coarse_logits, coarse_up


class FineGuidance(nn.Module):
    def __init__(self, channels=384):
        super().__init__()
        self.fine_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, z3, r4):
        fine_logits = self.fine_head(z3)
        fine_down = F.interpolate(torch.sigmoid(fine_logits), size=r4.shape[-2:],
                                  mode='bilinear', align_corners=False)
        r4_guided = r4 * (1.0 + torch.tanh(self.gamma) * fine_down)
        return r4_guided, fine_logits, fine_down


class FullModel(nn.Module):
    """Shared Swin-T, HiSym-CRGPE, bidirectional guidance and AMHCSFI-Res."""

    def __init__(self, core_sigma_y=25.0, core_sigma_x=None,
                 outer_sigma_y=50.0, outer_sigma_x=None):
        super().__init__()
        # Preserve the source module construction order and state_dict keys.
        self.encoder = SwinTMultiStageEncoder()
        self.position_embedding = HiSymCoreRingGPE(
            core_sigma=core_sigma_y, core_sigma_x=core_sigma_x,
            outer_sigma=outer_sigma_y, outer_sigma_x=outer_sigma_x)
        self.cvopm_stage3 = SpatialTransformer(
            384, 6, 64, depth=1, context_dim=384, use_self_attention=False)
        self.cvopm_stage4 = SpatialTransformer(
            768, 12, 64, depth=1, context_dim=768, use_self_attention=False)
        self.stage4_align = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.det_head_stage3 = nn.Conv2d(384, 45, kernel_size=1)
        self.det_head_stage4 = nn.Conv2d(384, 45, kernel_size=1)
        self.cross_scale_interaction = CrossScaleFeatureInteraction()
        self.coarse_guidance = CoarseGuidance()
        self.fine_guidance = FineGuidance()
        self.amhcsfi_res_refiner = ResidualAdaptiveMultiReceptiveCSFI()

    @staticmethod
    def _tokens(feature):
        return feature.flatten(2).transpose(1, 2).contiguous()

    def forward(self, query_imgs, reference_imgs, click_map):
        if click_map.shape != query_imgs.shape[:1] + query_imgs.shape[-2:]:
            raise ValueError('click_map must have shape [B,H,W] matching query images')
        query_input = self.position_embedding(query_imgs, click_map.unsqueeze(1))
        q3, q4 = self.encoder(query_input)
        r3, r4 = self.encoder(reference_imgs)
        z4_seed = self.cvopm_stage4(r4, context=self._tokens(q4))
        r3_guided, coarse_logits, _ = self.coarse_guidance(z4_seed, r3)
        z3 = self.cvopm_stage3(r3_guided, context=self._tokens(q3))
        r4_guided, fine_logits, _ = self.fine_guidance(z3, r4)
        z4 = self.cvopm_stage4(r4_guided, context=self._tokens(q4))
        z3, z4, _ = self.amhcsfi_res_refiner(z3, z4, self.cross_scale_interaction)
        aligned4 = self.stage4_align(z4)
        if aligned4.shape[-2:] != z3.shape[-2:]:
            raise RuntimeError('Stage3 and aligned Stage4 grids differ')
        p3 = self.det_head_stage3(z3)
        p4 = self.det_head_stage4(aligned4)
        return {'stage3': p3, 'stage4': p4,
                'coarse_logits': coarse_logits, 'fine_logits': fine_logits}, None
