"""Standalone TROGeo-style shared-Swin CVOPM detector without OST supervision."""

import torch
import torch.nn as nn
import torchvision.models as models

from .trogeo_attention import SpatialTransformer


class SwinFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        swin = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        self.features = swin.features

    def forward(self, x):
        return self.features(x).permute(0, 3, 1, 2).contiguous()


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
    )


class TROGeoWoOST(nn.Module):
    """TROGeo detection path only: shared Swin-S, CVOPM and 45-channel anchor head."""

    def __init__(self, emb_size=768, use_satellite_self_attention=True):
        super().__init__()
        if emb_size != 768:
            raise ValueError('TROGeoWoOST is fixed to the official emb_size=768')
        self.encoder = SwinFeatureEncoder()  # The single registered encoder is necessarily shared.
        self.position_embedding = double_conv(4, 3)
        self.use_satellite_self_attention = use_satellite_self_attention
        self.cvopm = SpatialTransformer(
            in_channels=768, n_heads=12, d_head=64, depth=1, context_dim=768,
            use_self_attention=use_satellite_self_attention,
        )
        self.det_head = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 9 * 5, kernel_size=1),
        )
        self._logged_sanity = False

    def forward(self, query_imgs, reference_imgs, click_map):
        query_input = self.position_embedding(torch.cat((query_imgs, click_map.unsqueeze(1)), dim=1))
        query_features = self.encoder(query_input)
        reference_features = self.encoder(reference_imgs)
        if query_features.shape[1:] != (768, 8, 8):
            raise RuntimeError('expected Drone query feature [B,768,8,8], got {}'.format(tuple(query_features.shape)))
        if reference_features.shape[1:] != (768, 32, 32):
            raise RuntimeError('expected satellite feature [B,768,32,32], got {}'.format(tuple(reference_features.shape)))
        context = query_features.flatten(2).transpose(1, 2).contiguous()
        fused_features = self.cvopm(reference_features, context=context)
        outbox = self.det_head(fused_features)
        if outbox.shape[1:] != (45, 64, 64):
            raise RuntimeError('expected detector output [B,45,64,64], got {}'.format(tuple(outbox.shape)))
        if not self._logged_sanity:
            print(
                '[TROGeo {} sanity] shared_encoder=True satellite_self_attention={} cross_attention=True '
                'Q=satellite KV=query query_input={} Fq={} Fr={} Fp={} outbox={} OST=False'.format(
                    'w/o OST' if self.use_satellite_self_attention else 'Direct-CA',
                    self.use_satellite_self_attention, tuple(query_input.shape), tuple(query_features.shape),
                    tuple(reference_features.shape), tuple(fused_features.shape), tuple(outbox.shape)
                ), flush=True,
            )
            self._logged_sanity = True
        return outbox, None
