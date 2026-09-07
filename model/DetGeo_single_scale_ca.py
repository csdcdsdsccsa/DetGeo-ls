"""Single-scale spatial cross-attention ablation for the original DetGeo."""

import torch
import torch.nn as nn

from .darknet import ConvBatchNormReLU, Darknet
from .DetGeo import MyResnet


class SpatialCrossAttentionFusion(nn.Module):
    """Satellite spatial tokens attend once to spatial query tokens."""

    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=1024):
        super().__init__()
        self.mhca = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query_features, reference_features):
        batch, channels, query_height, query_width = query_features.shape
        _, reference_channels, reference_height, reference_width = reference_features.shape
        if channels != reference_channels:
            raise RuntimeError('query/reference channel mismatch: {} vs {}'.format(channels, reference_channels))

        query_tokens = query_features.flatten(2).transpose(1, 2)
        reference_tokens = reference_features.flatten(2).transpose(1, 2)
        cross_features, attention_weights = self.mhca(
            reference_tokens,
            query_tokens,
            query_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        tokens = self.norm1(reference_tokens + cross_features)
        tokens = self.norm2(tokens + self.ffn(tokens))
        fused_features = tokens.transpose(1, 2).reshape(
            batch, channels, reference_height, reference_width
        )
        # Preserve the original DetGeo return contract: one scalar map per satellite cell.
        attention_map = attention_weights.mean(dim=-1).reshape(batch, reference_height, reference_width)
        return fused_features, attention_map, query_tokens, reference_tokens, cross_features


class DetGeoSingleScaleCA(nn.Module):
    """Original DetGeo with only QACVFM replaced by one spatial MHCA block."""

    def __init__(self, emb_size=512, leaky=True):
        super().__init__()
        if emb_size != 512:
            raise ValueError('DetGeoSingleScaleCA is fixed to emb_size=512 for the original detector head')
        use_instnorm = False
        self.query_resnet = MyResnet()
        self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('./saved_models/yolov3.weights')
        self.combine_clickptns_conv = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.query_mapping_visu = ConvBatchNormReLU(512, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(512, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.spatial_cross_attention = SpatialCrossAttentionFusion(embed_dim=emb_size, num_heads=8, ffn_dim=1024)
        self.fcn_out = nn.Sequential(
            ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1),
        )
        self._logged_shapes = False

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns.unsqueeze(1)), dim=1))
        query_fvisu = self.query_mapping_visu(self.query_resnet(query_imgs))
        reference_fvisu = self.reference_mapping_visu(self.reference_darknet(reference_imgs)[1])
        if reference_fvisu.shape[-2:] != (64, 64):
            raise RuntimeError('expected 64x64 satellite grid, got {}'.format(tuple(reference_fvisu.shape[-2:])))

        fused_features, attention_map, query_tokens, reference_tokens, cross_features = self.spatial_cross_attention(
            query_fvisu, reference_fvisu
        )
        outbox = self.fcn_out(fused_features)
        if not self._logged_shapes:
            print(
                'SingleScaleCA query_fvisu={} query_tokens={} reference_fvisu={} '
                'reference_tokens={} cross_features={} fused_features={} outbox={}'.format(
                    tuple(query_fvisu.shape), tuple(query_tokens.shape), tuple(reference_fvisu.shape),
                    tuple(reference_tokens.shape), tuple(cross_features.shape), tuple(fused_features.shape), tuple(outbox.shape)
                ),
                flush=True,
            )
            self._logged_shapes = True
        return outbox, attention_map
