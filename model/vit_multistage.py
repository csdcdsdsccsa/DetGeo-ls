"""ViT-T/ViT-S adapters for the strict two-scale E4 detector.

Plain ViTs expose one patch-token grid, whereas E4 expects aligned Stage3 and
Stage4 maps. We preserve all input pixels by applying the shared ImageNet ViT
to non-overlapping 256x256 tiles, stitch the patch tokens back together, and
derive Stage4 by 2x spatial pooling. Learned 1x1 projections retain E4's
original 384/768-channel Direct-CA and detector interfaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiledViTMultiStageEncoder(nn.Module):
    MODEL_NAMES = {
        'vit_t': ('deit_tiny_patch16_224', 192),
        'vit_s': ('deit_small_patch16_224', 384),
    }

    def __init__(self, backbone, tile_size=256, tile_batch_size=14):
        super().__init__()
        if backbone not in self.MODEL_NAMES:
            raise ValueError('unsupported ViT backbone: {}'.format(backbone))
        try:
            import timm
        except ImportError as error:
            raise ImportError('ViT E4 backbones require timm>=1.0,<1.1') from error

        model_name, hidden_dim = self.MODEL_NAMES[backbone]
        self.vit = timm.create_model(
            model_name, pretrained=True, num_classes=0, dynamic_img_size=True,
            pretrained_cfg_overlay={'hf_hub_id': None},
        )
        self.vit.set_grad_checkpointing(True)
        self.backbone_name = backbone
        self.hidden_dim = hidden_dim
        self.tile_size = tile_size
        self.tile_batch_size = tile_batch_size
        self.stage3_projection = nn.Conv2d(hidden_dim, 384, kernel_size=1)
        self.stage4_projection = nn.Conv2d(hidden_dim, 768, kernel_size=1)

    def _encode_tile_batch(self, tiles):
        stage3_parts, stage4_parts = [], []
        for chunk in tiles.split(self.tile_batch_size, dim=0):
            outputs = self.vit.forward_intermediates(
                chunk, indices=(7, 11), norm=True, output_fmt='NCHW',
                intermediates_only=True,
            )
            if len(outputs) != 2:
                raise RuntimeError('expected two ViT intermediate maps, got {}'.format(len(outputs)))
            stage3_parts.append(outputs[0])
            stage4_parts.append(outputs[1])
        return torch.cat(stage3_parts, dim=0), torch.cat(stage4_parts, dim=0)

    @staticmethod
    def _stitch_tiles(features, batch, tiles_h, tiles_w):
        _, channels, tile_h, tile_w = features.shape
        return features.view(batch, tiles_h, tiles_w, channels, tile_h, tile_w).permute(
            0, 3, 1, 4, 2, 5
        ).reshape(batch, channels, tiles_h * tile_h, tiles_w * tile_w).contiguous()

    def forward(self, x):
        batch, channels, height, width = x.shape
        if channels != 3 or height % self.tile_size or width % self.tile_size:
            raise RuntimeError(
                'ViT tiled encoder requires [B,3,H,W] with H/W divisible by {}, got {}'.format(
                    self.tile_size, tuple(x.shape)
                )
            )
        tiles_h, tiles_w = height // self.tile_size, width // self.tile_size
        tiles = x.unfold(2, self.tile_size, self.tile_size).unfold(
            3, self.tile_size, self.tile_size
        ).permute(0, 2, 3, 1, 4, 5).reshape(
            batch * tiles_h * tiles_w, 3, self.tile_size, self.tile_size
        ).contiguous()

        stage3_tiles, stage4_tiles = self._encode_tile_batch(tiles)
        stage3_native = self._stitch_tiles(stage3_tiles, batch, tiles_h, tiles_w)
        stage4_native = self._stitch_tiles(stage4_tiles, batch, tiles_h, tiles_w)
        stage3 = self.stage3_projection(stage3_native)
        stage4 = self.stage4_projection(F.avg_pool2d(stage4_native, kernel_size=2, stride=2))
        return stage3, stage4
