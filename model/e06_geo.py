"""E06: baseline-preserving residual multi-scale DetGeo adapter."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DetGeo import ConvBatchNormReLU, CrossViewFusionModule, MyResnet
from .darknet import Darknet


class E06ResidualMultiScaleDetGeo(nn.Module):
    """Keep DetGeo's 64x64, 9-anchor head and add a zero-scaled residual.

    All modules with names shared with :class:`DetGeo` have exactly the same
    topology.  Consequently a DetGeo checkpoint can load them by name, and at
    ``residual_scale == 0`` the output tensor is bitwise identical to the
    original 9x5 detector path in evaluation mode.
    """

    def __init__(self, emb_size=512, leaky=True, freeze_baseline=True,
                 fusion_mode="parallel"):
        super().__init__()
        if fusion_mode not in ("parallel", "sequential"):
            raise ValueError("fusion_mode must be parallel or sequential")
        self.fusion_mode = fusion_mode
        use_instnorm = False

        # Frozen DetGeo baseline: retain names and operations for checkpoint
        # compatibility and an epoch-0 equality check.
        self.query_resnet = MyResnet()
        self.reference_darknet = Darknet(config_path="./model/yolov3_rs.cfg")
        self.reference_darknet.load_weights("./saved_models/yolov3.weights")
        self.combine_clickptns_conv = ConvBatchNormReLU(
            4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.crossview_fusionmodule = CrossViewFusionModule()
        self.query_mapping_visu = ConvBatchNormReLU(
            512, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(
            512, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.fcn_out = nn.Sequential(
            ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1,
                              leaky=leaky, instance=use_instnorm),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1),
        )

        # E06's trainable branch.  Darknet gives 1024x32, 512x64 and 256x128
        # reference features.  The 32/128 branches are projected to the same
        # 512-dimensional space and resized to 64x64 before residual fusion.
        self.coarse_projection = ConvBatchNormReLU(
            1024, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.fine_projection = ConvBatchNormReLU(
            256, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        if fusion_mode == "parallel":
            self.residual_adapter = nn.Sequential(
                ConvBatchNormReLU(emb_size * 3, emb_size, 3, 1, 1, 1,
                                  leaky=leaky, instance=use_instnorm),
                ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1,
                                  leaky=leaky, instance=use_instnorm),
                nn.Conv2d(emb_size, emb_size, kernel_size=1),
            )
            self.residual_scale = nn.Parameter(torch.zeros(1))
        else:
            # Coarse evidence first changes an intermediate 64x64 feature;
            # fine evidence then conditions the final residual.  Both scales
            # begin at zero, so this branch cannot alter frozen DetGeo at
            # initialization.
            self.coarse_adapter = nn.Sequential(
                ConvBatchNormReLU(emb_size * 2, emb_size, 3, 1, 1, 1,
                                  leaky=leaky, instance=use_instnorm),
                nn.Conv2d(emb_size, emb_size, kernel_size=1),
            )
            self.fine_adapter = nn.Sequential(
                ConvBatchNormReLU(emb_size * 2, emb_size, 3, 1, 1, 1,
                                  leaky=leaky, instance=use_instnorm),
                nn.Conv2d(emb_size, emb_size, kernel_size=1),
            )
            self.coarse_scale = nn.Parameter(torch.zeros(1))
            self.fine_scale = nn.Parameter(torch.zeros(1))
        self._baseline_frozen = False
        self.freeze_baseline(freeze_baseline)

    def freeze_baseline(self, frozen=True):
        self._baseline_frozen = bool(frozen)
        modules = (
            self.query_resnet, self.reference_darknet,
            self.combine_clickptns_conv, self.query_mapping_visu,
            self.reference_mapping_visu, self.fcn_out,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = not self._baseline_frozen
            if self._baseline_frozen:
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self._baseline_frozen:
            for module in (
                    self.query_resnet, self.reference_darknet,
                    self.combine_clickptns_conv, self.query_mapping_visu,
                    self.reference_mapping_visu, self.fcn_out):
                module.eval()
        return self

    def _baseline_features(self, query_imgs, reference_imgs, mat_clickptns):
        query_input = self.combine_clickptns_conv(torch.cat(
            (query_imgs, mat_clickptns.unsqueeze(1)), dim=1))
        query_features = self.query_mapping_visu(self.query_resnet(query_input))
        reference_raw = self.reference_darknet(reference_imgs)
        reference_64 = self.reference_mapping_visu(reference_raw[1])
        batch_size, channels, query_height, query_width = query_features.shape
        query_vector = query_features.view(
            batch_size, channels, query_height * query_width).mean(dim=2)
        fused_64, attention_64 = self.crossview_fusionmodule(query_vector, reference_64)
        return query_vector, reference_raw, fused_64, attention_64.squeeze(1)

    def residual_scales(self):
        if self.fusion_mode == "parallel":
            return {"parallel_alpha": float(self.residual_scale.detach())}
        return {
            "coarse_alpha": float(self.coarse_scale.detach()),
            "fine_alpha": float(self.fine_scale.detach()),
        }

    def forward(self, query_imgs, reference_imgs, mat_clickptns, return_features=False):
        if self._baseline_frozen:
            with torch.no_grad():
                query_vector, reference_raw, fused_64, attention_64 = self._baseline_features(
                    query_imgs, reference_imgs, mat_clickptns)
        else:
            query_vector, reference_raw, fused_64, attention_64 = self._baseline_features(
                query_imgs, reference_imgs, mat_clickptns)

        spatial_size = fused_64.shape[-2:]
        coarse_64 = F.interpolate(self.coarse_projection(reference_raw[0]),
                                  size=spatial_size, mode="bilinear", align_corners=False)
        fine_64 = F.adaptive_avg_pool2d(self.fine_projection(reference_raw[2]), spatial_size)
        coarse_context, _ = self.crossview_fusionmodule(query_vector, coarse_64)
        fine_context, _ = self.crossview_fusionmodule(query_vector, fine_64)
        if self.fusion_mode == "parallel":
            residual = self.residual_adapter(torch.cat(
                (fused_64, coarse_context, fine_context), dim=1))
            fused_final = fused_64 + self.residual_scale * residual
        else:
            coarse_delta = self.coarse_adapter(torch.cat((fused_64, coarse_context), dim=1))
            coarse_enhanced = fused_64 + self.coarse_scale * coarse_delta
            fine_delta = self.fine_adapter(torch.cat((coarse_enhanced, fine_context), dim=1))
            residual = fine_delta
            fused_final = fused_64 + self.fine_scale * fine_delta
        output = self.fcn_out(fused_final)
        if return_features:
            return output, attention_64, fused_64, residual, fused_final
        return output, attention_64
