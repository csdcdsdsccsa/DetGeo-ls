"""E05: frozen DetGeo features with coarse-to-fine anchor-free localization."""

import torch
import torch.nn as nn

from .DetGeo import ConvBatchNormReLU, MyResnet
from .darknet import Darknet
from .oacfr_modules import AnchorFreeHead, CoarseToFineSearch, ConvNormAct, spatial_softmax


REFERENCE_CHANNELS = {32: 1024, 64: 512, 128: 256}
REFERENCE_INDICES = {32: 0, 64: 1, 128: 2}


class OACFRGeoE05(nn.Module):
    """Minimal E05 model without SAM, structured tokens, recurrence, or hard negatives."""

    def __init__(self, emb_size=512, match_dim=256, search_scales=(32, 64, 128),
                 temperature=0.07, freeze_backbone=True, leaky=True):
        super().__init__()
        self.search_scales = tuple(int(scale) for scale in search_scales)
        if not self.search_scales:
            raise ValueError("search_scales cannot be empty")
        if tuple(sorted(self.search_scales)) != self.search_scales:
            raise ValueError("search_scales must be ordered from coarse to fine")
        for scale in self.search_scales:
            if scale not in REFERENCE_CHANNELS:
                raise ValueError("unsupported search scale: %s" % scale)

        # Names intentionally match DetGeo so its checkpoint can initialize the backbone.
        self.query_resnet = MyResnet()
        self.reference_darknet = Darknet(config_path="./model/yolov3_rs.cfg")
        self.reference_darknet.load_weights("./saved_models/yolov3.weights")
        self.combine_clickptns_conv = ConvBatchNormReLU(
            4, 3, 1, 1, 0, 1, leaky=leaky, instance=False)
        self.query_mapping_visu = ConvBatchNormReLU(
            512, emb_size, 1, 1, 0, 1, leaky=leaky, instance=False)
        self.reference_mapping_visu = ConvBatchNormReLU(
            512, emb_size, 1, 1, 0, 1, leaky=leaky, instance=False)

        self.query_projection = nn.Linear(emb_size, match_dim)
        self.reference_projections = nn.ModuleDict()
        for scale in self.search_scales:
            in_channels = emb_size if scale == 64 else REFERENCE_CHANNELS[scale]
            self.reference_projections[str(scale)] = ConvNormAct(in_channels, match_dim, 1)

        self.search = CoarseToFineSearch(
            match_dim, self.search_scales, temperature=temperature)
        self.fine_fusion = ConvNormAct(match_dim * 2 + 1, match_dim, 3)
        self.anchor_free_head = AnchorFreeHead(match_dim, match_dim)
        self._backbone_frozen = False
        self.freeze_backbone(freeze_backbone)

    def freeze_backbone(self, frozen=True):
        self._backbone_frozen = bool(frozen)
        modules = (
            self.query_resnet,
            self.reference_darknet,
            self.combine_clickptns_conv,
            self.query_mapping_visu,
            self.reference_mapping_visu,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = not self._backbone_frozen
            if self._backbone_frozen:
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self._backbone_frozen:
            for module in (
                    self.query_resnet, self.reference_darknet, self.combine_clickptns_conv,
                    self.query_mapping_visu, self.reference_mapping_visu):
                module.eval()
        return self

    def _extract_backbone_features(self, query_imgs, reference_imgs, mat_clickptns):
        mat_clickptns = mat_clickptns.unsqueeze(1)
        query_input = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1))
        query_features = self.query_mapping_visu(self.query_resnet(query_input))
        reference_raw = self.reference_darknet(reference_imgs)
        return query_features, reference_raw

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        if self._backbone_frozen:
            with torch.no_grad():
                query_features, reference_raw = self._extract_backbone_features(
                    query_imgs, reference_imgs, mat_clickptns)
        else:
            query_features, reference_raw = self._extract_backbone_features(
                query_imgs, reference_imgs, mat_clickptns)

        query_vector = query_features.mean(dim=(-2, -1))
        query_vector = self.query_projection(query_vector)

        projected = {}
        for scale in self.search_scales:
            feature = reference_raw[REFERENCE_INDICES[scale]]
            if scale == 64:
                feature = self.reference_mapping_visu(feature)
            projected[str(scale)] = self.reference_projections[str(scale)](feature)

        search_logits = self.search(query_vector, projected)
        fine_scale = self.search_scales[-1]
        fine_feature = projected[str(fine_scale)]
        query_map = query_vector[:, :, None, None].expand_as(fine_feature)
        attention_map = spatial_softmax(search_logits[-1])
        fused = self.fine_fusion(torch.cat((fine_feature, query_map, attention_map), dim=1))
        outputs = self.anchor_free_head(fused)
        outputs["search_logits"] = search_logits
        return outputs
