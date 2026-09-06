"""Pure backbone ablations retaining DetGeo's P0, fusion and YOLO head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .darknet import ConvBatchNormReLU, Darknet
from .DetGeo import CrossViewFusionModule


class ResNet50C4(nn.Module):
    """ImageNet-pretrained ResNet-50 through layer3 (stride 16, C4)."""
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3 = base.layer1, base.layer2, base.layer3

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer3(self.layer2(self.layer1(x)))


class DetGeoBackboneAblation(nn.Module):
    EXPERIMENTS = ('darknet53_noshare', 'darknet53_shared', 'resnet50_shared')

    def __init__(self, emb_size=512, leaky=True, backbone_exp='darknet53_noshare'):
        super().__init__()
        if backbone_exp not in self.EXPERIMENTS:
            raise ValueError('unknown backbone experiment: {}'.format(backbone_exp))
        self.backbone_exp = backbone_exp
        use_instnorm = False
        # Construct common DetGeo modules first to keep their seed consumption aligned.
        self.combine_clickptns_conv = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.crossview_fusionmodule = CrossViewFusionModule()
        feature_channels = 1024 if backbone_exp == 'resnet50_shared' else 512
        self.query_mapping_visu = ConvBatchNormReLU(feature_channels, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(feature_channels, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.fcn_out = nn.Sequential(ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
                                     nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1))
        if backbone_exp == 'darknet53_noshare':
            self.query_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
            self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
            self.query_darknet.load_weights('./saved_models/yolov3.weights')
            self.reference_darknet.load_weights('./saved_models/yolov3.weights')
        elif backbone_exp == 'darknet53_shared':
            self.shared_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
            self.shared_darknet.load_weights('./saved_models/yolov3.weights')
        else:
            self.shared_resnet50 = ResNet50C4()
        self._logged_shapes = False

    def _extract(self, query_imgs, reference_imgs):
        if self.backbone_exp == 'darknet53_noshare':
            return self.query_darknet.forward_features(query_imgs)[1], self.reference_darknet.forward_features(reference_imgs)[1]
        if self.backbone_exp == 'darknet53_shared':
            return self.shared_darknet.forward_features(query_imgs)[1], self.shared_darknet.forward_features(reference_imgs)[1]
        return self.shared_resnet50(query_imgs), self.shared_resnet50(reference_imgs)

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns.unsqueeze(1)), dim=1))
        query_raw, reference_raw = self._extract(query_imgs, reference_imgs)
        query_fvisu = self.query_mapping_visu(query_raw)
        reference_fvisu = self.reference_mapping_visu(reference_raw)
        if query_fvisu.shape[1] != 512 or reference_fvisu.shape[1] != 512 or reference_fvisu.shape[-2:] != (64, 64):
            raise RuntimeError('invalid mapped backbone shapes q={} r={}'.format(tuple(query_fvisu.shape), tuple(reference_fvisu.shape)))
        batch, channels, height, width = query_fvisu.shape
        query_global = torch.mean(query_fvisu.reshape(batch, channels, height * width), dim=2)
        fused, attention = self.crossview_fusionmodule(query_global, reference_fvisu)
        outbox = self.fcn_out(fused)
        if not self._logged_shapes:
            print('Backbone experiment={} raw_q={} raw_r={} mapped_q={} mapped_r={} global={} fused={} outbox={} attention={}'.format(
                self.backbone_exp, tuple(query_raw.shape), tuple(reference_raw.shape), tuple(query_fvisu.shape),
                tuple(reference_fvisu.shape), tuple(query_global.shape), tuple(fused.shape), tuple(outbox.shape), tuple(attention.squeeze(1).shape)), flush=True)
            self._logged_shapes = True
        return outbox, attention.squeeze(1)
