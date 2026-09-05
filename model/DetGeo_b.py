"""Original-P0 DetGeo B-framework: MSST, bidirectional interaction, B1 and B2."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .darknet import ConvBatchNormReLU, Darknet
from .DetGeo import CrossViewFusionModule


def build_2d_sincos_pos_embed(height, width, channels, device, dtype):
    if channels % 4:
        raise ValueError('2D sine/cosine position encoding requires channels divisible by four')
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                            torch.arange(width, device=device, dtype=dtype), indexing='ij')
    omega = torch.arange(channels // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000.0 ** (omega / (channels // 4)))
    y, x = yy.reshape(-1, 1) * omega, xx.reshape(-1, 1) * omega
    return torch.cat((y.sin(), y.cos(), x.sin(), x.cos()), dim=1).unsqueeze(0)


class MultiScaleResNet18(nn.Module):
    """ResNet18 retaining original checkpoint-compatible base_model naming."""
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        base = self.base_model
        x = base.maxpool(base.relu(base.bn1(base.conv1(x))))
        x1 = base.layer1(x)
        x2 = base.layer2(x1)
        x3 = base.layer3(x2)
        x4 = base.layer4(x3)
        return x2, x3, x4


class TokenFeatureEncoder(nn.Module):
    def __init__(self, channels, num_heads, ffn_dim):
        super().__init__()
        layer = nn.TransformerEncoderLayer(channels, num_heads, ffn_dim, dropout=0.0,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):
        return self.encoder(x)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross attention with a residual FFN."""
    def __init__(self, channels, num_heads, ffn_dim):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=0.0, batch_first=True)
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, channels))

    def forward(self, query_seq, kv_seq, return_attn=False):
        attention_out, attention = self.attn(self.norm_q(query_seq), self.norm_kv(kv_seq), self.norm_kv(kv_seq),
                                             need_weights=return_attn, average_attn_weights=False)
        output = query_seq + attention_out
        output = output + self.ffn(self.norm_ffn(output))
        return output, attention.mean(dim=1) if return_attn else None


class DetGeoB(nn.Module):
    VARIANTS = ('msst', 'core', 'b1', 'b2')

    def __init__(self, emb_size=512, leaky=True, variant='core', num_tokens=8, num_heads=8, ffn_dim=1024):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError('unknown B variant: {}'.format(variant))
        if emb_size != 512:
            raise ValueError('DetGeoB currently requires emb_size=512')
        self.variant, self.num_tokens, self.emb_size = variant, num_tokens, emb_size
        self.query_resnet = MultiScaleResNet18()
        self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('./saved_models/yolov3.weights')
        use_instnorm = False
        self.combine_clickptns_conv = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)

        self.query_proj_f = ConvBatchNormReLU(128, 512, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.query_proj_m = ConvBatchNormReLU(256, 512, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.query_mapping_visu = ConvBatchNormReLU(512, 512, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_proj_f = ConvBatchNormReLU(256, 512, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(512, 512, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_proj_c = ConvBatchNormReLU(1024, 512, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.crossview_fusionmodule = CrossViewFusionModule()

        self.shared_tokens = nn.Parameter(torch.randn(1, num_tokens, 512) * 0.02)
        self.query_encoder_f = TokenFeatureEncoder(512, num_heads, ffn_dim)
        self.query_encoder_m = TokenFeatureEncoder(512, num_heads, ffn_dim)
        self.query_encoder_c = TokenFeatureEncoder(512, num_heads, ffn_dim)
        self.t2x_f = CrossAttentionBlock(512, num_heads, ffn_dim)
        self.t2x_m = CrossAttentionBlock(512, num_heads, ffn_dim)
        self.t2x_c = CrossAttentionBlock(512, num_heads, ffn_dim)
        self.x2t_f = CrossAttentionBlock(512, num_heads, ffn_dim)
        self.x2t_m = CrossAttentionBlock(512, num_heads, ffn_dim)
        self.x2t_c = CrossAttentionBlock(512, num_heads, ffn_dim)
        self.token_score = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.scale_mlp = nn.Sequential(nn.Linear(1536, 512), nn.GELU(), nn.Linear(512, 3))
        self.fcn_out = nn.Sequential(ConvBatchNormReLU(512, 256, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
                                     nn.Conv2d(256, 9 * 5, kernel_size=1))
        self._logged_shapes = False

    @staticmethod
    def _to_sequence(feature):
        return feature.flatten(2).transpose(1, 2)

    @staticmethod
    def _to_map(sequence, height, width):
        return sequence.transpose(1, 2).reshape(sequence.shape[0], sequence.shape[2], height, width)

    def _encode_query_scale(self, feature, encoder):
        batch, channels, height, width = feature.shape
        sequence = self._to_sequence(feature) + build_2d_sincos_pos_embed(height, width, channels, feature.device, feature.dtype)
        tokens = self.shared_tokens.expand(batch, -1, -1)
        return encoder(torch.cat((tokens, sequence), dim=1))[:, :self.num_tokens]

    @staticmethod
    def _bidirectional(tokens, feature_seq, t2x, x2t):
        tokens_1, _ = t2x(tokens, feature_seq)
        feature_1, _ = x2t(feature_seq, tokens_1)
        return tokens_1, feature_1

    def _token_and_scale_weights(self, tokens_f, tokens_m, tokens_c):
        pooled, token_weights = [], []
        for tokens in (tokens_f, tokens_m, tokens_c):
            weights = torch.softmax(self.token_score(tokens).squeeze(-1), dim=1)
            token_weights.append(weights)
            pooled.append((weights.unsqueeze(-1) * tokens).sum(dim=1))
        scale_weights = torch.softmax(self.scale_mlp(torch.cat(pooled, dim=-1)), dim=-1)
        return token_weights, scale_weights

    def _fuse_features(self, seq_f, shape_f, seq_m, shape_m, seq_c, shape_c, scale_weights):
        map_f = self._to_map(seq_f, *shape_f)
        map_m = self._to_map(seq_m, *shape_m)
        map_c = self._to_map(seq_c, *shape_c)
        target_size = map_m.shape[-2:]
        map_f = F.interpolate(map_f, size=target_size, mode='bilinear', align_corners=False)
        map_c = F.interpolate(map_c, size=target_size, mode='bilinear', align_corners=False)
        weights = [scale_weights[:, index].view(-1, 1, 1, 1) for index in range(3)]
        return F.normalize(weights[0] * map_f + weights[1] * map_m + weights[2] * map_c, p=2, dim=1)

    def _response_map(self, attention_f, shape_f, attention_m, shape_m, attention_c, shape_c, token_weights, scale_weights):
        response_maps = []
        for attention, shape, weights in zip((attention_f, attention_m, attention_c), (shape_f, shape_m, shape_c), token_weights):
            response = (weights.unsqueeze(-1) * attention).sum(dim=1).reshape(-1, 1, *shape)
            response_maps.append(response)
        target_size = response_maps[1].shape[-2:]
        response_maps[0] = F.interpolate(response_maps[0], size=target_size, mode='bilinear', align_corners=False)
        response_maps[2] = F.interpolate(response_maps[2], size=target_size, mode='bilinear', align_corners=False)
        raw = sum(scale_weights[:, index].view(-1, 1, 1, 1) * response_maps[index] for index in range(3))
        flat = raw.flatten(1)
        minimum = flat.min(dim=1).values.detach().view(-1, 1, 1, 1)
        maximum = flat.max(dim=1).values.detach().view(-1, 1, 1, 1)
        return (raw - minimum) / (maximum - minimum + 1e-6)

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns.unsqueeze(1)), dim=1))
        q2, q3, q4 = self.query_resnet(query_imgs)
        qf = self.query_proj_f(F.avg_pool2d(q2, kernel_size=2, stride=2))
        qm, qc = self.query_proj_m(q3), self.query_mapping_visu(q4)
        tokens_f0 = self._encode_query_scale(qf, self.query_encoder_f)
        tokens_m0 = self._encode_query_scale(qm, self.query_encoder_m)
        tokens_c0 = self._encode_query_scale(qc, self.query_encoder_c)

        reference_raw = self.reference_darknet(reference_imgs)
        if len(reference_raw) != 3 or reference_raw[0].shape[1] != 1024 or reference_raw[1].shape[1] != 512 or reference_raw[2].shape[1] != 256:
            raise RuntimeError('unexpected Darknet scale channels: {}'.format([tuple(x.shape) for x in reference_raw]))
        feature_c0 = self.reference_proj_c(reference_raw[0])
        feature_m0 = self.reference_mapping_visu(reference_raw[1])
        feature_f0 = self.reference_proj_f(reference_raw[2])
        if not self._logged_shapes:
            print('DetGeoB shapes q(f/m/c)={} / {} / {}; r(f/m/c)={} / {} / {}'.format(
                tuple(qf.shape), tuple(qm.shape), tuple(qc.shape), tuple(feature_f0.shape), tuple(feature_m0.shape), tuple(feature_c0.shape)), flush=True)
            self._logged_shapes = True

        if self.variant == 'msst':
            query_token = torch.cat((tokens_f0, tokens_m0, tokens_c0), dim=1).mean(dim=1)
            fused, attention = self.crossview_fusionmodule(query_token, feature_m0)
            return self.fcn_out(fused), attention

        seq_f0, seq_m0, seq_c0 = self._to_sequence(feature_f0), self._to_sequence(feature_m0), self._to_sequence(feature_c0)
        tokens_f1, seq_f1 = self._bidirectional(tokens_f0, seq_f0, self.t2x_f, self.x2t_f)
        tokens_m1, seq_m1 = self._bidirectional(tokens_m0, seq_m0, self.t2x_m, self.x2t_m)
        tokens_c1, seq_c1 = self._bidirectional(tokens_c0, seq_c0, self.t2x_c, self.x2t_c)
        shape_f, shape_m, shape_c = feature_f0.shape[-2:], feature_m0.shape[-2:], feature_c0.shape[-2:]
        token_weights, scale_weights = self._token_and_scale_weights(tokens_f1, tokens_m1, tokens_c1)
        core = self._fuse_features(seq_f1, shape_f, seq_m1, shape_m, seq_c1, shape_c, scale_weights)
        if self.variant == 'core':
            return self.fcn_out(core), None
        if self.variant == 'b1':
            tokens_f2, attention_f = self.t2x_f(tokens_f1, seq_f1, return_attn=True)
            tokens_m2, attention_m = self.t2x_m(tokens_m1, seq_m1, return_attn=True)
            tokens_c2, attention_c = self.t2x_c(tokens_c1, seq_c1, return_attn=True)
            del tokens_f2, tokens_m2, tokens_c2
            response = self._response_map(attention_f, shape_f, attention_m, shape_m, attention_c, shape_c, token_weights, scale_weights)
            return self.fcn_out(core * response), response.squeeze(1)
        seq_f2, _ = self.x2t_f(seq_f1, tokens_f1)
        seq_m2, _ = self.x2t_m(seq_m1, tokens_m1)
        seq_c2, _ = self.x2t_c(seq_c1, tokens_c1)
        refined = self._fuse_features(seq_f2, shape_f, seq_m2, shape_m, seq_c2, shape_c, scale_weights)
        return self.fcn_out(refined), None
