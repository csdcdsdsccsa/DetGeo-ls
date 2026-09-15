"""Agreement-conditioned local residual box refinement for frozen Bi-Res."""

import torch
import torch.nn as nn
from torchvision.ops import roi_align


def box_to_code(box, image_wh, eps=1e-6):
    """Convert pixel xyxy boxes to normalized center/log-size codes."""
    width = (box[:, 2] - box[:, 0]).clamp_min(eps)
    height = (box[:, 3] - box[:, 1]).clamp_min(eps)
    center_x = 0.5 * (box[:, 0] + box[:, 2])
    center_y = 0.5 * (box[:, 1] + box[:, 3])
    scale = float(image_wh)
    return torch.stack((center_x / scale, center_y / scale,
                        torch.log(width / scale + eps), torch.log(height / scale + eps)), dim=1)


def code_to_box(code, image_wh):
    """Convert normalized center/log-size codes to pixel xyxy boxes."""
    scale = float(image_wh)
    center_x, center_y = code[:, 0] * scale, code[:, 1] * scale
    width, height = torch.exp(code[:, 2]) * scale, torch.exp(code[:, 3]) * scale
    return torch.stack((center_x - 0.5 * width, center_y - 0.5 * height,
                        center_x + 0.5 * width, center_y + 0.5 * height), dim=1)


class ACRRefinementHead(nn.Module):
    """Local two-scale residual refinement on an already-agreeing Bi-Res pair.

    The zero initialized output layers make the initial result *exactly* the
    score-weighted fusion used by the established agreement decoder.
    """

    def __init__(self, channels=384, hidden_dim=128, roi_size=5, num_heads=4):
        super().__init__()
        self.roi_size = roi_size
        self.proj3 = nn.Sequential(nn.Conv2d(channels, hidden_dim, 1, bias=False),
                                   nn.GroupNorm(8, hidden_dim), nn.GELU())
        self.proj4 = nn.Sequential(nn.Conv2d(channels, hidden_dim, 1, bias=False),
                                   nn.GroupNorm(8, hidden_dim), nn.GELU())
        self.norm3, self.norm4 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.cross34 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross43 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm3, self.ffn_norm4 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.ffn3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.ffn4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.refine_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + 7, 256), nn.LayerNorm(256), nn.GELU(),
                                        nn.Linear(256, 128), nn.GELU())
        self.alpha_head, self.box_head = nn.Linear(128, 1), nn.Linear(128, 4)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.box_head.weight)
        nn.init.zeros_(self.box_head.bias)
        self.register_buffer('box_delta_scale', torch.tensor([0.10, 0.10, 0.20, 0.20]), persistent=False)

    @staticmethod
    def _geometry(box):
        center_x = 0.5 * (box[:, 0] + box[:, 2])
        center_y = 0.5 * (box[:, 1] + box[:, 3])
        width = (box[:, 2] - box[:, 0]).clamp_min(1e-6)
        height = (box[:, 3] - box[:, 1]).clamp_min(1e-6)
        return center_x, center_y, width, height

    @staticmethod
    def _sanitize_roi(box, image_wh):
        box = box.detach()
        upper = float(image_wh - 1)
        x1, y1 = box[:, 0].clamp(0.0, upper - 1.0), box[:, 1].clamp(0.0, upper - 1.0)
        x2, y2 = box[:, 2].clamp(1.0, upper), box[:, 3].clamp(1.0, upper)
        return torch.stack((x1, y1, torch.maximum(x2, x1 + 1.0), torch.maximum(y2, y1 + 1.0)), dim=1)

    def forward(self, feat3, feat4, box3, box4, score3, score4, pair_iou, image_wh):
        eps = 1e-6
        alpha0 = score3 / (score3 + score4).clamp_min(1e-12)
        base_fused_box = alpha0[:, None] * box3 + (1.0 - alpha0[:, None]) * box4
        roi_box = self._sanitize_roi(base_fused_box, image_wh)
        batch_index = torch.arange(feat3.shape[0], device=feat3.device, dtype=roi_box.dtype)[:, None]
        rois = torch.cat((batch_index, roi_box), dim=1)
        scale = float(feat3.shape[-1]) / float(image_wh)
        roi3 = roi_align(feat3, rois, output_size=(self.roi_size, self.roi_size), spatial_scale=scale,
                         sampling_ratio=2, aligned=True)
        roi4 = roi_align(feat4, rois, output_size=(self.roi_size, self.roi_size), spatial_scale=scale,
                         sampling_ratio=2, aligned=True)
        p3, p4 = self.proj3(roi3).flatten(2).transpose(1, 2), self.proj4(roi4).flatten(2).transpose(1, 2)
        n3, n4 = self.norm3(p3), self.norm4(p4)
        a34, _ = self.cross34(n3, n4, n4, need_weights=False)
        a43, _ = self.cross43(n4, n3, n3, need_weights=False)
        p3, p4 = p3 + a34, p4 + a43
        p3, p4 = p3 + self.ffn3(self.ffn_norm3(p3)), p4 + self.ffn4(self.ffn_norm4(p4))
        f3, f4 = p3.mean(dim=1), p4.mean(dim=1)
        cx3, cy3, w3, h3 = self._geometry(box3)
        cx4, cy4, w4, h4 = self._geometry(box4)
        geometry = torch.stack((score3, score4, pair_iou, (cx3 - cx4) / float(image_wh),
                                (cy3 - cy4) / float(image_wh), torch.log((w3 + eps) / (w4 + eps)),
                                torch.log((h3 + eps) / (h4 + eps))), dim=1)
        hidden = self.refine_mlp(torch.cat((f3, f4, geometry), dim=1))
        delta_alpha = self.alpha_head(hidden).squeeze(1)
        # A bounded residual confidence correction.  Writing the fusion in
        # residual form is important: at zero-init it is bit-exactly the
        # established score-weighted fusion, rather than only algebraically so.
        alpha = (alpha0 + 0.5 * torch.tanh(delta_alpha)).clamp(0.0, 1.0)
        adjusted_fused_box = base_fused_box + (alpha - alpha0)[:, None] * (box3 - box4)
        box_delta = torch.tanh(self.box_head(hidden)) * self.box_delta_scale.to(hidden)
        fused_code = box_to_code(adjusted_fused_box, image_wh)
        refined_code = fused_code + box_delta
        # Convert the code correction to a direct xyxy residual.  Every term
        # is exactly zero when box_delta is zero, preserving baseline decoding
        # without log/exp reconstruction round-off.
        _, _, fused_width, fused_height = self._geometry(adjusted_fused_box)
        delta_center_x, delta_center_y = box_delta[:, 0] * float(image_wh), box_delta[:, 1] * float(image_wh)
        delta_width = fused_width * torch.expm1(box_delta[:, 2])
        delta_height = fused_height * torch.expm1(box_delta[:, 3])
        refined_box = adjusted_fused_box + torch.stack((delta_center_x - 0.5 * delta_width,
                                                         delta_center_y - 0.5 * delta_height,
                                                         delta_center_x + 0.5 * delta_width,
                                                         delta_center_y + 0.5 * delta_height), dim=1)
        return {'refined_box': refined_box, 'refined_code': refined_code,
                'base_fused_box': base_fused_box, 'alpha0': alpha0, 'alpha': alpha,
                'delta_alpha': delta_alpha, 'box_delta': box_delta}
