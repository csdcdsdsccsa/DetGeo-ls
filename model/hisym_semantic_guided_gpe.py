import torch
import torch.nn as nn
import torch.nn.functional as F
from .hisym_pae_fusion import HiSymPAEFusion


class HiSymSemanticGuidedGPE(nn.Module):
    """Zero-start SAM semantic residual over the unmodified HiSym-GPE branch."""
    def __init__(self, hidden_channels=16):
        super().__init__()
        # Natural construction order: geometry first, semantic branch second.
        self.geometry_fusion = HiSymPAEFusion()
        self.semantic_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 1, bias=True))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.last_diagnostics = {}

    @staticmethod
    def _boundary(mask):
        return torch.clamp(F.max_pool2d(mask, 3, 1, 1) + F.max_pool2d(-mask, 3, 1, 1), 0.0, 1.0)

    def forward(self, rgb, gaussian, sam_mask):
        expected = (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3])
        if rgb.ndim != 4 or rgb.shape[1] != 3 or tuple(gaussian.shape) != expected or tuple(sam_mask.shape) != expected:
            raise ValueError('requires rgb [B,3,H,W], gaussian and sam_mask [B,1,H,W]')
        geometry = self.geometry_fusion(rgb, gaussian)
        mask = sam_mask.to(device=rgb.device, dtype=rgb.dtype).clamp(0.0, 1.0)
        masked_gaussian, boundary = gaussian * mask, self._boundary(mask)
        delta = self.semantic_encoder(torch.cat((mask, masked_gaussian, boundary), dim=1))
        weight = torch.tanh(self.alpha)
        output = geometry + weight * delta
        self.last_diagnostics = {'alpha': self.alpha.detach(), 'semantic_weight': weight.detach(),
                                 'boundary_mean': boundary.detach().mean(), 'semantic_delta_abs_mean': delta.detach().abs().mean()}
        return output
