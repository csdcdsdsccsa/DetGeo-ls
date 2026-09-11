"""Post-matching hierarchical adaptive bidirectional relation reasoning."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


HABR_CONFIGS = {
    'core': {'use_prior': False, 'use_reliability': False, 'rounds': 1},
    'prior': {'use_prior': True, 'use_reliability': False, 'rounds': 1},
    'adapt': {'use_prior': True, 'use_reliability': True, 'rounds': 1},
    'full': {'use_prior': True, 'use_reliability': True, 'rounds': 2},
}


class SpatialChannelGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 2, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim * 2, 64, 1), nn.GELU(),
            nn.Conv2d(64, dim, 1), nn.Sigmoid())

    def forward(self, target, message):
        fused = torch.cat((target, message), dim=1)
        return self.spatial_gate(fused) * self.channel_gate(fused)


class DeformableRelationRetrieval(nn.Module):
    """Retrieve coarse relation features at learned offsets for every fine cell."""
    def __init__(self, dim, num_samples=4, offset_scale=2.0, eps=1e-6):
        super().__init__()
        self.num_samples, self.offset_scale, self.eps = num_samples, offset_scale, eps
        self.sampling_predictor = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim), nn.GELU(),
            nn.Conv2d(dim, num_samples * 3, 1))

    def forward(self, f3, f4, prior4=None):
        batch, _, height3, width3 = f3.shape
        _, _, height4, width4 = f4.shape
        sampling = self.sampling_predictor(torch.cat((f3, F.interpolate(
            f4, size=(height3, width3), mode='bilinear', align_corners=False)), dim=1))
        offsets = torch.tanh(sampling[:, :2 * self.num_samples].view(
            batch, self.num_samples, 2, height3, width3)) * self.offset_scale
        weight_logits = sampling[:, 2 * self.num_samples:].view(batch, self.num_samples, height3, width3)
        ys = (torch.arange(height3, device=f3.device, dtype=f3.dtype) + 0.5) / height3 * 2.0 - 1.0
        xs = (torch.arange(width3, device=f3.device, dtype=f3.dtype) + 0.5) / width3 * 2.0 - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        sampled_features, sampled_priors = [], []
        for index in range(self.num_samples):
            offset_grid = torch.stack((2.0 * offsets[:, index, 0] / width4,
                                       2.0 * offsets[:, index, 1] / height4), dim=-1)
            grid = base_grid + offset_grid
            sampled_features.append(F.grid_sample(f4, grid, mode='bilinear', padding_mode='zeros',
                                                  align_corners=False))
            if prior4 is not None:
                sampled_priors.append(F.grid_sample(prior4, grid, mode='bilinear', padding_mode='zeros',
                                                    align_corners=False))
        weights = torch.softmax(weight_logits, dim=1).unsqueeze(1)
        if prior4 is not None:
            weights = weights * (self.eps + torch.stack(sampled_priors, dim=2))
            weights = weights / (weights.sum(dim=2, keepdim=True) + self.eps)
        message = (torch.stack(sampled_features, dim=2) * weights).sum(dim=2)
        return message, offsets.abs().mean()


class LocalRelationAttention(nn.Module):
    """Attend to a 4x4 fine-relation neighborhood for each coarse cell."""
    def __init__(self, dim, num_heads=4, window_size=4, eps=1e-6):
        super().__init__()
        if dim % num_heads:
            raise ValueError('relation dim must divide num_heads')
        self.num_heads, self.head_dim, self.window_size, self.eps = num_heads, dim // num_heads, window_size, eps
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, f4, f3, prior3=None):
        batch, channels, height4, width4 = f4.shape
        locations, tokens = height4 * width4, self.window_size * self.window_size
        query = self.q_proj(f4).view(batch, self.num_heads, self.head_dim, locations).permute(0, 1, 3, 2)
        def unfold_projected(value):
            patches = F.unfold(value, kernel_size=self.window_size, stride=2, padding=1)
            if patches.shape[-1] != locations:
                raise RuntimeError('local relation window count does not match Stage4 size')
            return patches.view(batch, self.num_heads, self.head_dim, tokens, locations).permute(0, 1, 4, 3, 2)
        keys, values = unfold_projected(self.k_proj(f3)), unfold_projected(self.v_proj(f3))
        attention = (query.unsqueeze(-2) * keys).sum(dim=-1) / math.sqrt(self.head_dim)
        attention = torch.softmax(attention, dim=-1)
        if prior3 is not None:
            prior = F.unfold(prior3, kernel_size=self.window_size, stride=2, padding=1)
            prior = prior.view(batch, tokens, locations).permute(0, 2, 1)
            attention = attention * (self.eps + prior.unsqueeze(1))
            attention = attention / (attention.sum(dim=-1, keepdim=True) + self.eps)
        message = (attention.unsqueeze(-1) * values).sum(dim=-2)
        message = message.permute(0, 1, 3, 2).contiguous().view(batch, channels, height4, width4)
        return self.out_proj(message)


class HABRBlock(nn.Module):
    def __init__(self, dim, num_samples, num_heads, window_size, offset_scale, use_prior, use_reliability):
        super().__init__()
        self.use_prior, self.use_reliability = use_prior, use_reliability
        self.coarse_to_fine = DeformableRelationRetrieval(dim, num_samples, offset_scale)
        self.fine_to_coarse = LocalRelationAttention(dim, num_heads, window_size)
        if use_reliability:
            self.gate3, self.gate4 = SpatialChannelGate(dim), SpatialChannelGate(dim)
            self.direction_predictor = nn.Sequential(nn.Linear(dim * 2, 128), nn.GELU(), nn.Linear(128, 2))
        self.alpha43, self.alpha34 = nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))

    def forward(self, f3, f4, prior3=None, prior4=None):
        message43, offset_mean = self.coarse_to_fine(f3, f4, prior4 if self.use_prior else None)
        if self.use_reliability:
            gate3 = self.gate3(f3, message43)
            directions = torch.softmax(self.direction_predictor(torch.cat((f3.mean((2, 3)), f4.mean((2, 3))), 1)), 1)
            lambda43, lambda34 = directions[:, 0, None, None, None], directions[:, 1, None, None, None]
        else:
            gate3, lambda43, lambda34 = 1.0, 0.5, 0.5
        f3_new = f3 + lambda43 * torch.tanh(self.alpha43) * gate3 * message43
        message34 = self.fine_to_coarse(f4, f3_new, prior3 if self.use_prior else None)
        gate4 = self.gate4(f4, message34) if self.use_reliability else 1.0
        f4_new = f4 + lambda34 * torch.tanh(self.alpha34) * gate4 * message34
        diagnostics = {
            'lambda43': (lambda43.mean() if torch.is_tensor(lambda43) else f3.new_tensor(lambda43)),
            'lambda34': (lambda34.mean() if torch.is_tensor(lambda34) else f3.new_tensor(lambda34)),
            'offset_mean': offset_mean,
            'gate3_mean': gate3.mean() if torch.is_tensor(gate3) else f3.new_tensor(gate3),
            'gate4_mean': gate4.mean() if torch.is_tensor(gate4) else f3.new_tensor(gate4),
        }
        return f3_new, f4_new, diagnostics


class HABRFormer(nn.Module):
    def __init__(self, mode='full', relation_dim=256, num_samples=4, num_heads=4, window_size=4,
                 offset_scale=2.0):
        super().__init__()
        if mode not in HABR_CONFIGS:
            raise ValueError('unsupported HABR mode: {}'.format(mode))
        config = HABR_CONFIGS[mode]
        self.mode, self.use_prior, self.rounds = mode, config['use_prior'], config['rounds']
        self.proj3 = nn.Sequential(nn.Conv2d(384, relation_dim, 1, bias=False), nn.GroupNorm(32, relation_dim), nn.GELU())
        self.proj4 = nn.Sequential(nn.Conv2d(768, relation_dim, 1, bias=False), nn.GroupNorm(32, relation_dim), nn.GELU())
        if self.use_prior:
            self.prior3_head, self.prior4_head = nn.Conv2d(384, 1, 1), nn.Conv2d(768, 1, 1)
        self.block = HABRBlock(relation_dim, num_samples, num_heads, window_size, offset_scale,
                               self.use_prior, config['use_reliability'])
        self.out3, self.out4 = nn.Conv2d(relation_dim, 384, 1, bias=False), nn.Conv2d(relation_dim, 768, 1, bias=False)

    def forward(self, z3, z4):
        base3, base4 = self.proj3(z3), self.proj4(z4)
        prior3_logits = self.prior3_head(z3) if self.use_prior else None
        prior4_logits = self.prior4_head(z4) if self.use_prior else None
        prior3 = torch.sigmoid(prior3_logits) if self.use_prior else None
        prior4 = torch.sigmoid(prior4_logits) if self.use_prior else None
        f3, f4, diagnostics = base3, base4, None
        for _ in range(self.rounds):
            f3, f4, diagnostics = self.block(f3, f4, prior3, prior4)
        return z3 + self.out3(f3 - base3), z4 + self.out4(f4 - base4), prior3_logits, prior4_logits, diagnostics
