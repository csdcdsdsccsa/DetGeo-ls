"""Official TROGeo CVOPM attention blocks, ported for the w/o-OST ablation."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn


def default(value, fallback):
    return value if value is not None else fallback


def zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def normalize(channels):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim))

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = default(context_dim, query_dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        context = default(context, x)
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda tensor: rearrange(tensor, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        similarity = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
            similarity.masked_fill_(~mask, -torch.finfo(similarity.dtype).max)
        attention = similarity.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attention, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, dropout=0.0):
        super().__init__()
        self.attn1 = CrossAttention(dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, mult=4, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    """TROGeo CVOPM: satellite self-attention then query-conditioned cross-attention."""

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim, dropout=dropout) for _ in range(depth)]
        )
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1))

    def forward(self, x, context=None):
        _, _, height, width = x.shape
        residual = x
        x = self.proj_in(self.norm(x))
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width).contiguous()
        return self.proj_out(x) + residual
