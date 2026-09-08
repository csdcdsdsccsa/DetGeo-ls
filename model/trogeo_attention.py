"""Official TROGeo CVOPM attention blocks, ported for the w/o-OST ablation."""

import torch
import torch.nn.functional as F
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
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0,
                 query_chunk_size=None):
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = default(context_dim, query_dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.query_chunk_size = query_chunk_size
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None, context_key=None, context_value=None):
        context = default(context, x)
        # Defaulting both optional contexts to ``context`` preserves every
        # existing caller's K/V behavior bit-for-bit.
        context_key = default(context_key, context)
        context_value = default(context_value, context)
        q, k, v = self.to_q(x), self.to_k(context_key), self.to_v(context_value)
        batch, query_tokens, _ = q.shape
        if k.shape[1] != v.shape[1]:
            raise RuntimeError('CrossAttention requires equal K/V token counts, got K={} V={}'.format(
                k.shape[1], v.shape[1]))
        context_tokens = k.shape[1]
        head_dim = q.shape[-1] // self.heads
        def split_heads(tensor):
            return tensor.view(batch, tensor.shape[1], self.heads, head_dim).permute(0, 2, 1, 3).reshape(
                batch * self.heads, tensor.shape[1], head_dim
            )
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        if mask is not None:
            mask = mask.reshape(batch, -1)
            mask = mask[:, None, None, :].expand(batch, self.heads, 1, context_tokens).reshape(
                batch * self.heads, 1, context_tokens
            )
        if self.query_chunk_size is None or query_tokens <= self.query_chunk_size:
            similarity = einsum('b i d, b j d -> b i j', q, k) * self.scale
            if mask is not None:
                similarity.masked_fill_(~mask, -torch.finfo(similarity.dtype).max)
            out = einsum('b i j, b j d -> b i d', similarity.softmax(dim=-1), v)
        else:
            # Rows of Q are independent because softmax is over K/V.  Splitting
            # only the Q axis is therefore mathematically identical to full CA.
            chunks = []
            for q_chunk in q.split(self.query_chunk_size, dim=1):
                similarity = einsum('b i d, b j d -> b i j', q_chunk, k) * self.scale
                if mask is not None:
                    similarity.masked_fill_(~mask, -torch.finfo(similarity.dtype).max)
                chunks.append(einsum('b i j, b j d -> b i d', similarity.softmax(dim=-1), v))
            out = torch.cat(chunks, dim=1)
        out = out.view(batch, self.heads, query_tokens, head_dim).permute(0, 2, 1, 3).reshape(
            batch, query_tokens, self.heads * head_dim
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, dropout=0.0, use_self_attention=True,
                 query_chunk_size=None):
        super().__init__()
        self.use_self_attention = use_self_attention
        # Preserve the original construction order so all surviving modules
        # receive the same initialization stream as full CVOPM.
        self.attn1 = CrossAttention(dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout, query_chunk_size=query_chunk_size)
        self.ff = FeedForward(dim, mult=4, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        if not use_self_attention:
            self.attn1 = None
            self.norm1 = None

    def forward(self, x, context=None, context_key=None, context_value=None):
        if self.use_self_attention:
            x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, context_key=context_key,
                       context_value=context_value) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    """TROGeo CVOPM: satellite self-attention then query-conditioned cross-attention."""

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None,
                 use_self_attention=True, query_chunk_size=None):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim, dropout=dropout,
                                   use_self_attention=use_self_attention,
                                   query_chunk_size=query_chunk_size) for _ in range(depth)]
        )
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1))

    def forward(self, x, context=None, context_key=None, context_value=None):
        _, _, height, width = x.shape
        residual = x
        x = self.proj_in(self.norm(x))
        x = x.flatten(2).transpose(1, 2).contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context, context_key=context_key, context_value=context_value)
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width).contiguous()
        return self.proj_out(x) + residual
