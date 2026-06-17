"""Shared building blocks for VAE, IR encoder, and LDM U-Net."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint


# ---------------------------------------------------------------------------
# Gradient checkpointing helpers
# ---------------------------------------------------------------------------
def checkpoint_fn(fn, *args, use_checkpoint=True):
    """Wrap fn(*args) with gradient checkpointing when enabled and grad is on."""
    if use_checkpoint and torch.is_grad_enabled():
        return _torch_checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def checkpoint_resblock(block, x, use_checkpoint=True):
    """Checkpoint a single ResBlock call:  block(x) → tensor."""
    return checkpoint_fn(block, x, use_checkpoint=use_checkpoint)


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------
def _group_norm(channels, num_groups=32):
    return nn.GroupNorm(min(num_groups, channels), channels)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim, max_period=10000):
    """
    Sinusoidal embedding of DDPM timesteps.

    Args:
        timesteps: (B,) long tensor of integer timesteps.
        embedding_dim: output dimension (must be even).
        max_period: controls the minimum frequency.

    Returns:
        (B, embedding_dim) float tensor.
    """
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class SinusoidalEmbedding(nn.Module):
    """Project sinusoidal time embedding to a higher dimension."""

    def __init__(self, dim, proj_dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, t):
        emb = get_timestep_embedding(t, self.dim)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# ResBlock with optional time-conditioning (FiLM)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU + Conv3×3.

    When time_emb_dim is provided, the time embedding is projected to
    scale + shift and applied after the first GroupNorm (FiLM modulation).
    """

    def __init__(self, in_ch, out_ch, time_emb_dim=None, dropout=0.0, num_groups=32):
        super().__init__()
        self.norm1 = _group_norm(in_ch, num_groups)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = _group_norm(out_ch, num_groups)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_emb_dim = time_emb_dim
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_ch * 2)
            # nn.init.zeros_(self.time_proj.weight)
            # nn.init.zeros_(self.time_proj.bias)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        if self.time_emb_dim is not None and t_emb is not None:
            scale, shift = self.time_proj(t_emb).chunk(2, dim=1)
            # scale = 2.0 * torch.tanh(scale / 2.0)   # range [-2, 2]
            # shift = 2.0 * torch.tanh(shift / 2.0)   # range [-2, 2]
            h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Self-attention
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """
    Multi-head self-attention with residual connection.

    Uses F.scaled_dot_product_attention which automatically dispatches to
    Flash Attention 2, Memory-Efficient Attention, or a fallback, depending
    on CUDA capability and input shape.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # (B, C, H, W) → (B, heads, HW, head_dim)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Cross-attention (Q from local features, K/V from conditioning)
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: Q from U-Net features, K/V from conditioning.

    Q source (x) and K/V source (kv) must have the same spatial resolution.
    The KV channels are projected to match the Q channels internally.
    """

    def __init__(self, query_channels, kv_channels=None, num_heads=4):
        super().__init__()
        if kv_channels is None:
            kv_channels = query_channels
        assert query_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = query_channels // num_heads

        self.norm_q = _group_norm(query_channels)
        self.norm_kv = _group_norm(kv_channels)
        self.q = nn.Conv2d(query_channels, query_channels, 1, bias=False)
        self.k = nn.Conv2d(kv_channels, query_channels, 1, bias=False)
        self.v = nn.Conv2d(kv_channels, query_channels, 1, bias=False)
        self.proj = nn.Conv2d(query_channels, query_channels, 1)

    def forward(self, x, kv):
        """
        Args:
            x:  (B, C_q, H, W) — U-Net features (Q source).
            kv: (B, C_kv, H, W) — conditioning features (K/V source), same H,W.

        Returns:
            (B, C_q, H, W) — attended features with residual connection.
        """
        B, C, H, W = x.shape
        q = self.q(self.norm_q(x))
        k = self.k(self.norm_kv(kv))
        v = self.v(self.norm_kv(kv))

        q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------
class Downsample(nn.Module):
    """Stride-2 convolution for spatial downsampling."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsampling followed by a 3×3 convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
