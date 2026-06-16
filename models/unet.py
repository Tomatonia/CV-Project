"""Conditional LDM U-Net operating at 128×128 latent resolution.

Input:  [noisy B03 latent || IR latent || angles downsampled]  →  128×128×12
Output: predicted noise                     →  128×128×4

Channel multipliers [1, 2, 2, 2] × base_ch=128  →  [128, 256, 256, 256].
Self-attention at 32×32 and 16×16.  Time embedding via sinusoidal → FiLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    ResBlock, SelfAttention, Downsample, Upsample,
    SinusoidalEmbedding, _group_norm,
)


class LDMUNet(nn.Module):
    """
    U-Net for latent diffusion (Section 3.1).

    Args:
        in_channels:  concatenated input channels (z_dim + z_dim + n_angles = 8).
        out_channels: output channels (= z_dim = 4).
        ch:           base channel count.
        ch_mult:      channel multipliers for encoder levels.
        num_res_blocks: ResBlocks per level.
        attn_resolutions: spatial sizes at which to apply self-attention.
        z_dim:        latent dimension (used for sizing).
    """

    def __init__(
        self,
        in_channels=12,
        out_channels=4,
        ch=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(32, 16),
        dropout=0.1,
        num_heads=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = len(ch_mult)  # 4 levels: 128→64→32→16

        # Time embedding: sinusoidal → MLP → FiLM in every ResBlock
        self.time_emb = SinusoidalEmbedding(dim=256, proj_dim=ch * 4)

        # ---- Encoder ----
        in_chs = [ch * m for m in ch_mult]              # [128, 256, 256, 256]
        out_chs = [ch * m for m in ch_mult]             # [128, 256, 256, 256]
        # first ResBlock at each level handles in_ch → out_ch transition
        in_chs[1] = ch * ch_mult[0]                      # level 1 in = 128 → out = 256 (transition)

        self.in_conv = nn.Conv2d(in_channels, ch * ch_mult[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(self.num_levels):
            use_attn = (128 // (2 ** i)) in attn_resolutions
            block_in = in_chs[i] if i > 0 else ch * ch_mult[0]
            block_out = out_chs[i]
            blocks = []
            for j in range(num_res_blocks):
                blk_in = block_in if j == 0 else block_out
                blk_out = block_out
                blocks.append(ResBlock(blk_in, blk_out, time_emb_dim=ch * 4, dropout=dropout))
                if use_attn:
                    blocks.append(SelfAttention(blk_out, num_heads=num_heads))
            self.enc_blocks.append(nn.ModuleList(blocks))
            if i < self.num_levels - 1:
                self.downs.append(Downsample(block_out))

        # ---- Bottleneck ----
        bottleneck_res = 128 // (2 ** (self.num_levels - 1))  # 16
        use_attn = bottleneck_res in attn_resolutions
        bot_ch = out_chs[-1]  # 256
        self.mid_block = nn.ModuleList([
            ResBlock(bot_ch, bot_ch, time_emb_dim=ch * 4, dropout=dropout),
        ])
        if use_attn:
            self.mid_block.append(SelfAttention(bot_ch, num_heads=num_heads))
        self.mid_block.append(ResBlock(bot_ch, bot_ch, time_emb_dim=ch * 4, dropout=dropout))

        # ---- Decoder ----
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(self.num_levels)):
            use_attn = (128 // (2 ** i)) in attn_resolutions
            enc_skip_ch = out_chs[i]  # channels from encoder skip at this level
            if i < self.num_levels - 1:
                dec_in_ch = out_chs[i + 1]  # channels from previous (lower-res) level
            else:
                dec_in_ch = bot_ch

            concat_ch = dec_in_ch + enc_skip_ch
            block_out = out_chs[i]

            blocks = []
            for j in range(num_res_blocks):
                blk_in = concat_ch if j == 0 else block_out
                blk_out = block_out
                blocks.append(ResBlock(blk_in, blk_out, time_emb_dim=ch * 4, dropout=dropout))
                if use_attn:
                    blocks.append(SelfAttention(blk_out, num_heads=num_heads))
            self.dec_blocks.append(nn.ModuleList(blocks))
            if i > 0:
                self.ups.append(Upsample(dec_in_ch))

        # ---- Output ----
        self.out_norm = _group_norm(ch * ch_mult[0])
        self.out_conv = nn.Conv2d(ch * ch_mult[0], out_channels, 3, padding=1)

    def forward(self, x, t):
        """
        Args:
            x: (B, in_channels, 128, 128) — noisy B03 latent concatenated with IR latent.
            t: (B,) long tensor of diffusion timesteps.

        Returns:
            (B, out_channels, 128, 128) predicted noise.
        """
        t_emb = self.time_emb(t)

        # ---- Encoder ----
        h = self.in_conv(x)
        skips = []
        for i in range(self.num_levels):
            for layer in self.enc_blocks[i]:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            skips.append(h)
            if i < self.num_levels - 1:
                h = self.downs[i](h)

        # ---- Bottleneck ----
        for layer in self.mid_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # ---- Decoder ----
        for i in range(self.num_levels):
            if i > 0:
                h = self.ups[i - 1](h)
            enc_skip = skips[self.num_levels - 1 - i]
            h = torch.cat([h, enc_skip], dim=1)
            for layer in self.dec_blocks[i]:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
