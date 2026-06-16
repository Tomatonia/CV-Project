"""Lightweight IR encoder for conditioning the LDM U-Net.

A simple 3-level conv net:  512 → 256 → 128,  with channels 64 → 128 → 256.
Outputs a 4-channel feature map at 128×128 that is concatenated alongside
z_t and angle maps at the U-Net input.

Trained jointly with the LDM U-Net via the diffusion noise-prediction loss
(no separate pre-training stage needed).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResBlock, _group_norm


class LightweightIREncoder(nn.Module):
    """
    IR encoder: (N_IR, 512, 512) → (C_IR, 128, 128).

    Architecture:
      Conv(N_IR → ch) + ResBlock(ch)         512×512
      Strided Conv(ch → 2·ch) + ResBlock    256×256
      Strided Conv(2·ch → 4·ch) + ResBlock  128×128
      GN → SiLU → Conv(4·ch → C_IR)
    """

    def __init__(self, in_channels=3, out_channels=4, ch=64, dropout=0.0):
        super().__init__()
        c0, c1, c2 = ch, ch * 2, ch * 4  # 64, 128, 256

        # --- 512×512 ---
        self.in_conv = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.enc_block0 = ResBlock(c0, c0, dropout=dropout)

        # --- 256×256 ---
        self.down0 = nn.Conv2d(c0, c1, 4, stride=2, padding=1)
        self.enc_block1 = ResBlock(c1, c1, dropout=dropout)

        # --- 128×128 ---
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.enc_block2 = ResBlock(c2, c2, dropout=dropout)

        # --- Output ---
        self.out_norm = _group_norm(c2)
        self.out_conv = nn.Conv2d(c2, out_channels, 3, padding=1)

    def forward(self, x):
        h = self.in_conv(x)
        h = self.enc_block0(h)
        h = F.silu(self.down0(h))
        h = self.enc_block1(h)
        h = F.silu(self.down1(h))
        h = self.enc_block2(h)
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return torch.tanh(h)
