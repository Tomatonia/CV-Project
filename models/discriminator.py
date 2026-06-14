"""PatchGAN discriminator for VAE adversarial training.

70×70 receptive field, operating on 512×512 (or random 256×256 crops).
Spectral norm on every conv layer enforces 1-Lipschitz, which stabilises
hinge-loss GAN training and prevents unbounded D outputs.
"""

import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    Input:  B × in_channels × 512 × 512
    Output: B × 1 × 62 × 62   (each element classifies a 70×70 patch)
    """

    def __init__(self, in_channels=1, ch=64):
        super().__init__()
        self.in_channels = in_channels

        def sn_conv(ci, co, k, s, p):
            return spectral_norm(nn.Conv2d(ci, co, k, stride=s, padding=p))

        self.layers = nn.Sequential(
            sn_conv(in_channels, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(ch, ch * 2, 4, 2, 1),
            nn.InstanceNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(ch * 2, ch * 4, 4, 2, 1),
            nn.InstanceNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(ch * 4, ch * 8, 4, 1, 1),
            nn.InstanceNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(ch * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)
