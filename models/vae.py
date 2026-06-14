"""VAE autoencoder for B03 visible satellite images (512×512×1).

f=4 compression: 512×512 → 128×128 latent (z_dim=4).
Trained with L1 + LPIPS + adversarial (hinge) + KL (β=1e-6).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    ResBlock, Downsample, Upsample, _group_norm,
    checkpoint_fn, checkpoint_resblock,
)


class VAEEncoder(nn.Module):
    """
    Encoder: 512×512×in_ch → 128×128×(2*z_dim) [μ, logvar].

    Architecture:
      Conv(1→ch) → 2×ResBlock(ch→ch)                            512×512
      Down → ResBlock(ch→2·ch) + ResBlock(2·ch→2·ch)           256×256
      Down → ResBlock(2·ch→4·ch) + ResBlock(4·ch→4·ch)        128×128
      GN → SiLU → Conv(4·ch → 2*z_dim)
    """

    def __init__(self, in_channels=1, z_dim=4, ch=64, dropout=0.0, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.use_checkpoint = use_checkpoint
        c0, c1, c2 = ch, ch * 2, ch * 4  # 64, 128, 256

        # --- 512×512 ---
        self.in_conv = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.enc_block0 = nn.ModuleList([
            ResBlock(c0, c0, dropout=dropout),
            ResBlock(c0, c0, dropout=dropout),
        ])

        # --- 256×256 ---
        self.down0 = Downsample(c0)
        self.enc_block1 = nn.ModuleList([
            ResBlock(c0, c1, dropout=dropout),
            ResBlock(c1, c1, dropout=dropout),
        ])

        # --- 128×128 ---
        self.down1 = Downsample(c1)
        self.enc_block2 = nn.ModuleList([
            ResBlock(c1, c2, dropout=dropout),
            ResBlock(c2, c2, dropout=dropout),
        ])

        # --- Output ---
        self.out_norm = _group_norm(c2)
        self.out_conv = nn.Conv2d(c2, z_dim * 2, 3, padding=1)

    def forward(self, x):
        h = self.in_conv(x)
        for layer in self.enc_block0:
            h = checkpoint_resblock(layer, h, self.use_checkpoint)

        h = self.down0(h)
        for layer in self.enc_block1:
            h = checkpoint_resblock(layer, h, False)

        h = self.down1(h)
        for layer in self.enc_block2:
            h = checkpoint_resblock(layer, h, False)

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        mu, logvar = h.chunk(2, dim=1)

        # soft-clamp the logvar to prevent KL from exploding
        a = 2.0 * math.log(3)  # std between 1/3 and 3
        logvar = a * torch.tanh(logvar / a)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Decoder: 128×128×z_dim → 512×512×out_ch.

    Architecture:
      Conv(z_dim→4·ch) → 2×ResBlock(4·ch→4·ch)               128×128
      Up → ResBlock(4·ch→2·ch) + ResBlock(2·ch→2·ch)        256×256
      Up → ResBlock(2·ch→ch) + ResBlock(ch→ch)               512×512
      GN → SiLU → Conv(ch → out_ch) → tanh
    """

    def __init__(self, out_channels=1, z_dim=4, ch=64, dropout=0.0, use_checkpoint=True):
        super().__init__()
        self.out_channels = out_channels
        self.use_checkpoint = use_checkpoint
        c0, c1, c2 = ch, ch * 2, ch * 4  # 64, 128, 256

        # --- 128×128 ---
        self.in_conv = nn.Conv2d(z_dim, c2, 3, padding=1)
        self.dec_block2 = nn.ModuleList([
            ResBlock(c2, c2, dropout=dropout),
            ResBlock(c2, c2, dropout=dropout),
        ])

        # --- 256×256 ---
        self.up0 = Upsample(c2)
        self.dec_block1 = nn.ModuleList([
            ResBlock(c2, c1, dropout=dropout),
            ResBlock(c1, c1, dropout=dropout),
        ])

        # --- 512×512 ---
        self.up1 = Upsample(c1)
        self.dec_block0 = nn.ModuleList([
            ResBlock(c1, c0, dropout=dropout),
            ResBlock(c0, c0, dropout=dropout),
        ])

        # --- Output ---
        self.out_norm = _group_norm(c0)
        self.out_conv = nn.Conv2d(c0, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.in_conv(z)
        for layer in self.dec_block2:
            h = checkpoint_resblock(layer, h, False)

        h = self.up0(h)
        for layer in self.dec_block1:
            h = checkpoint_resblock(layer, h, False)

        h = self.up1(h)
        for layer in self.dec_block0:
            h = checkpoint_resblock(layer, h, self.use_checkpoint)

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return torch.tanh(h)


class VAE(nn.Module):
    """
    Full VAE: encoder + decoder with reparameterization.

    Usage:
        vae = VAE(in_channels=1, z_dim=4)
        recon, mu, logvar = vae(x)
        z = vae.encode(x)          # sample latent (deterministic in eval mode)
        recon = vae.decode(z)      # decode from latent
    """

    def __init__(self, in_channels=1, out_channels=1, z_dim=4, ch=64, dropout=0.0,
                 use_checkpoint=True):
        super().__init__()
        self.z_dim = z_dim
        self.use_checkpoint = use_checkpoint
        self.encoder = VAEEncoder(in_channels, z_dim, ch, dropout, use_checkpoint)
        self.decoder = VAEDecoder(out_channels, z_dim, ch, dropout, use_checkpoint)

    def encode(self, x, sample=True):
        """Return z (sampled if sample=True, else μ)."""
        mu, logvar = self.encoder(x)
        if sample:
            return self._reparameterize(mu, logvar)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Checkpoint encoder as a single block: encoder activations are freed
        # during decoder forward/backward, then recomputed on demand.  This
        # prevents encoder (~40 %) and decoder (~60 %) activations from being
        # live simultaneously.
        mu, logvar = self.encoder(x)
        z = self._reparameterize(mu, logvar)
        # Per-ResBlock checkpointing inside decoder handles the decoder side.
        recon = self.decoder(z)
        return recon, mu, logvar

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# ---------------------------------------------------------------------------
# VAE loss helpers
# ---------------------------------------------------------------------------
def kl_loss(mu, logvar):
    """KL(N(mu, σ²) || N(0, 1)), averaged over batch and spatial dims."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def hinge_loss_d(real_pred, fake_pred):
    """
    Discriminator hinge loss.
    D wants: real ≥ 1, fake ≤ -1.
    """
    return F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()


def hinge_loss_g(fake_pred):
    """Generator hinge loss: G wants D(fake) as high as possible."""
    return -fake_pred.mean()
