"""CV Project — Latent Diffusion Model for IR→Visible Satellite Imagery.

Modules
-------
modules       – ResBlock, SelfAttention, Downsample/Upsample, time embedding
vae           – VAE autoencoder for B03 visible (512→128 latent, z_dim=4)
discriminator – PatchGAN discriminator (70×70 receptive field)
ir_encoder    – IR encoder for B11/B13/B15 conditioning
unet          – LDM U-Net (128×128 latent, channel-wise conditioning)
diffusion     – Gaussian diffusion (cosine schedule, DDIM sampling)
"""

from .modules import (
    ResBlock, SelfAttention, Downsample, Upsample,
    SinusoidalEmbedding, get_timestep_embedding, _group_norm,
)
from .vae import VAE, VAEEncoder, VAEDecoder, kl_loss, hinge_loss_d, hinge_loss_g, ssim_loss
from .discriminator import PatchGANDiscriminator
from .ir_encoder import LightweightIREncoder
from .unet import LDMUNet
from .diffusion import GaussianDiffusion
