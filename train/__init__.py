"""Training scripts for the two-stage IR→VIS latent diffusion pipeline.

Stage 1:  python -m train.train_vae     — VAE autoencoder (B03 visible)
Stage 2:  python -m train.train_ldm     — LDM U-Net + lightweight IR encoder (joint)
"""

from .dataset import VisDataset, IRDataset, PairedDataset
