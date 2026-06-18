"""Gaussian diffusion with cosine noise schedule and DDIM sampling.

Supports:
  - Cosine (or linear) noise schedule over T=1000 timesteps.
  - DDPM training:  sample t, noise x, predict noise.
  - DDIM sampling:  deterministic fast inference (200 steps from 1000).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _cosine_schedule(T, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal, 2021)."""
    steps = torch.arange(0, T + 1, dtype=torch.float32)
    f = torch.cos((steps / T + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    betas = torch.clamp(betas, max=0.999)
    return betas


def _linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule (Ho et al., 2020)."""
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


class GaussianDiffusion(nn.Module):
    """
    DDPM diffusion helper.

    Usage (training):
        diff = GaussianDiffusion(T=1000, schedule="cosine")
        t = diff.sample_timesteps(batch_size)
        x_t, noise = diff.q_sample(x_0, t)
        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

    Usage (inference):
        x_0 = diff.ddim_sample(model, shape, conditioning_fn)
    """

    def __init__(self, T=1000, schedule="linear", beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T

        if schedule == "cosine":
            betas = _cosine_schedule(T)
        else:
            betas = _linear_schedule(T, beta_start, beta_end)

        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def sample_timesteps(self, batch_size, device=None):
        """Sample random timesteps ~ Uniform(0, T)."""
        return torch.randint(0, self.T, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε.

        Args:
            x_0:   (B, C, H, W) clean latent.
            t:     (B,) long tensor of timesteps.
            noise: optional pre-sampled noise.

        Returns:
            x_t:   noisy latent.
            noise: the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape to (B, 1, 1, 1)
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(self, model, shape, conditioning_fn, steps=200, eta=0.0):
        """
        DDIM sampling (deterministic when η=0).

        Args:
            model:           U-Net predicting noise given (x_t, t).
            shape:           (B, C, H, W) of the latent to generate.
            conditioning_fn: callable that returns the conditioning tensor
                             (e.g., IR latent).  Called once at the start;
                             the result is concatenated to x_t at each step.
            steps:           number of DDIM steps (≤ T).
            eta:             0 = deterministic DDIM, 1 = stochastic DDPM.

        Returns:
            x_0: denoised latent.
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Get conditioning once
        cond = conditioning_fn()
        if cond is not None:
            cond = cond.to(device)

        # DDIM sub-sequence from noisiest → cleanest
        seq = torch.linspace(self.T - 1, -1, steps + 1, dtype=torch.long, device=device)

        x = torch.randn(shape, device=device)  # x at timestep seq[0] = T

        for i in range(len(seq) - 1):
            t_cur = seq[i]       # current timestep (noisier)
            t_next = seq[i + 1]  # next timestep (less noisy), -1 = clean state with alpha=1

            t = torch.full((batch_size,), t_cur.item(), device=device, dtype=torch.long)

            # Build model input: [x_t || conditioning]
            model_input = torch.cat([x, cond], dim=1) if cond is not None else x
            pred_noise = model(model_input, t)

            alpha_cur = self.alphas_cumprod[t_cur]
            if t_next == -1: # the final step
                alpha_next = 1
            else:
                alpha_next = self.alphas_cumprod[t_next]

            sqrt_alpha_cur = alpha_cur.sqrt()
            sqrt_one_minus_cur = (1.0 - alpha_cur).sqrt()
            pred_x0 = (x - sqrt_one_minus_cur * pred_noise) / sqrt_alpha_cur

            # DDIM update: x at t_cur → x at t_next
            #   x_next = √(α_next)·pred_x0 + √(1-α_next - σ²)·ε̂ + σ·z
            if eta > 0:
                sigma = eta * ((1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next)).sqrt()
                pred_noise_coeff = (1.0 - alpha_next - sigma ** 2).clamp(min=0).sqrt()
                x = alpha_next.sqrt() * pred_x0 + pred_noise_coeff * pred_noise + sigma * torch.randn_like(x)
            else:
                x = alpha_next.sqrt() * pred_x0 + (1.0 - alpha_next).sqrt() * pred_noise

        return x

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, cond_latent, steps=200, eta=0.0):
        """
        Convenience wrapper when conditioning is pre-computed.

        Args:
            model:       U-Net.
            shape:       (B, C, H, W).
            cond_latent: pre-computed conditioning tensor (B, C_cond, H, W),
                         or None.
            steps:       DDIM steps.
            eta:         stochasticity.

        Returns:
            denoised latent.
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        cond = cond_latent.to(device) if cond_latent is not None else None

        # DDIM sub-sequence from noisiest → cleanest
        seq = torch.linspace(self.T - 1, -1, steps + 1, dtype=torch.long, device=device)

        x = torch.randn(shape, device=device)  # x at timestep seq[0] = T - 1

        for i in range(len(seq) - 1):
            t_cur = seq[i]       # current timestep (noisier)
            t_next = seq[i + 1]  # next timestep (less noisy)

            t = torch.full((batch_size,), t_cur.item(), device=device, dtype=torch.long)
            model_input = torch.cat([x, cond], dim=1) if cond is not None else x
            pred_noise = model(model_input, t)

            alpha_cur = self.alphas_cumprod[t_cur]
            if t_next == -1: # the final step
                alpha_next = 1
            else:
                alpha_next = self.alphas_cumprod[t_next]

            sqrt_alpha_cur = alpha_cur.sqrt()
            sqrt_one_minus_cur = (1.0 - alpha_cur).sqrt()
            pred_x0 = (x - sqrt_one_minus_cur * pred_noise) / sqrt_alpha_cur

            # DDIM update: x at t_cur → x at t_next
            #   x_next = √(α_next)·pred_x0 + √(1-α_next - σ²)·ε̂ + σ·z
            if eta > 0:
                sigma = eta * ((1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next)).sqrt()
                pred_noise_coeff = (1.0 - alpha_next - sigma ** 2).clamp(min=0).sqrt()
                x = alpha_next.sqrt() * pred_x0 + pred_noise_coeff * pred_noise + sigma * torch.randn_like(x)
            else:
                x = alpha_next.sqrt() * pred_x0 + (1.0 - alpha_next).sqrt() * pred_noise

        return x
