# CV-Project
Final project for CV course (2026 spring semester): Latent Diffusion Model for IR-to-Visible Satellite Imagery

A conditional latent diffusion model (LDM) that translates 3 Himawari-9 IR bands (B11, B13, B15) plus 4 geometry angles to 512×512 visible B03. The system has two stages: (1) a VAE compresses B03 images 4× (512→128), (2) a lightweight IR encoder and conditional diffusion U-Net are trained jointly via the diffusion loss — the IR encoder feeds directly into the U-Net alongside resized angle maps (bypassing any bottleneck).

## Band Summary

| Band | Wavelength | Native Res.  | Segment Size | Quantity          | Role      |
| ---- | ---------- | ------------ | ------------ | ----------------- | --------- |
| B03  | 0.64 μm    | 0.5 km (R05) | 2000×2000    | Reflectance %     | Target    |
| B11  | 8.6 μm     | 2 km (R20)   | 500×500      | Brightness Temp   | Condition |
| B13  | 10.4 μm    | 2 km (R20)   | 500×500      | Brightness Temp   | Condition |
| B15  | 12.4 μm    | 2 km (R20)   | 500×500      | Brightness Temp   | Condition |
| —    | —          | —            | —            | Solar zenith      | Condition |
| —    | —          | —            | —            | Solar azimuth     | Condition |
| —    | —          | —            | —            | Satellite zenith  | Condition |
| —    | —          | —            | —            | Satellite azimuth | Condition |

B11, B13, B15 are window/absorption channels in the thermal infrared, chosen for their sensitivity to cloud-top temperature, water vapor, and surface emissivity. The four geometry angles provide explicit sun/satellite position per pixel, critical for disentangling illumination effects from cloud albedo.

---

## 1. VAE Autoencoder (B03 Visible Domain)

Trained from scratch on B03 reflectance images at 512×512 (single-channel). Downsampling factor f=4, so a 512×512 input → 256×256 → 128×128 latent with z_dim channels. Each latent element represents a 4×4 input patch — a modest 4:1 spatial compression that preserves fine cloud edges.

### 1.1 Encoder

```
Input: 512×512×1 (B03 reflectance, normalized to [-1, 1])
Conv2d(1, 64, kernel=3, padding=1)                  → 512×512×64
ResBlock(64→64) × 2                                   → 512×512×64
Downsample (stride=2 conv)                            → 256×256×64
ResBlock(64→128) + ResBlock(128→128)                  → 256×256×128
Downsample                                             → 128×128×128
ResBlock(128→256) + ResBlock(256→256)                 → 128×128×256
GroupNorm → SiLU → Conv2d(256, 2*z_dim, 3, pad=1)    → 128×128×(2*z_dim)  [μ, log σ²]
```

z_dim = 4 (8 channels for μ+logvar pair). GroupNorm(32) throughout — for layers with <32 channels, use GroupNorm(min(ch, 32)). SiLU activation.

### 1.2 Decoder

```
Input: 128×128×z_dim (sampled via reparameterization)
Conv2d(z_dim, 256, kernel=3, padding=1)                → 128×128×256
ResBlock(256→256) × 2                                   → 128×128×256
Upsample (nearest + conv)                                → 256×256×256
ResBlock(256→128) + ResBlock(128→128)                   → 256×256×128
Upsample                                                  → 512×512×128
ResBlock(128→64) + ResBlock(64→64)                      → 512×512×64
GroupNorm → SiLU → Conv2d(64, 1, kernel=3, pad=1)       → 512×512×1  [tanh]
```

### 1.3 Discriminator (PatchGAN)

PatchGAN with 70×70 receptive field, operating on 512×512 images (or random 256×256 crops during training to save memory).

```
Input: 512×512×1
Conv2d(1, 64, kernel=4, stride=2, padding=1) → LeakyReLU(0.2)  → 256×256×64
Conv2d(64, 128, kernel=4, stride=2, padding=1) → InstanceNorm → LeakyReLU(0.2) → 128×128×128
Conv2d(128, 256, kernel=4, stride=2, padding=1) → InstanceNorm → LeakyReLU(0.2) → 64×64×256
Conv2d(256, 512, kernel=4, stride=1, padding=1) → InstanceNorm → LeakyReLU(0.2) → 63×63×512
Conv2d(512, 1, kernel=4, stride=1, padding=1)                                    → 62×62×1
```

Hinge loss: L_D = max(0, 1 - D(real)) + max(0, 1 + D(fake)), L_G = -D(fake).

### 1.4 VAE Losses

```
L_VAE = L_rec + λ_perc * L_perc + λ_adv * L_adv + λ_kl * L_kl
```

- **L_rec**: L1 + λ\_ssim \* (1 - SSIM). L1 anchors absolute brightness; SSIM compares local patch statistics (mean, variance, covariance) through an 11×11 Gaussian window, penalising structural differences like blurred cloud edges. λ\_ssim = 1.0 by default; set to 0 for pure L1. **SSIM inputs in range $[-1,1]$.**
- **L_perc**: LPIPS with AlexNet backbone. Since LPIPS expects 3-channel RGB, replicate the single-channel B03 to 3 channels before computing LPIPS. λ\_perc = 1.0. **Disabled during training.**
- **L_adv**: Generator adversarial loss (hinge). λ\_adv = 0.5.
- **L_kl**: KL divergence between encoder posterior N(μ, σ²) and standard normal prior N(0, 1). λ\_kl = 1e-4, with a linear warmup from 0 over the first 10 epochs. This is 100× larger than the Stable Diffusion value (1e-6) which works only after pre-training with higher KL weight. Warmup lets the VAE learn to reconstruct before the prior constraint engages.

### 1.5 VAE Training

- Optimizer: Adam, lr=1e-4, β=(0.5, 0.9) for both generator and discriminator.
- Batch size: target 8–16 with gradient accumulation.
- Discriminator updates: 1 G step per 1 D step.
- Train until reconstruction quality plateaus on held-out validation set.
- EMA on generator weights (decay 0.999) for final checkpoint.

---

## 2. Lightweight IR Encoder (Joint Training)

A simple 3-level conv net trained jointly with the LDM U-Net via the diffusion noise-prediction loss. No separate pre-training stage needed.

### 2.1 Architecture

```
Input: (3, 512, 512) [B11, B13, B15]
Conv2d(3 → 64, 3, pad=1) + ResBlock(64)            → 512×512×64
Conv2d(64 → 128, 4, stride=2, pad=1) + ResBlock(128) → 256×256×128
Conv2d(128 → 256, 4, stride=2, pad=1) + ResBlock(256) → 128×128×256
GN → SiLU → Conv2d(256 → 4, 3, pad=1)               → 128×128×4
```

2.2M parameters. Strided conv downsampling instead of separate Downsample ops. Only 3 ResBlocks total (one per resolution). Outputs C_IR=4 feature maps at 128×128.

### 2.2 Joint Training

At each step:

1. Encode IR: `f_IR = IR_encoder(x_IR)` (trainable, with grad)
2. Encode VIS: `z_vis = VAE_encoder(x_vis)` (frozen, no grad)
3. Add noise: `z_t = √ᾱ_t · z_vis + √(1-ᾱ_t) · ε`
4. U-Net input: `[z_t(4) || f_IR(4) || angles_128(4)]` = 12 channels
5. Loss: MSE between predicted noise and ε

The IR encoder parameters are updated via the same diffusion loss, jointly with the U-Net. This eliminates the separate Stage 2 IR autoencoder pre-training entirely.

### 2.3 Angle Maps (Bypass)

The 4 geometry angles (solar zenith, solar azimuth, satellite zenith, satellite azimuth) do NOT go through the IR encoder. They are normalized to [-1, 1] at 512×512, then bilinear-resized to 128×128 and concatenated directly with the encoded latents at the U-Net input:

```
U-Net input = [z_t (4) || z_ir (4) || angles_128 (4)]  = 12 channels
```

This avoids forcing the encoder to compress smooth geometric fields through a texture-oriented bottleneck, and gives the U-Net direct per-pixel sun/satellite geometry.

---

## 3. Conditional Latent Diffusion Model

A U-Net operating at 128×128 in latent space. Conditioning is injected through two complementary paths: (1) channel-wise concatenation at the input for all-level spatial access, and (2) cross-attention at 32×32 and 16×16 where Q comes from U-Net features and K,V come from a lightweight ConditionEncoder that processes IR+angles into multi-scale feature maps.

```
U-Net input = [z_t (4) || z_ir (4) || angles_128 (4)] = 12 channels
```

### 3.1 U-Net Architecture

```
Input: 128×128×12  [noisy B03 latent (4) || IR latent (4) || angles (4)]

Time embedding:
  Sinusoidal(t, dim=256) → Linear(256→512) → SiLU → Linear(512→512)

Condition encoder (IR+angles → multi-scale K,V):
  Conv2d(8→128, 3, pad=1) + GN + SiLU                    → 128×128×128
  Strided-Conv(128→128, 4, stride=2) + GN + SiLU         → 64×64×128
  Strided-Conv(128→256, 4, stride=2) + GN + SiLU         → 32×32×256  (→ KV@32)
  Strided-Conv(256→256, 4, stride=2) + GN + SiLU         → 16×16×256  (→ KV@16)

Encoder path:
  Conv2d(12, 128, kernel=3, padding=1)                    → 128×128×128
  ResBlock(128→128) + ResBlock(128→128)                   → 128×128×128
  Downsample (Conv2d stride=2)                             → 64×64×128
  ResBlock(128→256) + ResBlock(256→256)                   → 64×64×256
  Downsample                                               → 32×32×256
  ResBlock(256→256) + Cross-Attn + ResBlock(256→256)     → 32×32×256
  Downsample                                               → 16×16×256
  ResBlock(256→256) + Cross-Attn + ResBlock(256→256)     → 16×16×256

Bottleneck (16×16):
  ResBlock(256→256) + Cross-Attn + ResBlock(256→256)     → 16×16×256

Decoder path:
  Upsample (nearest + conv)                                          → 32×32×256
  Concat(skip from encoder) → ResBlock(512→256) + Cross-Attn + ResBlock(256→256) → 32×32×256
  Upsample                                                           → 64×64×256
  Concat(skip) → ResBlock(512→256) + ResBlock(256→256)              → 64×64×256
  Upsample                                                           → 128×128×256
  Concat(skip) → ResBlock(384→128) + ResBlock(128→128)              → 128×128×128

Output:
  GroupNorm → SiLU → Conv2d(128, z_dim, kernel=3, padding=1)       → 128×128×4
```

- Channel multipliers: [128, 256, 256, 256].
- ResBlock: GroupNorm(32) → SiLU → Conv2d(3×3) → GroupNorm → SiLU → Conv2d(3×3) + skip (1×1 conv if channel dims differ). Time embedding injected via FiLM (scale + shift) after the first GroupNorm.
- Cross-Attention: multi-head (num_heads=4), at 32×32 and 16×16. Q from U-Net features; K,V from a lightweight ConditionEncoder that downsamples the IR+angles conditioning (8 ch at 128×128) to matching spatial resolutions (32×32, 16×16) via a strided-conv chain with GroupNorm+SiLU.
- Conditioning injection is dual-path: (1) channel-wise concatenation at the input (12 ch) gives all layers spatial access to IR+angle cues; (2) cross-attention at semantic resolutions (32×32, 16×16) lets the U-Net explicitly query conditioning features at each position. The ConditionEncoder output at each resolution is shared across encoder, bottleneck, and decoder cross-attention layers.
- Dropout: None.

### 3.2 Diffusion Process

- Timesteps T: 1000 (DDPM).
- Noise schedule: linear.
- Loss: L2 noise prediction loss (predict ε from noisy latent).
- Conditioning: dual-path — (1) z_ir (encoded IR) and angles_128 (bilinear-resized from 512×512) are concatenated with z_t → 12-channel U-Net input; (2) same conditioning processed through a lightweight ConditionEncoder (3× strided conv + GN + SiLU) → multi-scale feature maps at 32×32 and 16×16 that serve as K,V for cross-attention layers in the encoder, bottleneck, and decoder.

### 3.3 Sampling

- DDIM, 200 steps.
- No classifier-free guidance (deterministic mapping).
- At inference: sample z_T ~ N(0,I), run DDIM conditioned on z_ir, decode with VAE decoder → 512×512 B03.

### 3.4 LDM Training

- Optimizer: Adam, lr=1e-4, β=(0.9, 0.999).
- Batch size: 8–16 (128×128 latents, ~4× the pixels of a 64×64 latent, but still far cheaper than 512×512 pixel-space).
- EMA: decay 0.9999 on U-Net weights.
- Mixed precision (fp16). *Disabled for numeric stability.*

---

## 4. Full Training Pipeline

### Stage 1: VAE Training (B03 Only)

- Dataset: B03 reflectance images at 512×512, normalized to [-1, 1]. From each Himawari-9 target-area segment (2000×2000 at R05), extract a centered 512×512 crop (or random crops during training).
- Train VAE with L1 + SSIM + adversarial + KL.
- Save best generator checkpoint (lowest validation L1 + SSIM).

### Stage 2: LDM Training (Joint IR Encoder + U-Net)

- Dataset: paired (IR_stack, angles, B03), all at 512×512, from the same timestamp and segment.
- For each sample:
  1. Encode B03 with frozen VAE encoder → z_vis (128×128×4).
  2. Encode IR stack with frozen IR encoder → z_ir (128×128×4).
  3. Bilinear-resize angles from 512×512 → 128×128.
  4. Sample t ~ Uniform(0, T), noise ε ~ N(0, I).
  5. z_t = √(ᾱ_t) * z_vis + √(1-ᾱ_t) * ε.
  6. U-Net input: [z_t || z_ir || angles_128] (128×128×12).
  7. Predict noise ε̂, compute L2(ε̂, ε).

- At inference:
  1. Encode IR stack → z_ir.  Resize angles → angles_128.
  2. Sample z_T ~ N(0,I), run DDIM conditioned on [z_ir || angles_128] → z_0.
  3. Decode z_0 with VAE decoder → 512×512×1 B03.

---

## 5. Data Preparation

### 5.1 Download and Extraction

For each time step, download 4 target-area segments (i=1..4) for each band:

```
B03: R05 (0.5 km), 2000×2000 per segment — download with download_himawari9_data(dt, band=3, res=5)
B11: R20 (2 km),   500×500 per segment  — download with download_himawari9_data(dt, band=11, res=20)
B13: R20,          500×500 per segment  — download with download_himawari9_data(dt, band=13, res=20)
B15: R20,          500×500 per segment  — download with download_himawari9_data(dt, band=15, res=20)
```

### 5.2 Alignment and Cropping

1. Load each B03 segment with satpy → 2000×2000 array (reflectance %). Take a 512×512 center crop (rows 744:1256, cols 744:1256).
2. Load each B11/B13/B15 segment with satpy → 500×500 array (brightness temp K). Bilinear-upsample to 512×512 via `torch.nn.functional.interpolate` or `cv2.resize`.
3. Pair IR stack [B11, B13, B15] with B03 for the same segment.

### 5.3 Normalization

- B03: reflectance 0–100% → divide by 100 → [0, 1] → scale to [-1, 1].
- B11/B13/B15: per-channel min/max from training set → scale to [-1, 1]. Typical ranges: B11 ~190–330 K, B13 ~190–330 K, B15 ~190–330 K.

### 5.4 Caching

Pre-process and save as `.npy` files to avoid re-reading .DAT files during training. Each segment yields one file pair: `{timestamp}_seg{i}_b03.npy` (512×512) and `{timestamp}_seg{i}_ir.npy` (512×512×3).

---

## 6. Key Hyperparameters Summary

| Parameter                 | Value                                         |
| ------------------------- | --------------------------------------------- |
| IR encoder input          | B11, B13, B15 (3 channels)                    |
| Angle maps                | 4 channels, bypass encoder → 128×128 bilinear |
| U-Net input               | z_t(4) + z_ir(4) + angles(4) = 12 channels    |
| Target band (visible)     | B03 (1 channel)                               |
| VAE z_dim                 | 4                                             |
| VAE downsampling factor   | 4 (512→128 latent)                            |
| VAE λ_perc                | 1.0                                           |
| VAE λ_adv                 | 0.5                                           |
| VAE λ_kl                  | 1e-4 (warmup: 0→1e-4 over 10 epochs)          |
| VAE lr                    | 1e-4 (G and D), β=(0.5, 0.9)                  |
| LDM base channels         | 128                                           |
| LDM channel multipliers   | [1, 2, 2, 2]                                  |
| LDM attention resolutions | [32, 16]                                      |
| LDM lr                    | 1e-4, β=(0.9, 0.999)                          |
| Diffusion steps (train)   | 1000, cosine schedule                         |
| DDIM steps (inference)    | 200                                           |
| EMA decay                 | 0.9999                                        |
| Effective batch size      | 8–16                                          |
