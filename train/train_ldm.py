#!/usr/bin/env python3
"""Stage 2 (was 3): Train conditional latent diffusion model (IR + angles → B03).

The lightweight IR encoder is trained jointly with the U-Net via the diffusion
noise-prediction loss — no separate IR pre-training stage needed.

Requires a frozen VAE checkpoint from Stage 1.

U-Net input = [z_t(4) || f_IR(4) || angles(4)] = 12 channels.

Usage:
  python -m train.train_ldm \
      --vae_ckpt checkpoints/vae_stage1_e099.pt \
      --batch_size 16 --epochs 200
"""

import os
import gc
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch._dynamo
# torch._dynamo.config.cache_size_limit = 2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE, ssim_loss
from models.ir_encoder import LightweightIREncoder
from models.unet import LDMUNet
from models.diffusion import GaussianDiffusion
from train.dataset import PairedDataset

class EMA(nn.Module):
    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.store = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        self.store.clear()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.store[name] = p.detach().clone()
                p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.store[name])
        self.store.clear()


def _strip_compile_prefix(state_dict):
    """Remove '_orig_mod.' prefix added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


@torch.inference_mode()
def _validate(unet, diff, vae, ir_encoder, val_loader, device, latent_scale=0.25, steps=200):
    """DDIM sampling on validation set → decode → L1 + SSIM."""
    unet.eval()
    ir_encoder.eval()
    total_l1 = total_ssim = 0.0
    count = 0

    for ir, angles, vis in val_loader:
        B = ir.size(0)
        ir = ir.to(device)
        angles = angles.to(device)
        vis = vis.to(device)

        f_ir = ir_encoder(ir)
        angles_128 = F.interpolate(angles, size=(128, 128), mode="bilinear")
        cond = torch.cat([f_ir, angles_128], dim=1)  # (B, 8, 128, 128)

        z_0 = diff.ddim_sample_loop(unet, (B, 4, 128, 128), cond, steps=steps) # 50 forward steps for faster speed
        z_0 = z_0 / latent_scale
        recon = vae.decode(z_0)
        total_l1 += F.l1_loss(recon, vis).item() * B
        total_ssim += ssim_loss(recon, vis).item() * B
        count += B

    unet.train()
    ir_encoder.train()
    return total_l1 / count, total_ssim / count


def main():
    parser = argparse.ArgumentParser(description="LDM training (joint IR encoder + U-Net)")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/data")
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to Stage-1 VAE checkpoint")
    parser.add_argument("--ir_ckpt", type=str, default=None,
                        help="Optional: resume IR encoder weights")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--z_dim", type=int, default=4)
    parser.add_argument("--ir_out_ch", type=int, default=4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--log_dir", type=str, default="runs/ldm")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    scaler = None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    print(f"Device: {device}  |  AMP: {use_amp}  |  Logs: {args.log_dir}")

    # ---- Frozen VAE encoder ----
    print("Loading VAE …")
    vae = VAE(in_channels=1, out_channels=1, z_dim=args.z_dim).to(device)
    ckpt_vae = torch.load(args.vae_ckpt, map_location=device, weights_only=True)
    vae.load_state_dict(_strip_compile_prefix(ckpt_vae["vae"]))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    del ckpt_vae

    # ---- Lightweight IR encoder (trained jointly) ----
    ir_encoder = LightweightIREncoder(
        in_channels=3, out_channels=args.ir_out_ch, ch=64,
    ).to(device)
    if args.ir_ckpt:
        ckpt_ir = torch.load(args.ir_ckpt, map_location=device, weights_only=False)
        ir_encoder.load_state_dict(_strip_compile_prefix(ckpt_ir.get("ir_encoder", ckpt_ir)))
        del ckpt_ir

    # ---- Data ----
    train_ds = PairedDataset(split="train", data_dir=args.data_dir)
    val_ds = PairedDataset(split="val", data_dir=args.data_dir)
    print(f"Train pairs: {len(train_ds)}  |  Val pairs: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Diffusion ----
    diff = GaussianDiffusion(T=args.T, schedule="cosine").to(device)

    # ---- U-Net ----
    unet = LDMUNet(
        in_channels=args.z_dim + args.ir_out_ch + 4,  # z_t(4) + f_IR(4) + angles(4)
        out_channels=args.z_dim,
        ch=128,
        ch_mult=(1, 2, 2, 2),
    ).to(device)

    if device.type == "cuda":
        ir_encoder = torch.compile(ir_encoder, mode="reduce-overhead")
        unet = torch.compile(unet, mode="reduce-overhead")

    # ---- Optimizer (IR encoder + U-Net, jointly) ----
    trainable_params = list(ir_encoder.parameters()) + list(unet.parameters())
    opt = torch.optim.Adam(trainable_params, lr=args.lr, betas=(0.9, 0.999))

    # EMA on U-Net only (IR encoder is small enough to not need it)
    ema = EMA(unet, decay=args.ema_decay)
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(_strip_compile_prefix(ckpt["unet"]))
        ir_encoder.load_state_dict(_strip_compile_prefix(ckpt["ir_encoder"]))
        opt.load_state_dict(ckpt["opt"])
        ema.shadow = ckpt["ema_shadow"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")
        del ckpt

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        
    # ---- Loop ----
    for epoch in range(start_epoch, args.epochs):
        ir_encoder.train()
        unet.train()
        loss_sum = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", unit="batch")
        for step, (ir, angles, vis) in enumerate(pbar):
            assert torch.isfinite(ir).all()
            ir = ir.to(device)
            angles = angles.to(device)
            vis = vis.to(device)
            B = ir.size(0)

            # Encode VIS (frozen, no grad)
            with torch.inference_mode():
                z_vis, _ = vae.encoder(vis) # mu, logvar
                if not torch.isfinite(z_vis).all():
                    opt.zero_grad(set_to_none=True)
                    tqdm.write(f"  [WARN] NaN in VAE mu at epoch {epoch} step {step} — skipping")
                    continue 
                z_vis = 0.25 * z_vis # scale to std around 1.0 to be the same magnitude as the noise added
                # z_vis = z_vis.clamp(-4.0, 4.0)

            if not torch.isfinite(angles).all():
                opt.zero_grad(set_to_none=True)
                tqdm.write(f"  [WARN] NaN in angles at epoch {epoch} step {step} — skipping")
                continue
            angles_128 = F.interpolate(angles, size=(128, 128), mode="bilinear")
            t = diff.sample_timesteps(B, device)
            z_t, noise = diff.q_sample(z_vis, t)

            # IR encoder + U-Net forward (trainable, with grad)
            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
                f_ir = ir_encoder(ir)
                if not torch.isfinite(z_t).all():
                    opt.zero_grad(set_to_none=True)
                    tqdm.write(f"  [WARN] NaN in z_t at epoch {epoch} step {step} — skipping")
                    continue
                if not torch.isfinite(angles_128).all():
                    opt.zero_grad(set_to_none=True)
                    tqdm.write(f"  [WARN] NaN in angles_128 at epoch {epoch} step {step} — skipping")
                    continue
                if not torch.isfinite(f_ir).all():
                    opt.zero_grad(set_to_none=True)
                    tqdm.write(f"  [WARN] NaN in f_ir at epoch {epoch} step {step} — skipping")
                    continue
                model_input = torch.cat([z_t, f_ir, angles_128], dim=1)
                pred_noise = unet(model_input, t)

            # MSE in fp32 — avoids AMP overflow
            loss = F.mse_loss(pred_noise.float(), noise.float())

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                tqdm.write(f"  [WARN] NaN loss at epoch {epoch} step {step} — skipping batch")
                # print(f"   pred_noise min/max: {pred_noise.min().item():.3f} {pred_noise.max().item():.3f}")
                gc.collect()
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_sum += loss.item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
                    scaler.step(opt)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(unet)

            '''
            if step % 100 == 0:
                with torch.no_grad():
                    print(
                        "z_vis:",
                        "min", z_vis.min().item(),
                        "max", z_vis.max().item(),
                        "mean", z_vis.mean().item(),
                        "std", z_vis.std().item(),
                    )
                    print("angles:",
                          "min", angles_128.min().item(),
                          "max", angles_128.max().item(),
                          "mean", angles_128.mean().item(),
                          "std", angles_128.std().item(),
                    )
            '''

            # Free large intermediates — prevents RAM bloat from torch.compile
            # caches and stale computation graph references
            del z_vis, angles_128, f_ir, z_t, noise, model_input, pred_noise, loss
            if step % 100 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            batches += 1
            pbar.set_postfix(MSE=f"{loss_sum / batches:.6f}")

        n = max(batches, 1)
        writer.add_scalar("train/MSE", loss_sum / n, epoch)

        # ---- Validation ----
        if (epoch + 1) % args.val_every == 0 and len(val_loader) > 0:
            ema.apply(unet)
            with torch.no_grad():
                val_l1, val_ssim = _validate(unet, diff, vae, ir_encoder, val_loader,
                                   device, steps=args.ddim_steps)
            ema.restore(unet)
            tqdm.write(f"  Val L1 (DDIM{args.ddim_steps}) = {val_l1:.4f}, Val SSIM = {val_ssim:.4f}")
            writer.add_scalar("val/L1", val_l1, epoch)
            writer.add_scalar("val/SSIM", val_ssim, epoch)
            gc.collect()

        # ---- Checkpoint ----
        if (epoch + 1) % args.save_every == 0:
            ema.apply(unet)
            path = os.path.join(args.out_dir, f"ldm_stage2_e{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "unet": unet.state_dict(),
                    "ir_encoder": ir_encoder.state_dict(),
                    "opt": opt.state_dict(),
                    "ema_shadow": ema.shadow,
                },
                path,
            )
            ema.restore(unet)
            tqdm.write(f"  Saved → {path}")

        pbar.close()
    
    writer.close()
    print("Stage 2 (LDM) complete.")


if __name__ == "__main__":
    main()
