#!/usr/bin/env python3
"""Stage 1: Train VAE autoencoder on B03 visible satellite images.

Losses:  L1_recon + λ_perc * LPIPS + λ_adv * hinge_G + λ_kl * KL

Usage:
  python -m train.train_vae --data_dir data --batch_size 8 --epochs 100
  python -m train.train_vae --resume checkpoints/vae_stage1.pt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE, kl_loss, hinge_loss_d, hinge_loss_g
from models.discriminator import PatchGANDiscriminator
from train.dataset import VisDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


class EMA(nn.Module):
    """Exponential Moving Average wrapper."""

    def __init__(self, model, decay=0.999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.store = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        """Copy shadow weights → model (call before eval)."""
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.store[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model weights after eval."""
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.store[name])


# ---------------------------------------------------------------------------
# LPIPS wrapper (handles 1→3 channel replication)
# ---------------------------------------------------------------------------
def _try_load_lpips(net="alex"):
    try:
        import lpips
        return lpips.LPIPS(net=net).eval()
    except ImportError:
        print("[WARN] lpips not installed — perceptual loss disabled.  pip install lpips")
        return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 1: VAE training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--z_dim", type=int, default=4)
    parser.add_argument("--lambda_perc", type=float, default=1.0)
    parser.add_argument("--lpips_every", type=int, default=0,
                        help="Compute LPIPS every N batches (0 = every batch)")
    parser.add_argument("--lambda_adv", type=float, default=0.5)
    parser.add_argument("--lambda_kl", type=float, default=5e-4)
    parser.add_argument("--kl_warmup_epochs", type=int, default=10)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--log_dir", type=str, default="runs/vae")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  |  AMP: {use_amp}  |  Logs: {args.log_dir}")

    # ---- LPIPS ----
    lpips_fn = _try_load_lpips()
    if lpips_fn is not None:
        lpips_fn = lpips_fn.to(device)
        for p in lpips_fn.parameters():
            p.requires_grad = False

    # ---- Data ----
    train_ds = VisDataset(split="train", data_dir=args.data_dir)
    val_ds = VisDataset(split="val", data_dir=args.data_dir)
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Models ----
    vae = VAE(in_channels=1, out_channels=1, z_dim=args.z_dim).to(device)
    disc = PatchGANDiscriminator(in_channels=1).to(device)
    if device.type == "cuda":
        vae = torch.compile(vae, mode="default")
        disc = torch.compile(disc, mode="default")

    opt_g = torch.optim.Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.9))

    ema = EMA(vae, decay=args.ema_decay)
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["vae"])
        disc.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        ema.shadow = ckpt["ema_shadow"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # ---- Loop ----
    for epoch in range(start_epoch, args.epochs):
        # KL warmup: ramp linearly from 0 → λ_kl over warmup epochs
        kl_weight = args.lambda_kl * min(1.0, (epoch + 1) / max(1, args.kl_warmup_epochs))

        vae.train()
        disc.train()
        g_loss_sum = d_loss_sum = rec_sum = kl_sum = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", unit="batch")
        for step, x in enumerate(pbar):
            x = x.to(device)

            # ============================================================
            # Discriminator
            # ============================================================
            _requires_grad(disc, True)
            opt_d.zero_grad()

            with torch.autocast(device_type="cuda", enabled=use_amp):
                with torch.no_grad():
                    recon, _, _ = vae(x)
                real_pred = disc(x)
                fake_pred = disc(recon.detach())

            d_loss = hinge_loss_d(real_pred, fake_pred)
            if scaler:
                scaler.scale(d_loss).backward()
            else:
                d_loss.backward()
            d_loss_sum += d_loss.item()

            # ============================================================
            # Generator (VAE)
            # ============================================================
            _requires_grad(disc, False)
            opt_g.zero_grad()

            with torch.autocast(device_type="cuda", enabled=use_amp):
                recon, mu, logvar = vae(x)

            l1 = F.l1_loss(recon, x)
            if lpips_fn is not None and (
                args.lpips_every <= 0 or (step + 1) % args.lpips_every == 0
            ):
                perc = lpips_fn(recon.repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1)).mean()
            else:
                perc = torch.tensor(0.0, device=device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                adv = hinge_loss_g(disc(recon))
            kld = kl_loss(mu, logvar)

            g_loss = l1 + args.lambda_perc * perc + args.lambda_adv * adv + kl_weight * kld

            if scaler:
                scaler.scale(g_loss).backward()
            else:
                g_loss.backward()
            g_loss_sum += g_loss.item()
            rec_sum += l1.item()
            kl_sum += kld.item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(opt_d)
                    scaler.unscale_(opt_g)
                    nn.utils.clip_grad_norm_(disc.parameters(), args.clip_grad)
                    nn.utils.clip_grad_norm_(vae.parameters(), args.clip_grad)
                    scaler.step(opt_d)
                    scaler.step(opt_g)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(disc.parameters(), args.clip_grad)
                    nn.utils.clip_grad_norm_(vae.parameters(), args.clip_grad)
                    opt_d.step()
                    opt_g.step()
                opt_d.zero_grad()
                opt_g.zero_grad()
                ema.update(vae)

            batches += 1
            pbar.set_postfix(G=f"{g_loss_sum / batches:.3f}", D=f"{d_loss_sum / batches:.3f}",
                             L1=f"{rec_sum / batches:.3f}", KL=f"{kl_sum / batches:.3f}")

        n = max(batches, 1)
        writer.add_scalar("train/G_loss", g_loss_sum / n, epoch)
        writer.add_scalar("train/D_loss", d_loss_sum / n, epoch)
        writer.add_scalar("train/L1", rec_sum / n, epoch)
        writer.add_scalar("train/KL", kl_sum / n, epoch)

        # ---- Validation ----
        if (epoch + 1) % args.val_every == 0:
            ema.apply(vae)
            vae.eval()
            val_l1 = val_kl = 0.0
            with torch.no_grad():
                for x_val in tqdm(val_loader, desc="  Val", unit="batch", leave=False):
                    x_val = x_val.to(device)
                    recon, mu, logvar = vae(x_val)
                    val_l1 += F.l1_loss(recon, x_val).item()
                    val_kl += kl_loss(mu, logvar).item()
            val_l1_avg = val_l1 / len(val_loader)
            val_kl_avg = val_kl / len(val_loader)
            print(f"  Val L1={val_l1_avg:.4f}  KL={val_kl_avg:.6f}")
            writer.add_scalar("val/L1", val_l1_avg, epoch)
            writer.add_scalar("val/KL", val_kl_avg, epoch)
            ema.restore(vae)
            vae.train()

        # ---- Checkpoint ----
        if (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.out_dir, f"vae_stage1_e{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "vae": vae.state_dict(),
                    "disc": disc.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "ema_shadow": ema.shadow,
                },
                path,
            )
            tqdm.write(f"  Saved → {path}")

    writer.close()
    print("Stage 1 (VAE) complete.")


if __name__ == "__main__":
    main()
