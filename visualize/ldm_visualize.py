import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from models.vae import VAE
from models.ir_encoder import LightweightIREncoder
from models.diffusion import GaussianDiffusion
from models.unet import LDMUNet


def _clean(state_dict):
    """Remove '_orig_mod.' prefix added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

def _normalise_angles(arr):
    """Normalise (4, H, W) float32 degree angles → [-1, 1].

    Channel 0 — solar zenith   [0, 90]   → [-1, 1]
    Channel 1 — solar azimuth   [0, 360] → [-180, 180] → [-1, 1]
    Channel 2 — satellite zenith [~0, ~90] → [-1, 1]
    Channel 3 — satellite azimuth [0, 360] → [-180, 180] → [-1, 1]
    """
    out = arr.astype(np.float32).copy()
    # Zenith channels (0, 2):  scale [0, 90] → [-1, 1]
    out[0] = 2.0 * out[0] / 90.0 - 1.0
    out[2] = 2.0 * out[2] / 90.0 - 1.0
    # Azimuth channels (1, 3):  wrap to [-180, 180] then scale to [-1, 1]
    # [0, 360] to [-1, 1] directly
    out[1] = out[1] / 180.0 - 1.0
    out[3] = out[3] / 180.0 - 1.0
    return np.clip(out, -1.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description="Visualizing LDM results")
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to Stage-1 VAE checkpoint")
    parser.add_argument("--ldm_ckpt", type=str, required=True,
                        help="Path to Stage-2 LDM checkpoint")
    parser.add_argument("--ir_image", type=str, default="h8_ir_201030-0500.npy") # 2019 Goni
    parser.add_argument("--vis_gt", action="store_true",
                        help="Also save VIS ground truth for comparison")
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--latent_scale", type=float, default=0.25)
    parser.add_argument("--output", type=str, default="results/ldm")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vis_base_path = "/root/autodl-tmp/data/vis/"
    ir_base_path = "/root/autodl-tmp/data/ir/"
    angles_base_path = "/root/autodl-tmp/data/angles/"

    # Parse filename: "h8_ir_201030-0500.npy" → sat=h8, ts=201030-0500
    basename = os.path.splitext(os.path.basename(args.ir_image))[0]
    satellite = basename.split("_")[0]
    timestamp = basename.split("_")[-1]

    vis_img_path = os.path.join(vis_base_path, f"{satellite}_vis_{timestamp}.npy")
    ir_img_path = os.path.join(ir_base_path, f"{satellite}_ir_{timestamp}.npy")
    angles_img_path = os.path.join(angles_base_path, f"{satellite}_angles_{timestamp}.npy")

    # ---- Load frozen VAE ----
    ckpt_vae = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae = VAE(in_channels=1, out_channels=1, z_dim=4).to(device)
    vae.load_state_dict(_clean(ckpt_vae["vae"]))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ---- Load LDM checkpoint (EMA weights already baked in) ----
    ckpt = torch.load(args.ldm_ckpt, map_location=device, weights_only=True)

    ir_encoder = LightweightIREncoder(in_channels=3, out_channels=4, ch=64).to(device)
    ir_encoder.load_state_dict(_clean(ckpt["ir_encoder"]))
    ir_encoder.eval()

    unet = LDMUNet(in_channels=12, out_channels=4, ch=128, ch_mult=(1, 2, 2, 2)).to(device)
    unet.load_state_dict(_clean(ckpt["unet"]))
    unet.eval()

    diff = GaussianDiffusion(T=args.T, schedule="linear")

    # ---- Load inputs ----
    ir_img = np.load(ir_img_path)
    ir_img = (ir_img.astype(np.float32) / 127.5) - 1.0
    ir_img_t = torch.from_numpy(ir_img).to(device).unsqueeze(0)  # [1, 3, 512, 512]

    angles = np.load(angles_img_path)
    angles = _normalise_angles(angles)
    angles = torch.from_numpy(angles).to(device).unsqueeze(0)    # [1, 4, 512, 512]

    # ---- Forward pass ----
    with torch.no_grad():
        f_ir = ir_encoder(ir_img_t)
        angles_128 = F.interpolate(angles, size=(128, 128), mode="bilinear")
        cond = torch.cat([f_ir, angles_128], dim=1)              # (1, 8, 128, 128)
        z_0 = diff.ddim_sample_loop(unet, (1, 4, 128, 128), cond, steps=args.ddim_steps, eta=args.ddim_eta)
        z_0 = z_0 / args.latent_scale
        recon = vae.decode(z_0)

    output = recon.squeeze().detach().cpu().numpy()
    output = ((output + 1.0) * 127.5).clip(0, 255) # .astype(np.uint8)
    plt.imsave(
        os.path.join(args.output, f"{satellite}_ldm_{timestamp}.jpg"),
        output, cmap="gray", vmin=0, vmax=255,
    )
    print(f"Saved → {args.output}/{satellite}_ldm_{timestamp}.jpg")

    # ---- Optional: VIS ground truth ----
    if args.vis_gt:
        vis_gt = np.load(vis_img_path)
        plt.imsave(
            os.path.join(args.output, f"{satellite}_vis-truth_{timestamp}.jpg"),
            vis_gt, cmap="gray", vmin=0, vmax=255,
        )


if __name__ == "__main__":
    main()
