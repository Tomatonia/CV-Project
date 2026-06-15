import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.vae import VAE

def main():
    parser = argparse.ArgumentParser(description="Visualizing VAE encoder results")
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--image", type=str, default="h8_vis_201030-0500.npy") # 2019 Goni
    parser.add_argument("--output", type=str, default="results/vis_recon")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vis_base_path = "/root/autodl-tmp/data/vis/"
    img_path = os.path.join(vis_base_path, args.image)
    basename = os.path.splitext(os.path.basename(img_path))[0]  # "h8_vis_201030-0500"
    satellite = basename.split('_')[0]
    timestamp = basename.split('_')[-1]

    img = np.load(img_path)
    plt.imsave(
        os.path.join(args.output, f"{satellite}_vis-truth_{timestamp}.jpg"),
        img, cmap="gray", vmin=0, vmax=255,
    )

    img = (img.astype(np.float32) / 127.5) - 1.0  # normalize to [-1, 1]
    img_t = torch.from_numpy(img).to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]

    ckpt_path = f"checkpoints/vae_stage1_e{(args.checkpoint - 1):03d}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    vae = VAE(in_channels=1, out_channels=1, z_dim=4).to(device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt["vae"].items()}
    vae.load_state_dict(clean_state_dict)
    vae.eval()

    with torch.no_grad():
        recon, _, _ = vae(img_t)
    output = recon.squeeze().detach().cpu().numpy()          # [512, 512]
    output = ((output + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    plt.imsave(
        os.path.join(args.output, f"epoch_{args.checkpoint}_{satellite}_vis-recon_{timestamp}.jpg"),
        output, cmap="gray", vmin=0, vmax=255,
    )


if __name__ == "__main__":
    main()