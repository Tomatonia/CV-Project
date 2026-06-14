"""Dataset classes for Himawari-8/9 preprocessed .npy files.

Data layout (from utils.py):
  data/vis/     {sat}_vis_{yymmdd-HHMM}.npy      uint8   (512, 512)
  data/ir/      {sat}_ir_{yymmdd-HHMM}.npy       uint8   (3, 512, 512)
  data/angles/  {sat}_angles_{yymmdd-HHMM}.npy   float32 (4, 512, 512)

The 4 angle channels are: solar zenith, solar azimuth, satellite zenith,
satellite azimuth — all in degrees (raw, no normalisation in the .npy files).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _find_files(data_dir, subdir):
    """Return sorted list of .npy paths in <data_dir>/<subdir>/."""
    d = os.path.join(data_dir, subdir)
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.endswith(".npy")
    )


def _parse_key(path):
    """Extract matching key from a path.

    'h8_vis_200101-0000.npy'    → 'h8_200101-0000'
    'h9_ir_200101-0000.npy'     → 'h9_200101-0000'
    'h8_angles_200101-0000.npy' → 'h8_200101-0000'
    """
    base = os.path.splitext(os.path.basename(path))[0]
    for tag in ("_vis_", "_ir_", "_angles_"):
        if tag in base:
            sat, ts = base.split(tag, 1)
            return f"{sat}_{ts}"
    return base


def _normalise_uint8(arr):
    """uint8 [0, 255] → float32 [-1, 1]."""
    return (arr.astype(np.float32) / 127.5) - 1.0


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
    out[1] = np.where(out[1] > 180.0, out[1] - 360.0, out[1]) / 180.0
    out[3] = np.where(out[3] > 180.0, out[3] - 360.0, out[3]) / 180.0
    return np.clip(out, -1.0, 1.0)


class VisDataset(Dataset):
    """Single-channel B03 visible images for VAE training."""

    def __init__(self, split="train", split_ratio=0.9, data_dir="data"):
        paths = _find_files(data_dir, "vis")
        if not paths:
            raise FileNotFoundError(f"No .npy files in {data_dir}/vis/")
        n = len(paths)
        cutoff = int(n * split_ratio)
        self.paths = paths[:cutoff] if split == "train" else paths[cutoff:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.load(self.paths[idx])                          # (512, 512) uint8
        img = _normalise_uint8(img)
        return torch.from_numpy(img).unsqueeze(0)               # (1, 512, 512)


class IRDataset(Dataset):
    """3-channel IR stacks (B11, B13, B15) for IR encoder training."""

    def __init__(self, split="train", split_ratio=0.9, data_dir="data"):
        paths = _find_files(data_dir, "ir")
        if not paths:
            raise FileNotFoundError(f"No .npy files in {data_dir}/ir/")
        n = len(paths)
        cutoff = int(n * split_ratio)
        self.paths = paths[:cutoff] if split == "train" else paths[cutoff:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        stack = np.load(self.paths[idx])                        # (3, 512, 512) uint8
        stack = _normalise_uint8(stack)
        return torch.from_numpy(stack)                          # (3, 512, 512)


class PairedDataset(Dataset):
    """
    Time-matched IR + angles → VIS triples for LDM training.

    Returns (ir, angles, vis):
      ir      — (3, 512, 512)  B11, B13, B15  [-1, 1]
      angles  — (4, 512, 512)  sol_zen, sol_az, sat_zen, sat_az  [-1, 1]
      vis     — (1, 512, 512)  B03  [-1, 1]

    Only includes timestamps present in ALL THREE directories.
    Angles bypass the IR encoder and are concatenated directly in the
    latent space (resized to 128×128) alongside z_ir and z_t.
    """

    def __init__(self, split="train", split_ratio=0.9, data_dir="data"):
        vis_paths = _find_files(data_dir, "vis")
        ir_paths = _find_files(data_dir, "ir")
        angle_paths = _find_files(data_dir, "angles")

        missing = []
        if not vis_paths:
            missing.append("vis")
        if not ir_paths:
            missing.append("ir")
        if not angle_paths:
            missing.append("angles")
        if missing:
            raise FileNotFoundError(f"Missing .npy files in: {', '.join(missing)}")

        vis_keys = {_parse_key(p): p for p in vis_paths}
        ir_keys = {_parse_key(p): p for p in ir_paths}
        angle_keys = {_parse_key(p): p for p in angle_paths}
        common = sorted(set(vis_keys) & set(ir_keys) & set(angle_keys))
        if not common:
            raise RuntimeError("No matching VIS/IR/angles triples found")

        n = len(common)
        cutoff = int(n * split_ratio)
        keys = common[:cutoff] if split == "train" else common[cutoff:]
        self.triples = [(vis_keys[k], ir_keys[k], angle_keys[k]) for k in keys]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        vis_path, ir_path, angle_path = self.triples[idx]
        vis = _normalise_uint8(np.load(vis_path))
        ir = _normalise_uint8(np.load(ir_path))
        angles = _normalise_angles(np.load(angle_path))
        return (
            torch.from_numpy(ir),                               # (3, 512, 512)
            torch.from_numpy(angles),                           # (4, 512, 512)
            torch.from_numpy(vis).unsqueeze(0),                 # (1, 512, 512)
        )
