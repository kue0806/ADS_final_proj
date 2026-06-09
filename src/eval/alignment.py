# -*- coding: utf-8 -*-
"""Per-wafer alignment between SAE features and defect masks.

Compactness is computed per wafer then averaged per defect type; alignment is
the per-wafer cosine between a feature's activation map and the defect mask.
"""
from __future__ import annotations
import json
import math
import os
import numpy as np
import torch

from src.data.wbm_loader import resize_native_to_grid


def downsample_mean(arr_52: np.ndarray, target: int) -> np.ndarray:
    """Map a (52,52) float map to (target,target) faithfully.

    Uses a bilinear downsample matching the 52->224 stretch + patchify of the
    forward path, instead of zero-padded block pooling (which biased the
    bottom-right ~1/3 of patches toward blank)."""
    return resize_native_to_grid(arr_52, target)


def per_wafer_compactness(mask_grid: np.ndarray) -> float:
    """Compactness of a single wafer's defect mask: 1 - H(p)/log(N),
    where p = mask/sum(mask). 1 = single bin, 0 = uniform."""
    flat = mask_grid.reshape(-1).astype(np.float64)
    s = flat.sum()
    if s <= 0:
        return 0.0
    p = flat / s
    eps = 1e-12
    h = -(p * np.log(p + eps)).sum()
    return float(1.0 - h / np.log(len(flat)))


def streaming_per_wafer_alignment(
    sae,
    acts_dir: str,
    raw_npz: str,
    grid: int = 16,
    chunk_images: int = 128,
    device: str = "cuda",
    single_fault_only: bool = False,
):
    """Single-pass streaming computation.

    single_fault_only: if True, use only wafers with exactly one defect type
    (removes mixed-label cross-contamination).

    Returns:
        align_sum_per_dim     (n_dim, dict_size)  — sum of per-wafer cosines
                                                    over wafers carrying dim c
        n_per_dim             (n_dim,)           — # wafers carrying dim c
        compact_per_wafer     (Ntot,)            — per-wafer compactness
        compact_sum_per_dim   (n_dim,)           — sum of compactness over
                                                    wafers carrying dim c
        mean_mask_per_dim     (n_dim, grid, grid) — averaged mask (for viz)
        Ntot, grid, dict_size
    """
    with open(os.path.join(acts_dir, "meta.json")) as f:
        meta = json.load(f)
    Ntot, N_patches, D = meta["Ntot"], meta["N_patches"], meta["D"]
    # Auto-detect grid from N_patches; override the grid argument if needed.
    auto_grid = int(round(math.sqrt(N_patches)))
    if auto_grid * auto_grid != N_patches:
        raise ValueError(f"N_patches={N_patches} not a perfect square")
    if auto_grid != grid:
        print(f"[align] adjusting grid from {grid} to {auto_grid} based on N_patches={N_patches}",
              flush=True)
        grid = auto_grid

    patches = np.load(os.path.join(acts_dir, "patches.npy"), mmap_mode="r")
    labels = np.load(os.path.join(acts_dir, "labels.npy"))
    indices = np.load(os.path.join(acts_dir, "indices.npy"))
    raw = np.load(raw_npz, allow_pickle=True)
    arr0 = raw["arr_0"]

    dict_size = sae.dict_size
    n_dim = labels.shape[1]

    align_sum_per_dim = np.zeros((n_dim, dict_size), dtype=np.float64)
    # Counted over wafers ACTUALLY processed (so single_fault_only uses the
    # correct, smaller denominator per dim — not the full multi-fault count).
    n_per_dim = np.zeros(n_dim, dtype=np.int64)
    compact_per_wafer = np.zeros(Ntot, dtype=np.float32)
    compact_sum_per_dim = np.zeros(n_dim, dtype=np.float64)
    mean_mask_sum = np.zeros((n_dim, grid, grid), dtype=np.float64)

    eps = 1e-12

    for start in range(0, Ntot, chunk_images):
        end = min(start + chunk_images, Ntot)
        y = labels[start:end]                                             # (B,8)
        ix = indices[start:end]

        if single_fault_only:
            keep = y.sum(axis=1) == 1
            if not keep.any():
                continue
            sel_rows = np.where(keep)[0]
            x = torch.from_numpy(
                np.asarray(patches[start:end])[sel_rows]).to(device)
            y = y[sel_rows]
            ix = ix[sel_rows]
            positions = start + sel_rows          # global indices of kept wafers
        else:
            x = torch.from_numpy(np.asarray(patches[start:end])).to(device)
            positions = np.arange(start, end)
        B = x.shape[0]

        # Defect masks per wafer (B, grid, grid) at downsample to grid
        masks = np.zeros((B, grid, grid), dtype=np.float32)
        for j in range(B):
            masks[j] = downsample_mean((arr0[ix[j]] == 2).astype(np.float32), grid)

        # Per-wafer compactness (write to each wafer's global position)
        compact_b = np.zeros(B, dtype=np.float32)
        for j in range(B):
            compact_b[j] = per_wafer_compactness(masks[j])
        compact_per_wafer[positions] = compact_b

        # Aggregate compactness per dim (and count processed wafers per dim)
        for c in range(n_dim):
            sel = y[:, c] > 0
            if sel.any():
                n_per_dim[c] += int(sel.sum())
                compact_sum_per_dim[c] += compact_b[sel].sum()
                mean_mask_sum[c] += masks[sel].sum(axis=0).astype(np.float64)

        # SAE codes (B, N, dict_size) on GPU
        flat = x.reshape(B * N_patches, D)
        with torch.inference_mode():
            codes = sae.encode(flat).view(B, N_patches, dict_size)        # (B,N,F)

        # Per-wafer cosine between each feature's spatial map and wafer mask
        masks_gpu = torch.from_numpy(masks.reshape(B, N_patches)).to(device).float()
        mask_norm = masks_gpu.norm(dim=1)                                  # (B,)
        act_norm = codes.norm(dim=1)                                       # (B, F)
        # dot: (B, F) = sum over N of codes[b, n, f] * masks[b, n]
        dot = torch.einsum("bnf,bn->bf", codes, masks_gpu)                 # (B, F)
        cos_bf = dot / (act_norm * mask_norm.unsqueeze(1) + eps)           # (B, F)
        cos_bf = cos_bf.cpu().numpy().astype(np.float64)

        # Per-dim accumulation: for each c, add cos rows of wafers with c=1
        for c in range(n_dim):
            sel = y[:, c] > 0
            if sel.any():
                align_sum_per_dim[c] += cos_bf[sel].sum(axis=0)

        if (start // chunk_images) % 10 == 0:
            print(f"  processed {end}/{Ntot}", flush=True)

    return {
        "align_sum_per_dim": align_sum_per_dim,
        "n_per_dim": n_per_dim,
        "compact_per_wafer": compact_per_wafer,
        "compact_sum_per_dim": compact_sum_per_dim,
        "mean_mask_per_dim": np.array(
            [mean_mask_sum[c] / max(int(n_per_dim[c]), 1) for c in range(n_dim)]
        ),
        "Ntot": Ntot,
        "grid": grid,
        "dict_size": dict_size,
    }
