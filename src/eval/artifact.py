# -*- coding: utf-8 -*-
"""F5: Artifact-vs-signal localization for SAE features.

Tests whether SAE features fire on patches that contain real wafer DIE signal,
or on patches that are mostly blank / interpolation fill. If features prefer
content patches, the "position > semantics" result reflects genuine spatial
structure; if they prefer blank/edge patches, it reflects upsampling /
patch-tokenization artifacts.

For each wafer we build a patch-grid "content mask": fraction of each patch's
receptive field (in the native 52x52 grid) that is non-blank (value != 0).
We then measure, per feature, the mean content-fraction at the patches where it
fires, and compare to the global mean content-fraction (a chance baseline).

  feature_content_pref[f] = E[content | feature f fires] - E[content]
      > 0  : feature prefers DIE-content patches  (signal)
      < 0  : feature prefers blank/fill patches   (artifact)
"""
from __future__ import annotations
import json
import math
import os
import numpy as np
import torch

from src.data.wbm_loader import resize_native_to_grid


def patch_content_fraction(wafer_52: np.ndarray, grid: int) -> np.ndarray:
    """Fraction of non-blank (value!=0) cells in each patch's receptive field.

    Returns (grid*grid,) in [0,1]. Uses a faithful bilinear downsample of the
    binary 'has die' map (value 1 or 2 -> 1, value 0 -> 0), matching the
    52->224 stretch + patchify of the forward path (no padding bias)."""
    has_die = (wafer_52 != 0).astype(np.float32)
    pooled = resize_native_to_grid(has_die, grid)   # (grid, grid)
    return pooled.reshape(-1)


def streaming_artifact_pref(
    sae,
    acts_dir: str,
    raw_npz: str,
    chunk_images: int = 256,
    device: str = "cuda",
):
    """Single-pass streaming computation of per-feature content preference.

    Returns dict with:
        fire_content_sum  (dict_size,) — sum of content-fraction over patch
                                         positions where feature fired
        fire_count        (dict_size,) — number of (wafer,patch) firings
        global_content_sum  float      — sum of content over all patches
        global_count        int        — total patches
        Ntot, grid
    """
    with open(os.path.join(acts_dir, "meta.json")) as f:
        meta = json.load(f)
    Ntot, N, D = meta["Ntot"], meta["N_patches"], meta["D"]
    grid = int(round(math.sqrt(N)))

    patches = np.load(os.path.join(acts_dir, "patches.npy"), mmap_mode="r")
    indices = np.load(os.path.join(acts_dir, "indices.npy"))
    raw = np.load(raw_npz, allow_pickle=True)
    arr0 = raw["arr_0"]

    dict_size = sae.dict_size
    fire_content_sum = np.zeros(dict_size, dtype=np.float64)
    fire_count = np.zeros(dict_size, dtype=np.int64)
    global_content_sum = 0.0
    global_count = 0

    for start in range(0, Ntot, chunk_images):
        end = min(start + chunk_images, Ntot)
        x = torch.from_numpy(np.asarray(patches[start:end])).to(device)  # (B,N,D)
        ix = indices[start:end]
        B = x.shape[0]

        # Content-fraction per patch per wafer (B, N)
        content = np.zeros((B, N), dtype=np.float32)
        for j in range(B):
            content[j] = patch_content_fraction(arr0[ix[j]], grid)
        global_content_sum += float(content.sum())
        global_count += content.size

        # SAE codes (B, N, dict_size) -> fired mask
        flat = x.reshape(B * N, D)
        with torch.inference_mode():
            codes = sae.encode(flat)                      # (B*N, dict_size)
        fired = (codes > 0).cpu().numpy().astype(np.float32)  # (B*N, dict_size)
        content_flat = content.reshape(B * N).astype(np.float32)  # (B*N,)

        # For each feature, accumulate content where it fired.
        # fire_content_sum[f] += sum_over_patches(content * fired[:,f])
        fire_content_sum += content_flat @ fired           # (dict_size,)
        fire_count += fired.sum(axis=0).astype(np.int64)

        if (start // chunk_images) % 10 == 0:
            print(f"  processed {end}/{Ntot}", flush=True)

    return {
        "fire_content_sum": fire_content_sum,
        "fire_count": fire_count,
        "global_content_sum": global_content_sum,
        "global_count": global_count,
        "Ntot": Ntot,
        "grid": grid,
    }


def content_preference(stats: dict) -> tuple:
    """Per-feature content preference vs global chance baseline.

    Returns (pref, global_mean, feat_mean):
        pref[f] = mean content where f fires - global mean content
    Only features that fired at least once get a finite value (others -> nan).
    """
    fc = stats["fire_count"].astype(np.float64)
    feat_mean = np.where(fc > 0, stats["fire_content_sum"] / np.maximum(fc, 1), np.nan)
    global_mean = stats["global_content_sum"] / max(stats["global_count"], 1)
    pref = feat_mean - global_mean
    return pref, global_mean, feat_mean
