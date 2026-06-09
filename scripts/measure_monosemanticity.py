# -*- coding: utf-8 -*-
"""Measure SAE dictionary monosemanticity (class / spatial / coverage).

Streaming version: reads mmap'd patches.npy in chunks.

Usage:
    python scripts/measure_monosemanticity.py \
        --acts_dir data/activations/dinov2-base__L-2__train \
        --sae outputs/sae_ckpts/sae__dinov2-base__L-2__train__d12288__k32__s30000.pt
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sae.topk_sae import TopKSAE
from src.eval.monosemanticity import (
    class_ms_score, spatial_ms_score, coverage, compose_ms,
)


def load_sae(path: str, device: str = "cuda") -> TopKSAE:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(d_model=cfg["d_model"], dict_size=cfg["dict_size"], k=cfg["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def build_balanced_mask(labels, seed=0, cap=None):
    """Boolean mask (len Ntot) selecting a CLASS-BALANCED subset.

    To get a clean, defensible balance under multi-label data, we keep only
    single-fault wafers and cap each class to the same count (default = the
    minimum single-fault class count). Result: every class has exactly `cap`
    single-label examples — matching CIFAR-10's balanced design, so any
    remaining class_ms gap is not explained by class imbalance.
    """
    rng = np.random.default_rng(seed)
    n_faults = labels.sum(axis=1)
    single = n_faults == 1
    n_cls = labels.shape[1]
    per_cls_single = [np.where(single & (labels[:, c] == 1))[0] for c in range(n_cls)]
    if cap is None:
        cap = min(len(idx) for idx in per_cls_single)
    mask = np.zeros(len(labels), dtype=bool)
    for idx in per_cls_single:
        chosen = rng.permutation(idx)[:cap]
        mask[chosen] = True
    return mask, cap


def streaming_stats(sae, acts_dir, chunk_images=512, device="cuda",
                    keep_mask=None):
    """Streaming per-feature statistics over a mmap'd activation directory.

    keep_mask: optional boolean array (len Ntot); if given, only those images
    are used (for the class-balanced control).

    Returns:
        per_class_sum, n_per_class, spatial_sum, coverage_num, coverage_den, Ntot, N
    """
    with open(os.path.join(acts_dir, "meta.json")) as f:
        meta = json.load(f)
    Ntot, N, D = meta["Ntot"], meta["N_patches"], meta["D"]
    dict_size = sae.dict_size

    patches = np.load(os.path.join(acts_dir, "patches.npy"), mmap_mode="r")
    labels = np.load(os.path.join(acts_dir, "labels.npy"))
    n_cls = labels.shape[1]

    if keep_mask is not None:
        n_per_class = labels[keep_mask].sum(axis=0).astype(np.int64)
    else:
        n_per_class = labels.sum(axis=0).astype(np.int64)

    per_class_sum = np.zeros((dict_size, n_cls), dtype=np.float64)
    spatial_sum = np.zeros((dict_size, N), dtype=np.float64)
    coverage_num = np.zeros((dict_size,), dtype=np.int64)
    coverage_den = 0

    for start in range(0, Ntot, chunk_images):
        end = min(start + chunk_images, Ntot)
        y = labels[start:end]
        if keep_mask is not None:
            sub = keep_mask[start:end]
            if not sub.any():
                continue
            rows = np.where(sub)[0]
            x = torch.from_numpy(np.asarray(patches[start:end])[rows]).to(device)
            y = y[rows]
        else:
            x = torch.from_numpy(np.asarray(patches[start:end])).to(device)
        B = x.shape[0]
        flat = x.reshape(B * N, D)
        with torch.inference_mode():
            codes = sae.encode(flat)  # (B*N, dict_size)
        codes_np = codes.cpu().numpy()
        codes_bnf = codes_np.reshape(B, N, dict_size)

        # per-image mean across patches (B, dict_size)
        per_image_act = codes_bnf.mean(axis=1)
        per_class_sum += per_image_act.T.astype(np.float64) @ y.astype(np.float64)

        spatial_sum += codes_bnf.sum(axis=0).T.astype(np.float64)
        coverage_num += (codes_np > 0).sum(axis=0).astype(np.int64)
        coverage_den += codes_np.shape[0]

        if (start // chunk_images) % 5 == 0:
            print(f"  processed {end}/{Ntot}")

    return {
        "per_class_sum": per_class_sum,
        "n_per_class": n_per_class,
        "spatial_sum": spatial_sum,
        "coverage_num": coverage_num,
        "coverage_den": coverage_den,
        "Ntot": Ntot,
        "N": N,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk_images", type=int, default=512)
    ap.add_argument("--balance_classes", action="store_true",
                    help="class-balanced control: single-fault wafers capped "
                         "to equal per-class count (confound check vs CIFAR)")
    ap.add_argument("--balance_cap", type=int, default=None,
                    help="per-class cap (default = min single-fault count)")
    ap.add_argument("--balance_seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[ms] acts_dir={args.acts_dir}")
    print(f"[ms] sae    ={args.sae}")

    sae = load_sae(args.sae, args.device)
    print(f"[ms] SAE loaded: dict={sae.dict_size}  k={sae.k}")

    keep_mask = None
    if args.balance_classes:
        labels = np.load(os.path.join(args.acts_dir, "labels.npy"))
        keep_mask, cap = build_balanced_mask(labels, seed=args.balance_seed,
                                             cap=args.balance_cap)
        print(f"[ms] CLASS-BALANCED control: {int(keep_mask.sum())} wafers "
              f"({cap} single-fault per class, {labels.shape[1]} classes)")

    print(f"[ms] computing per-feature statistics (streaming)...")
    stats = streaming_stats(sae, args.acts_dir,
                            chunk_images=args.chunk_images, device=args.device,
                            keep_mask=keep_mask)

    cls_ms = class_ms_score(stats["per_class_sum"], stats["n_per_class"])
    spa_ms = spatial_ms_score(stats["spatial_sum"], stats["Ntot"])
    cov = coverage(stats["coverage_num"], stats["coverage_den"])
    combined = compose_ms(cls_ms, spa_ms)

    n_dead = int((cov == 0).sum())
    alive = cov > 0

    print(f"[ms] feature counts: total={len(cov)}  dead={n_dead}  alive={alive.sum()}")
    print(f"[ms] class_ms  (alive): median={np.median(cls_ms[alive]):.3f}  "
          f"p10={np.percentile(cls_ms[alive], 10):.3f}  "
          f"p90={np.percentile(cls_ms[alive], 90):.3f}")
    print(f"[ms] spatial_ms(alive): median={np.median(spa_ms[alive]):.3f}  "
          f"p10={np.percentile(spa_ms[alive], 10):.3f}  "
          f"p90={np.percentile(spa_ms[alive], 90):.3f}")
    print(f"[ms] combined  (alive): median={np.median(combined[alive]):.3f}  "
          f"p90={np.percentile(combined[alive], 90):.3f}  "
          f"frac>0.5={(combined[alive] > 0.5).mean():.3f}")

    suffix = "__balanced" if args.balance_classes else ""
    tag = os.path.basename(args.sae).replace(".pt", "") + suffix
    out_npz = os.path.join(args.out_dir, "tables", f"ms__{tag}.npz")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, class_ms=cls_ms, spatial_ms=spa_ms,
                        coverage=cov, combined_ms=combined,
                        per_class_sum=stats["per_class_sum"],
                        n_per_class=stats["n_per_class"])
    print(f"[ms] saved {out_npz}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0,0].hist(cls_ms[alive], bins=40, color="#1E2761", edgecolor="white")
    axes[0,0].set_title("class_ms (alive)")
    axes[0,0].set_xlabel("score (1 = class-monosemantic)")
    axes[0,0].set_ylabel("# features")
    axes[0,1].hist(spa_ms[alive], bins=40, color="#1E2761", edgecolor="white")
    axes[0,1].set_title("spatial_ms (alive)")
    axes[0,1].set_xlabel("score (1 = spatially-localized)")
    axes[1,0].hist(combined[alive], bins=40, color="#E4572E", edgecolor="white")
    axes[1,0].set_title("combined_ms = class × spatial")
    axes[1,0].set_xlabel("score")
    axes[1,0].set_ylabel("# features")
    axes[1,1].hist(cov[alive], bins=40, color="#1E2761", edgecolor="white", log=True)
    axes[1,1].set_title(f"coverage (alive)  |  dead={n_dead}/{len(cov)}")
    axes[1,1].set_xlabel("fraction of patches where feature fires")
    axes[1,1].set_yscale("log")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "figures", f"ms__{tag}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"[ms] saved figure {fig_path}")


if __name__ == "__main__":
    main()
