# -*- coding: utf-8 -*-
"""Per-wafer alignment between SAE features and defect masks.

Outputs:
  outputs/tables/alignment__<sae>.npz
      align_mean_per_dim:  (n_dim, dict_size)  mean cosine per (defect dim, feature)
      compact_per_dim:     (n_dim,)            mean per-wafer compactness per dim
      compact_per_wafer:   (Ntot,)             per-wafer compactness
      n_per_dim, mean_mask_per_dim, dim_names

  outputs/figures/alignment__<sae>.png   panels:
      Row 0: mean defect masks per dim with per-wafer compactness label
      Row 1: best-aligned feature's mean activation map per dim
      Row 2: per-dim best alignment, colored by mean per-wafer compactness

Usage:
  python scripts/measure_alignment.py \
      --acts_dir data/activations/dinov2-base__L-2__train \
      --sae outputs/sae_ckpts/sae__dinov2-base__L-2__train__d12288__k32__s30000.pt
"""
import argparse
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
from src.eval.alignment import streaming_per_wafer_alignment


DIM_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
             "Loc", "Near-Full", "Random", "Scratch"]


def load_sae(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(d_model=cfg["d_model"], dict_size=cfg["dict_size"], k=cfg["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--raw", default="data/raw/MixedWM38.npz")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk_images", type=int, default=128)
    ap.add_argument("--single_fault_only", action="store_true",
                    help="use only single-defect wafers (F2 refinement)")
    args = ap.parse_args()

    import json, math
    with open(os.path.join(args.acts_dir, "meta.json")) as f:
        _meta = json.load(f)
    _grid = int(round(math.sqrt(_meta["N_patches"])))

    sae = load_sae(args.sae, args.device)
    print(f"[align] SAE dict={sae.dict_size} k={sae.k}  grid={_grid}  "
          f"single_fault_only={args.single_fault_only}", flush=True)
    print(f"[align] per-wafer alignment (streaming)...", flush=True)
    agg = streaming_per_wafer_alignment(
        sae, args.acts_dir, args.raw,
        grid=_grid, chunk_images=args.chunk_images, device=args.device,
        single_fault_only=args.single_fault_only,
    )

    n = np.maximum(agg["n_per_dim"], 1)
    align_mean_per_dim = agg["align_sum_per_dim"] / n[:, None]     # (n_dim, F)
    compact_per_dim = agg["compact_sum_per_dim"] / n               # (n_dim,)
    n_dim = align_mean_per_dim.shape[0]

    per_dim_max = align_mean_per_dim.max(axis=1)
    per_dim_p95 = np.percentile(align_mean_per_dim, 95, axis=1)
    per_dim_top_count = (align_mean_per_dim > 0.3).sum(axis=1)

    print(f"\n[align] === per-defect-dim summary (per-wafer protocol) ===", flush=True)
    print(f"{'dim':<3}  {'name':<10}  {'n':>6}  {'pw_cmp':>7}  "
          f"{'max_align':>9}  {'p95':>6}  {'>.3':>5}", flush=True)
    for c in range(n_dim):
        print(f"{c:<3}  {DIM_NAMES[c]:<10}  {int(agg['n_per_dim'][c]):>6}  "
              f"{compact_per_dim[c]:>7.3f}  {per_dim_max[c]:>9.3f}  "
              f"{per_dim_p95[c]:>6.3f}  {int(per_dim_top_count[c]):>5}", flush=True)

    sorted_c = np.argsort(-compact_per_dim)
    top_clu = sorted_c[:4]
    bot_spa = sorted_c[-4:]
    avg_align_top = float(per_dim_max[top_clu].mean())
    avg_align_bot = float(per_dim_max[bot_spa].mean())
    print(f"\n[align] compactness vs. alignment:", flush=True)
    print(f"  top-4 clustered: {top_clu.tolist()}  "
          f"({[DIM_NAMES[i] for i in top_clu]})  "
          f"cmp={compact_per_dim[top_clu].mean():.3f}  "
          f"max_align={avg_align_top:.3f}", flush=True)
    print(f"  bot-4 sparse   : {bot_spa.tolist()}  "
          f"({[DIM_NAMES[i] for i in bot_spa]})  "
          f"cmp={compact_per_dim[bot_spa].mean():.3f}  "
          f"max_align={avg_align_bot:.3f}", flush=True)
    print(f"  diff (clustered - sparse): {avg_align_top - avg_align_bot:+.3f}",
          flush=True)
    # Correlation between compactness and max_align
    corr = float(np.corrcoef(compact_per_dim, per_dim_max)[0, 1])
    print(f"  Pearson corr (compactness, max_align): {corr:+.3f}", flush=True)

    # Save
    tag = os.path.basename(args.sae).replace(".pt", "")
    if args.single_fault_only:
        tag += "__single"
    out_npz = os.path.join(args.out_dir, "tables", f"alignment__{tag}.npz")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        align_mean_per_dim=align_mean_per_dim.astype(np.float32),
        compact_per_dim=compact_per_dim,
        compact_per_wafer=agg["compact_per_wafer"],
        n_per_dim=agg["n_per_dim"],
        mean_mask_per_dim=agg["mean_mask_per_dim"],
        dim_names=np.array(DIM_NAMES),
    )
    print(f"\n[align] saved {out_npz}", flush=True)

    # Visualize: 3-row figure
    fig, axes = plt.subplots(3, 8, figsize=(16, 7))
    for c in range(n_dim):
        axes[0, c].imshow(agg["mean_mask_per_dim"][c], cmap="hot")
        axes[0, c].set_title(
            f"{DIM_NAMES[c]}\nn={int(agg['n_per_dim'][c])}  "
            f"pw_cmp={compact_per_dim[c]:.2f}", fontsize=9)
        axes[0, c].set_xticks([]); axes[0, c].set_yticks([])

    for c in range(n_dim):
        f_best = int(np.argmax(align_mean_per_dim[c]))
        # Recompute that feature's mean activation map for dim c by another pass
        # (omitted for speed; show the dim summary instead)
        axes[1, c].text(0.5, 0.5,
                        f"feat {f_best}\nalign={align_mean_per_dim[c, f_best]:.3f}",
                        ha="center", va="center", fontsize=10,
                        transform=axes[1, c].transAxes)
        axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
        for spine in axes[1, c].spines.values():
            spine.set_visible(False)

    for c in range(n_dim):
        axes[2, c].axis("off")
    ax_sum = fig.add_subplot(3, 1, 3)
    median_cmp = float(np.median(compact_per_dim))
    bar_colors = ["#1E2761" if compact_per_dim[c] > median_cmp else "#E4572E"
                  for c in range(n_dim)]
    ax_sum.bar(range(n_dim), per_dim_max, color=bar_colors)
    ax_sum.set_xticks(range(n_dim))
    ax_sum.set_xticklabels(
        [f"{DIM_NAMES[c]}\n(cmp={compact_per_dim[c]:.2f})" for c in range(n_dim)],
        fontsize=8)
    ax_sum.set_ylabel("max mean-cosine alignment (over features)")
    ax_sum.set_title(
        "Per-dim best alignment  |  blue=clustered, orange=sparse  "
        f"(corr(cmp, max_align)={corr:+.3f})",
        fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "figures", f"alignment__{tag}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"[align] saved figure {fig_path}", flush=True)


if __name__ == "__main__":
    main()
