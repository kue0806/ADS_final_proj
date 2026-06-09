# -*- coding: utf-8 -*-
"""F5: Does the SAE encode wafer DIE signal or upsampling/patchization artifact?

Computes per-feature content preference (mean die-content where a feature fires,
minus the global mean content). Aggregates to answer:
  - Do features, on average, prefer content patches (>0) or blank/fill (<0)?
  - Weighted by firing frequency, where does the dictionary's "mass" sit?

Run this on activation dirs extracted with different --upsample settings to
compare bilinear vs nearest, and (optionally) a native-resolution run.

Usage:
  python scripts/measure_artifact.py \
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
from src.eval.artifact import streaming_artifact_pref, content_preference


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
    ap.add_argument("--chunk_images", type=int, default=256)
    args = ap.parse_args()

    sae = load_sae(args.sae, args.device)
    print(f"[artifact] SAE dict={sae.dict_size} k={sae.k}", flush=True)
    print(f"[artifact] computing per-feature content preference (streaming)...",
          flush=True)
    stats = streaming_artifact_pref(sae, args.acts_dir, args.raw,
                                    chunk_images=args.chunk_images,
                                    device=args.device)

    pref, global_mean, feat_mean = content_preference(stats)
    fc = stats["fire_count"].astype(np.float64)
    alive = fc > 0

    # Firing-frequency-weighted mean preference (where the dictionary mass sits)
    w = fc[alive] / fc[alive].sum()
    weighted_pref = float(np.nansum(pref[alive] * w))
    unweighted_pref = float(np.nanmean(pref[alive]))

    # Fraction of (alive) features that prefer content vs artifact
    frac_signal = float((pref[alive] > 0).mean())
    frac_artifact = float((pref[alive] < 0).mean())

    print(f"\n[artifact] === content-preference summary ===", flush=True)
    print(f"  global mean content-fraction (chance baseline): {global_mean:.3f}",
          flush=True)
    print(f"  alive features: {int(alive.sum())}", flush=True)
    print(f"  mean feature content where it fires: "
          f"{np.nanmean(feat_mean[alive]):.3f}", flush=True)
    print(f"  unweighted mean preference (feat - global): {unweighted_pref:+.4f}",
          flush=True)
    print(f"  firing-weighted mean preference:            {weighted_pref:+.4f}",
          flush=True)
    print(f"  fraction of features preferring CONTENT (signal):  {frac_signal:.3f}",
          flush=True)
    print(f"  fraction preferring BLANK/FILL (artifact):         {frac_artifact:.3f}",
          flush=True)
    verdict = ("SIGNAL-dominated" if weighted_pref > 0.02
               else "ARTIFACT-dominated" if weighted_pref < -0.02
               else "MIXED/neutral")
    print(f"\n[artifact] verdict: {verdict}  "
          f"(firing-weighted pref = {weighted_pref:+.4f})", flush=True)

    # Save
    tag = os.path.basename(args.sae).replace(".pt", "")
    out_npz = os.path.join(args.out_dir, "tables", f"artifact__{tag}.npz")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, pref=pref, feat_mean=feat_mean,
                        fire_count=stats["fire_count"],
                        global_mean=global_mean,
                        weighted_pref=weighted_pref,
                        unweighted_pref=unweighted_pref)
    print(f"[artifact] saved {out_npz}", flush=True)

    # Figure: histogram of per-feature preference (firing-weighted)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    vals = pref[alive]
    ax.hist(vals, bins=50, color="#1E2761", edgecolor="white",
            weights=fc[alive])
    ax.axvline(0, color="#E4572E", lw=2, ls="--",
               label="chance (no preference)")
    ax.axvline(weighted_pref, color="black", lw=2,
               label=f"weighted mean = {weighted_pref:+.3f}")
    ax.set_xlabel("content preference  (mean die-content where feature fires "
                  "- global mean)")
    ax.set_ylabel("firing-weighted # features")
    ax.set_title(f"F5: signal vs artifact  ({os.path.basename(args.acts_dir)})\n"
                 f"{verdict}")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "figures", f"artifact__{tag}.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"[artifact] saved figure {fig_path}", flush=True)


if __name__ == "__main__":
    main()
