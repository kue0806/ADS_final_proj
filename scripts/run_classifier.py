# -*- coding: utf-8 -*-
"""Interpretability vs. accuracy: multi-label probe on raw vs. SAE features.

Compares multi-label probe performance using:
  raw  = mean patch-token features
  sae  = mean SAE dictionary codes

Trains on the train split's features, evaluates on a held-out val split that we
build from train activations (90/10 internal split, since we only extracted the
train split). Reports F1/mAP and the raw-vs-sae gap.

Also supports K-shot probing (per-class label budget).

Usage:
  python scripts/run_classifier.py \
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

from src.sae.topk_sae import TopKSAE
from src.eval.classify import build_image_features, train_eval_probe, multilabel_metrics


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
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk_images", type=int, default=512)
    args = ap.parse_args()

    sae = load_sae(args.sae, args.device)
    print(f"[clf] SAE dict={sae.dict_size} k={sae.k}", flush=True)

    print("[clf] building RAW image features...", flush=True)
    Xraw, Y = build_image_features(sae, args.acts_dir, device=args.device,
                                   chunk_images=args.chunk_images, use_sae=False)
    print(f"[clf] Xraw {Xraw.shape}", flush=True)

    print("[clf] building SAE image features...", flush=True)
    Xsae, _ = build_image_features(sae, args.acts_dir, device=args.device,
                                   chunk_images=args.chunk_images, use_sae=True)
    print(f"[clf] Xsae {Xsae.shape}", flush=True)

    # Internal train/val split
    rng = np.random.default_rng(args.seed)
    N = Xraw.shape[0]
    perm = rng.permutation(N)
    n_val = int(N * args.val_frac)
    va_idx, tr_idx = perm[:n_val], perm[n_val:]

    results = {}
    for name, X in [("raw", Xraw), ("sae", Xsae)]:
        prob = train_eval_probe(
            X[tr_idx], Y[tr_idx], X[va_idx], Y[va_idx],
            epochs=args.epochs, device=args.device, seed=args.seed,
        )
        m = multilabel_metrics(prob, Y[va_idx])
        results[name] = m
        print(f"\n[clf] === {name} probe (val) ===", flush=True)
        for k, v in m.items():
            print(f"  {k:<14} {v:.4f}", flush=True)

    # Trade-off
    print(f"\n[clf] === interpretability vs. accuracy ===", flush=True)
    for metric in ["micro_f1", "macro_f1", "mAP", "subset_acc"]:
        gap = results["raw"][metric] - results["sae"][metric]
        print(f"  {metric:<12}  raw={results['raw'][metric]:.4f}  "
              f"sae={results['sae'][metric]:.4f}  gap(raw-sae)={gap:+.4f}", flush=True)

    micro_gap = results["raw"]["micro_f1"] - results["sae"]["micro_f1"]
    verdict = "SUPPORTED" if abs(micro_gap) <= 0.02 else "NOT SUPPORTED"
    print(f"\n[clf] retention (raw vs. SAE micro-F1): {verdict}  "
          f"(gap={micro_gap*100:+.2f} pp)", flush=True)

    # Save
    tag = os.path.basename(args.sae).replace(".pt", "")
    out = os.path.join(args.out_dir, "tables", f"classify__{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"results": results,
                   "micro_f1_gap": micro_gap,
                   "raw_dim": int(Xraw.shape[1]),
                   "sae_dim": int(Xsae.shape[1])}, f, indent=2)
    print(f"[clf] saved {out}", flush=True)


if __name__ == "__main__":
    main()
