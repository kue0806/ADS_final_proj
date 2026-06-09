# -*- coding: utf-8 -*-
"""Extract DINO patch-token activations across a MixedWM38 split.

Saves activations as separate uncompressed .npy files inside a per-run dir:

    data/activations/<backbone>__L<layer>__<split>[__N<limit>]/
        patches.npy   (Ntot, N, D)  float32   (mmap-friendly)
        cls.npy       (Ntot, D)     float32
        labels.npy    (Ntot, 8)     int32
        indices.npy   (Ntot,)       int32
        meta.json

This layout supports memory-mapped streaming in downstream training.

Usage:
    python scripts/extract_activations.py --split train --batch 64
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.wbm_loader import load_mixedwm38, make_splits, MixedWM38
from src.models.dino_extractor import DinoExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/MixedWM38.npz")
    ap.add_argument("--backbone", default="facebook/dinov2-base")
    ap.add_argument("--layer", type=int, default=-2)
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--target_hw", type=int, default=224)
    ap.add_argument("--upsample", default="bilinear")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="limit number of samples (smoke testing)")
    ap.add_argument("--out_root", default="data/activations")
    args = ap.parse_args()

    print(f"[extract] backbone={args.backbone} layer={args.layer} split={args.split}")

    arr0, arr1 = load_mixedwm38(args.data)
    splits = make_splits(arr0.shape[0], seed=42)
    idx = splits[args.split]
    if args.limit:
        idx = idx[: args.limit]
    print(f"[extract] processing {len(idx)} samples of split {args.split}")

    ds = MixedWM38(arr0, arr1, indices=idx,
                   target_hw=args.target_hw, upsample=args.upsample)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=0, shuffle=False)

    t0 = time.time()
    extractor = DinoExtractor(model_name=args.backbone, layer=args.layer)
    print(f"[extract] model loaded in {time.time()-t0:.1f}s | "
          f"hidden={extractor.hidden_dim} patch={extractor.patch_size}")

    # Pre-allocate output arrays based on first batch
    first_batch = next(iter(dl))
    out0 = extractor(first_batch[0])
    N_patches = out0["patches"].shape[1]
    D = out0["patches"].shape[2]
    Ntot = len(idx)

    bn = args.backbone.split("/")[-1]
    if args.backbone.startswith("random:"):
        bn = "random-" + bn
    tag = f"{bn}__L{args.layer}__{args.split}"
    if args.limit:
        tag += f"__N{args.limit}"
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Open memory-mapped writers (write zeros then fill)
    patches_path = os.path.join(out_dir, "patches.npy")
    cls_path = os.path.join(out_dir, "cls.npy")
    labels_path = os.path.join(out_dir, "labels.npy")
    indices_path = os.path.join(out_dir, "indices.npy")

    # np.lib.format supports memmap allocation
    patches_mm = np.lib.format.open_memmap(
        patches_path, mode="w+", dtype=np.float32, shape=(Ntot, N_patches, D)
    )
    cls_mm = np.lib.format.open_memmap(
        cls_path, mode="w+", dtype=np.float32, shape=(Ntot, D)
    )
    labels_mm = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int32, shape=(Ntot, 8)
    )
    indices_mm = np.lib.format.open_memmap(
        indices_path, mode="w+", dtype=np.int32, shape=(Ntot,)
    )

    # Optional RAM monitor
    try:
        import psutil
        _proc = psutil.Process()
    except Exception:
        _proc = None

    # Flush the big patches array periodically too. This is the key fix:
    # without it, dirty mmap pages accumulate (~20 GB) in RAM and, over the
    # 9P path used by \\wsl.localhost, trigger swap thrash / OOM. Flushing
    # every FLUSH_EVERY batches forces incremental write-back so the dirty
    # set stays bounded to roughly FLUSH_EVERY * one-batch.
    FLUSH_EVERY = 20

    t0 = time.time()
    cursor = 0
    for bi, (x, y, ix) in enumerate(dl):
        out = extractor(x)
        B = x.shape[0]
        patches_mm[cursor:cursor+B] = out["patches"].cpu().numpy().astype(np.float32)
        cls_mm[cursor:cursor+B] = out["cls"].cpu().numpy().astype(np.float32)
        labels_mm[cursor:cursor+B] = y.numpy().astype(np.int32)
        indices_mm[cursor:cursor+B] = ix.numpy().astype(np.int32)
        cursor += B
        if bi % FLUSH_EVERY == (FLUSH_EVERY - 1):
            patches_mm.flush()
            cls_mm.flush()
            labels_mm.flush()
            indices_mm.flush()
        if bi % 25 == 0:
            mem = ""
            if _proc is not None:
                rss = _proc.memory_info().rss / 1e9
                vm = psutil.virtual_memory()
                sw = psutil.swap_memory()
                mem = (f"  | RSS {rss:.1f}GB  RAM {vm.percent:.0f}%  "
                       f"swap {sw.used/1e9:.1f}GB")
            print(f"  batch {bi+1}/{len(dl)}  ({cursor}/{Ntot}){mem}", flush=True)
    patches_mm.flush()
    elapsed = time.time() - t0

    # Flush to disk
    del patches_mm, cls_mm, labels_mm, indices_mm

    # Meta
    meta = {
        "backbone": args.backbone, "layer": args.layer, "split": args.split,
        "limit": args.limit, "target_hw": args.target_hw, "upsample": args.upsample,
        "Ntot": int(Ntot), "N_patches": int(N_patches), "D": int(D),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_bytes = (Ntot * N_patches * D * 4 + Ntot * D * 4 +
                   Ntot * 8 * 4 + Ntot * 4)
    print(f"[extract] forward done in {elapsed:.1f}s for {Ntot} samples")
    print(f"[extract] saved {out_dir}/ "
          f"({total_bytes/1e9:.2f} GB total on disk, mmap-ready)")


if __name__ == "__main__":
    main()
