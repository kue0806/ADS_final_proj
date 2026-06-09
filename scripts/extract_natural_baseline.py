# -*- coding: utf-8 -*-
"""F1 baseline: extract DINOv2 activations on a NATURAL-image dataset (CIFAR-10)
so the same SAE + monosemanticity protocol can be run, giving a reference
distribution to judge whether WBM monosemanticity is "low".

Saves into the same activation-dir layout (patches.npy / labels.npy /
indices.npy / meta.json) so train_sae.py and measure_monosemanticity.py work
unchanged. Labels are stored as 10-dim one-hot to match the multi-label code
path (each image has exactly one active class).

Usage:
    python scripts/extract_natural_baseline.py --backbone facebook/dinov2-base \
        --n 8000 --batch 64
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
from torch.utils.data import DataLoader, Dataset

from src.models.dino_extractor import DinoExtractor
from src.data.wbm_loader import IMAGENET_MEAN, IMAGENET_STD


def _ensure_cifar10(root="data/raw/cifar"):
    """Download + extract CIFAR-10 python batches without torchvision.
    Returns (images uint8 (N,32,32,3), labels int (N,))."""
    import os
    import pickle
    import tarfile
    import urllib.request

    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, "cifar-10-python.tar.gz")
    batch_dir = os.path.join(root, "cifar-10-batches-py")
    if not os.path.isdir(batch_dir):
        if not os.path.exists(tgz):
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            print(f"[nat] downloading CIFAR-10 from {url} ...", flush=True)
            urllib.request.urlretrieve(url, tgz)
        print("[nat] extracting ...", flush=True)
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(root)

    imgs, labels = [], []
    for b in range(1, 6):
        with open(os.path.join(batch_dir, f"data_batch_{b}"), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        imgs.append(d[b"data"])
        labels.extend(d[b"labels"])
    data = np.concatenate(imgs, axis=0)                  # (50000, 3072)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N,32,32,3) uint8
    return data, np.array(labels, dtype=np.int64)


class CIFARNatural(Dataset):
    """CIFAR-10 -> 224x224 ImageNet-normalized, returns (img, onehot10, idx).
    torchvision-free (downloads raw python batches directly)."""

    def __init__(self, root="data/raw/cifar", train=True, n=None, target_hw=224):
        from PIL import Image
        self.Image = Image
        self.data, self.labels = _ensure_cifar10(root)
        self.n = n if n is not None else len(self.data)
        self.target_hw = target_hw

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        arr = self.data[i]                            # (32,32,3) uint8
        label = int(self.labels[i])
        img = self.Image.fromarray(arr).resize(
            (self.target_hw, self.target_hw), self.Image.BILINEAR)
        x = np.asarray(img).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = x.transpose(2, 0, 1)
        onehot = np.zeros(10, dtype=np.int32)
        onehot[label] = 1
        return torch.from_numpy(x), torch.from_numpy(onehot), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="facebook/dinov2-base")
    ap.add_argument("--layer", type=int, default=-2)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out_root", default="data/activations")
    args = ap.parse_args()

    ds = CIFARNatural(n=args.n)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=0, shuffle=False)
    print(f"[nat] CIFAR-10 natural baseline: {len(ds)} images", flush=True)

    extractor = DinoExtractor(model_name=args.backbone, layer=args.layer)
    print(f"[nat] {args.backbone} hidden={extractor.hidden_dim} "
          f"patch={extractor.patch_size}", flush=True)

    first = next(iter(dl))
    out0 = extractor(first[0])
    N, D = out0["patches"].shape[1], out0["patches"].shape[2]
    Ntot = len(ds)

    bn = args.backbone.split("/")[-1]
    tag = f"NAT-cifar10-{bn}__L{args.layer}__train__N{args.n}"
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    patches_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "patches.npy"), mode="w+",
        dtype=np.float32, shape=(Ntot, N, D))
    labels_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "labels.npy"), mode="w+",
        dtype=np.int32, shape=(Ntot, 10))
    indices_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "indices.npy"), mode="w+",
        dtype=np.int32, shape=(Ntot,))

    t0 = time.time()
    cursor = 0
    FLUSH_EVERY = 20
    for bi, (x, y, ix) in enumerate(dl):
        out = extractor(x)
        B = x.shape[0]
        patches_mm[cursor:cursor+B] = out["patches"].cpu().numpy().astype(np.float32)
        labels_mm[cursor:cursor+B] = y.numpy().astype(np.int32)
        indices_mm[cursor:cursor+B] = ix.numpy().astype(np.int32)
        cursor += B
        if bi % FLUSH_EVERY == (FLUSH_EVERY - 1):
            patches_mm.flush(); labels_mm.flush(); indices_mm.flush()
        if bi % 25 == 0:
            print(f"  batch {bi+1}/{len(dl)}  ({cursor}/{Ntot})", flush=True)
    patches_mm.flush()
    del patches_mm, labels_mm, indices_mm

    meta = {"backbone": args.backbone, "layer": args.layer, "split": "train",
            "dataset": "CIFAR10-natural", "Ntot": int(Ntot),
            "N_patches": int(N), "D": int(D), "n_classes": 10}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[nat] done in {time.time()-t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
