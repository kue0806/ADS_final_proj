# -*- coding: utf-8 -*-
"""Train a top-k SAE on DINO patch-token activations (streaming version).

Reads activations from a memory-mapped patches.npy via chunked streaming —
peak RAM/GPU usage is bounded by --chunk_images regardless of dataset size.

Usage:
    python scripts/train_sae.py \
        --acts_dir data/activations/dinov2-base__L-2__train \
        --steps 30000 --batch 1024 --k 32 --dict_mult 16 --chunk_images 2000
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
import torch.nn.functional as F

from src.sae.topk_sae import TopKSAE


def load_meta(acts_dir: str) -> dict:
    with open(os.path.join(acts_dir, "meta.json")) as f:
        return json.load(f)


def streaming_mean(patches_path: str, chunk_images: int) -> tuple:
    """Single-pass streaming mean over patch tokens (N, D)."""
    arr = np.load(patches_path, mmap_mode="r")  # (Ntot, N, D)
    Ntot, N, D = arr.shape
    total = Ntot * N
    s = np.zeros(D, dtype=np.float64)
    for start in range(0, Ntot, chunk_images):
        end = min(start + chunk_images, Ntot)
        chunk = np.asarray(arr[start:end]).reshape(-1, D)  # (chunk*N, D)
        s += chunk.sum(axis=0).astype(np.float64)
    mean = (s / total).astype(np.float32)
    return mean, total


def chunk_iter(patches_path: str, chunk_images: int):
    """Yield (chunk_idx, flat_chunk_tensor) for sequential chunks."""
    arr = np.load(patches_path, mmap_mode="r")  # (Ntot, N, D)
    Ntot, N, D = arr.shape
    for start in range(0, Ntot, chunk_images):
        end = min(start + chunk_images, Ntot)
        chunk = np.asarray(arr[start:end]).reshape(-1, D).copy()  # (chunk*N, D)
        yield start, end, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True,
                    help="dir containing patches.npy/cls.npy/labels.npy/meta.json")
    ap.add_argument("--out_dir", default="outputs/sae_ckpts")
    ap.add_argument("--dict_mult", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=30000,
                    help="total optimizer steps across all chunks/epochs")
    ap.add_argument("--chunk_images", type=int, default=2000,
                    help="images held in GPU at once "
                         "(2000 imgs * 256 patches * 768 * 4B ~= 1.5 GB)")
    ap.add_argument("--steps_per_chunk", type=int, default=None,
                    help="optimizer steps per chunk; if unset, derived from steps")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = load_meta(args.acts_dir)
    Ntot, N, D = meta["Ntot"], meta["N_patches"], meta["D"]
    patches_path = os.path.join(args.acts_dir, "patches.npy")
    dict_size = args.dict_mult * D
    n_chunks = (Ntot + args.chunk_images - 1) // args.chunk_images

    if args.steps_per_chunk is None:
        # spread total steps across chunks * outer epochs, with at least 1 epoch
        steps_per_chunk = max(1, args.steps // max(n_chunks, 1))
    else:
        steps_per_chunk = args.steps_per_chunk

    print(f"[train] dataset: Ntot={Ntot} N={N} D={D}")
    print(f"[train] dict={dict_size} k={args.k} batch={args.batch}")
    print(f"[train] chunks: {n_chunks} of {args.chunk_images} imgs each, "
          f"{steps_per_chunk} steps/chunk")
    print(f"[train] computing streaming mean...")

    t0 = time.time()
    mean, total = streaming_mean(patches_path, args.chunk_images)
    print(f"[train] mean computed over {total:,} patches in {time.time()-t0:.1f}s")

    sae = TopKSAE(d_model=D, dict_size=dict_size, k=args.k).to(args.device)
    sae.init_bias_from_mean(torch.from_numpy(mean).to(args.device))
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    feat_activations = torch.zeros(dict_size, device=args.device, dtype=torch.long)
    step = 0
    t0 = time.time()

    outer_epoch = 0
    keep_going = True
    while keep_going:
        outer_epoch += 1
        for ci, (cs, ce, chunk_np) in enumerate(chunk_iter(patches_path,
                                                            args.chunk_images)):
            x_gpu = torch.from_numpy(chunk_np).to(args.device)
            M = x_gpu.shape[0]
            for inner in range(steps_per_chunk):
                idx = torch.randint(0, M, (args.batch,), device=args.device)
                x = x_gpu[idx]
                recon, codes = sae(x)
                loss = ((recon - x) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sae.renorm_decoder()

                with torch.no_grad():
                    active_mask = (codes != 0).any(dim=0)
                    feat_activations += active_mask.long()

                step += 1
                if step % args.log_every == 0 or step == 1:
                    with torch.no_grad():
                        ev = 1.0 - (((recon - x) ** 2).mean() /
                                    (x.var(unbiased=False) + 1e-8))
                        n_active = int(active_mask.sum().item())
                        n_dead = int((feat_activations == 0).sum().item())
                    print(f"  step {step:6d} ep{outer_epoch} chunk{ci+1}/{n_chunks}  "
                          f"loss {loss.item():.4f}  ev {ev.item():.3f}  "
                          f"active {n_active}/{dict_size}  dead {n_dead}")
                if step >= args.steps:
                    keep_going = False
                    break

            del x_gpu
            torch.cuda.empty_cache()
            if not keep_going:
                break

    elapsed = time.time() - t0
    print(f"[train] done {step} steps in {elapsed:.1f}s "
          f"({elapsed/step*1000:.1f} ms/step) | outer epochs {outer_epoch}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(args.acts_dir.rstrip("/\\"))
    out_path = os.path.join(
        args.out_dir,
        f"sae__{tag}__d{dict_size}__k{args.k}__s{step}.pt",
    )
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {
            "d_model": D, "dict_size": dict_size, "k": args.k,
            "acts_dir": args.acts_dir, "steps": step,
            "batch": args.batch, "lr": args.lr, "seed": args.seed,
            "chunk_images": args.chunk_images,
        },
        "feat_activations": feat_activations.cpu().numpy(),
    }, out_path)
    n_dead = int((feat_activations == 0).sum().item())
    print(f"[train] saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"[train] final dead features {n_dead}/{dict_size} "
          f"({n_dead/dict_size*100:.1f}%)")


if __name__ == "__main__":
    main()
