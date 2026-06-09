# -*- coding: utf-8 -*-
"""Interpretability vs. accuracy: linear probe on raw vs. SAE features.

Trains a small multi-label classifier on top of frozen DINO representations,
comparing two input representations:

  raw  : mean-pooled patch tokens (B, D)          — the standard frozen-feature probe
  sae  : mean-pooled SAE dictionary codes (B, F)  — the interpretable representation

Wafer label = arr_1 multi-label (8 basic defect types). We use the image-level
mean over patch tokens as the feature (a simple, standard linear-probe setup).

Tests how much downstream accuracy is retained when raw features are (dictionary
completeness for the downstream task).
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
import torch.nn as nn


def build_image_features(sae, acts_dir, device="cuda", chunk_images=512, use_sae=False):
    """Return (X, Y): image-level features and multi-label targets.

    X: (Ntot, D) if use_sae=False (mean patch token)
       (Ntot, F) if use_sae=True  (mean SAE code over patches)
    Y: (Ntot, 8) float32
    """
    with open(os.path.join(acts_dir, "meta.json")) as f:
        meta = json.load(f)
    Ntot, N, D = meta["Ntot"], meta["N_patches"], meta["D"]
    patches = np.load(os.path.join(acts_dir, "patches.npy"), mmap_mode="r")
    labels = np.load(os.path.join(acts_dir, "labels.npy")).astype(np.float32)

    feat_dim = sae.dict_size if use_sae else D
    X = np.zeros((Ntot, feat_dim), dtype=np.float32)

    for s in range(0, Ntot, chunk_images):
        e = min(s + chunk_images, Ntot)
        x = torch.from_numpy(np.asarray(patches[s:e])).to(device)  # (B,N,D)
        B = x.shape[0]
        if use_sae:
            flat = x.reshape(B * N, D)
            with torch.inference_mode():
                codes = sae.encode(flat).view(B, N, sae.dict_size)
            feat = codes.mean(dim=1)            # (B, F)
        else:
            feat = x.mean(dim=1)                # (B, D)
        X[s:e] = feat.cpu().numpy()
    return X, labels


class LinearHead(nn.Module):
    def __init__(self, in_dim, n_cls=8):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_cls)

    def forward(self, x):
        return self.fc(x)


def train_eval_probe(
    Xtr, Ytr, Xva, Yva,
    epochs=100, lr=1e-3, weight_decay=1e-4, batch=2048,
    device="cuda", seed=0, standardize=True,
):
    """Train a linear multi-label probe and return val metrics at best-F1 threshold."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if standardize:
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xva = (Xva - mu) / sd

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    model = LinearHead(Xtr.shape[1], Ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()

    M = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(M, device=device)
        for s in range(0, M, batch):
            idx = perm[s:s+batch]
            logits = model(Xtr_t[idx])
            loss = lossf(logits, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.inference_mode():
        prob_va = torch.sigmoid(model(Xva_t)).cpu().numpy()
    return prob_va


def multilabel_metrics(prob, Y, thresholds=None):
    """Compute micro/macro F1 (at best global threshold), mAP, subset accuracy."""
    from sklearn.metrics import average_precision_score, f1_score
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    # Best global threshold by micro-F1
    best_t, best_micro = 0.5, -1
    for t in thresholds:
        pred = (prob >= t).astype(int)
        micro = f1_score(Y, pred, average="micro", zero_division=0)
        if micro > best_micro:
            best_micro, best_t = micro, t
    pred = (prob >= best_t).astype(int)
    micro_f1 = f1_score(Y, pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y, pred, average="macro", zero_division=0)
    # mAP (per-class AP averaged)
    aps = []
    for c in range(Y.shape[1]):
        if Y[:, c].sum() > 0:
            aps.append(average_precision_score(Y[:, c], prob[:, c]))
    mAP = float(np.mean(aps)) if aps else 0.0
    subset_acc = float((pred == Y).all(axis=1).mean())
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "mAP": mAP,
        "subset_acc": subset_acc,
        "best_threshold": float(best_t),
    }
