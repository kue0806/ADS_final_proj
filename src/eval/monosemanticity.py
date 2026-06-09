"""Per-feature monosemanticity scores for SAE dictionaries.

class_ms   in [0,1]: concentration of a feature's activation across classes.
spatial_ms in [0,1]: concentration across patch positions.
coverage   in [0,1]: fraction of patches in which the feature fires.
combined = class_ms * spatial_ms.
"""
from __future__ import annotations
import numpy as np
import torch


def feature_activations_per_class(
    sae,                              # TopKSAE
    patches: np.ndarray,              # (Ntot, N, D) float32
    labels: np.ndarray,               # (Ntot, 8) int32 (multi-label one-hot)
    batch_size: int = 256,            # images per batch
    device: str = "cuda",
):
    """Accumulate per-class and spatial activation statistics.

    Returns:
        per_class_sum   (dict_size, n_classes) float64 — sum of feature activations
                                                          for images carrying class c
        n_per_class     (n_classes,) int — count of images carrying class c
        spatial_sum     (dict_size, N) float64 — sum of per-patch activations
                                                  (over all images)
        spatial_count   (dict_size,)   int   — number of images that ever
                                               activated feature f at any patch
        coverage_num    (dict_size,)   int   — total patches where feature fires
        coverage_den    int                  — total patches inspected
    """
    sae.eval()
    Ntot, N, D = patches.shape
    n_classes = labels.shape[1]
    dict_size = sae.dict_size

    per_class_sum = np.zeros((dict_size, n_classes), dtype=np.float64)
    n_per_class = np.zeros((n_classes,), dtype=np.int64)
    spatial_sum = np.zeros((dict_size, N), dtype=np.float64)
    spatial_count = np.zeros((dict_size,), dtype=np.int64)
    coverage_num = np.zeros((dict_size,), dtype=np.int64)
    coverage_den = 0

    for s in range(0, Ntot, batch_size):
        e = min(s + batch_size, Ntot)
        x = torch.from_numpy(patches[s:e]).to(device)           # (B, N, D)
        y = labels[s:e]                                          # (B, 8)
        B = x.shape[0]
        # Flatten (B, N, D) -> (B*N, D) for encode
        flat = x.reshape(B * N, D)
        with torch.inference_mode():
            codes = sae.encode(flat)                             # (B*N, dict_size)
        codes_np = codes.cpu().numpy()                            # (B*N, dict_size)
        codes_bnf = codes_np.reshape(B, N, dict_size)             # (B, N, dict_size)

        # Per-image mean activation per feature (mean over patches)
        per_image_act = codes_bnf.mean(axis=1)                    # (B, dict_size)
        # Class accumulation: for each image, add per_image_act to classes it carries
        # per_class_sum (dict_size, n_classes) += per_image_act.T @ y
        per_class_sum += (per_image_act.T.astype(np.float64) @ y.astype(np.float64))
        n_per_class += y.sum(axis=0).astype(np.int64)

        # Spatial accumulation: for each feature, mean over images of its
        # activation across N patches.
        # We accumulate sum and divide by Ntot at the end.
        spatial_sum += codes_bnf.sum(axis=0).T.astype(np.float64)  # (dict_size, N)

        # Coverage: how many patches across the batch had each feature firing
        coverage_num += (codes_np > 0).sum(axis=0).astype(np.int64)
        coverage_den += codes_np.shape[0]

        # Spatial count: images where feature ever fired anywhere
        any_fire = (codes_bnf > 0).any(axis=1)                    # (B, dict_size)
        spatial_count += any_fire.sum(axis=0).astype(np.int64)

    return {
        "per_class_sum": per_class_sum,
        "n_per_class": n_per_class,
        "spatial_sum": spatial_sum,
        "spatial_count": spatial_count,
        "coverage_num": coverage_num,
        "coverage_den": coverage_den,
        "Ntot": Ntot,
        "N": N,
    }


def class_ms_score(per_class_sum: np.ndarray, n_per_class: np.ndarray) -> np.ndarray:
    """Per-feature class-conditional monosemanticity score in [0, 1].

    For each feature f, compute the *normalized* mean activation per class:
        mu[f, c] = per_class_sum[f, c] / n_per_class[c]
    Then convert to a probability distribution over classes and report
        MS_class[f] = 1 - H(p_f) / log(C)
    where H is Shannon entropy. 1 = activates only on one class, 0 = uniform.
    """
    n_per_class = np.maximum(n_per_class, 1)  # avoid /0
    mu = per_class_sum / n_per_class           # (dict_size, n_classes)
    eps = 1e-12
    a = np.clip(mu, 0, None)
    Z = a.sum(axis=1, keepdims=True) + eps
    p = a / Z
    h = -(p * np.log(p + eps)).sum(axis=1)
    h_norm = h / np.log(per_class_sum.shape[1])
    return (1.0 - h_norm).astype(np.float32)


def spatial_ms_score(spatial_sum: np.ndarray, Ntot: int) -> np.ndarray:
    """Per-feature spatial concentration score in [0, 1].

    Computes the mean spatial activation profile per feature (N patches),
    normalizes to a probability distribution, and reports
        MS_spatial[f] = 1 - H(p_f) / log(N)
    1 = activation concentrated at a single patch, 0 = uniform across patches.
    """
    mu = spatial_sum / max(Ntot, 1)            # (dict_size, N)
    eps = 1e-12
    a = np.clip(mu, 0, None)
    Z = a.sum(axis=1, keepdims=True) + eps
    p = a / Z
    h = -(p * np.log(p + eps)).sum(axis=1)
    h_norm = h / np.log(spatial_sum.shape[1])
    return (1.0 - h_norm).astype(np.float32)


def coverage(coverage_num: np.ndarray, coverage_den: int) -> np.ndarray:
    return (coverage_num / max(coverage_den, 1)).astype(np.float32)


def compose_ms(class_ms: np.ndarray, spatial_ms: np.ndarray) -> np.ndarray:
    """Combined monosemanticity: feature must be both class- and spatially-
    concentrated. Hadamard product in [0,1]."""
    return (class_ms * spatial_ms).astype(np.float32)
