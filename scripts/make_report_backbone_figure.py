# -*- coding: utf-8 -*-
"""Report Fig. 3: 3-panel backbone comparison matching the report text.

Panels: downstream accuracy (raw micro-F1) | dictionary usage (alive features)
        | sparse-coding trade-off (SAE - raw micro-F1, pp).
Excludes the class-monosemanticity and prior-violation panels (confounded /
de-emphasized in the report).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1E2761"
TEAL = "#1C7293"
ORANGE = "#E4572E"

BACK = [
    ("DINOv2\n(SSL)", "dinov2-base", NAVY),
    ("ImageNet-ViT\n(supervised)", "vit-base-patch16-224", TEAL),
    ("Random\n(no pretrain)", "random-vit-base-patch16-224", ORANGE),
]
T = "outputs/tables"

raw_f1, alive, gap = [], [], []
for _, bn, _ in BACK:
    d = np.load(f"{T}/ms__sae__{bn}__L-2__train__d12288__k32__s30000.npz")
    alive.append(int((d["coverage"] > 0).sum()))
    c = json.load(open(f"{T}/classify__sae__{bn}__L-2__train__d12288__k32__s30000.json"))
    raw_f1.append(c["results"]["raw"]["micro_f1"])
    gap.append(c["micro_f1_gap"] * 100.0)

labels = [b[0] for b in BACK]
cols = [b[2] for b in BACK]
x = np.arange(3)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))


def bar(ax, vals, panel, ylabel, fmt, ylim, baseline=None):
    bars = ax.bar(x, vals, color=cols, width=0.62, edgecolor="white")
    ax.set_title(panel, loc="left", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(*ylim)
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls="--", lw=1)
    for b, v in zip(bars, vals):
        va = "bottom" if v >= (baseline or ylim[0]) else "top"
        off = (ylim[1]-ylim[0]) * 0.02
        ax.text(b.get_x()+b.get_width()/2,
                b.get_height() + (off if v >= (baseline or 0) else -off),
                fmt.format(v), ha="center", va=va, fontsize=11,
                fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


bar(axes[0], raw_f1, "(a)", "raw micro-F1", "{:.2f}", (0, 1.12))
bar(axes[1], alive, "(b)", "alive features (of 12,288)", "{:,}", (0, 12288))
bar(axes[2], gap, "(c)", "SAE − raw micro-F1 (pp)", "{:+.1f}", (-11, 6),
    baseline=0.0)

plt.tight_layout()
out = "outputs/figures/report_backbone.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
print("raw_f1", [round(v, 3) for v in raw_f1], "alive", alive,
      "gap", [round(v, 1) for v in gap])
