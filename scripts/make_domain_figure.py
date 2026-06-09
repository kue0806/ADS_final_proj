# -*- coding: utf-8 -*-
"""WBM vs CIFAR (natural-image) monosemanticity comparison figure (F1).

Shows the key finding: the AXIS of monosemanticity shifts under transfer to an
abstract domain — natural images encode "what" (class-selective), wafer maps
encode "where" (spatially-selective).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1E2761"
ORANGE = "#E4572E"

T = "outputs/tables"
WBM = f"{T}/ms__sae__dinov2-base__L-2__train__d12288__k32__s30000.npz"
NAT = f"{T}/ms__sae__NAT-cifar10-dinov2-base__L-2__train__N8000__d12288__k32__s30000.npz"

dw = np.load(WBM)
dn = np.load(NAT)


def alive_vals(d, key):
    cov = d["coverage"]
    al = cov > 0
    return d[key][al]


fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

# Panel 1: class_ms distributions
axes[0].hist(alive_vals(dn, "class_ms"), bins=40, alpha=0.7, color=ORANGE,
             density=True, label="CIFAR-10 (natural)")
axes[0].hist(alive_vals(dw, "class_ms"), bins=40, alpha=0.7, color=NAVY,
             density=True, label="Wafer maps")
axes[0].axvline(np.median(alive_vals(dn, "class_ms")), color=ORANGE, ls="--", lw=2)
axes[0].axvline(np.median(alive_vals(dw, "class_ms")), color=NAVY, ls="--", lw=2)
axes[0].set_title("(a)", loc="left", fontsize=13, fontweight="bold", color=NAVY)
axes[0].set_xlabel("class monosemanticity\n(1 = selective for a single class)")
axes[0].set_ylabel("density")
axes[0].legend(fontsize=9)
axes[0].spines[["top", "right"]].set_visible(False)

# Panel 2: spatial_ms distributions
axes[1].hist(alive_vals(dn, "spatial_ms"), bins=40, alpha=0.7, color=ORANGE,
             density=True, label="CIFAR-10 (natural)")
axes[1].hist(alive_vals(dw, "spatial_ms"), bins=40, alpha=0.7, color=NAVY,
             density=True, label="Wafer maps")
axes[1].axvline(np.median(alive_vals(dn, "spatial_ms")), color=ORANGE, ls="--", lw=2)
axes[1].axvline(np.median(alive_vals(dw, "spatial_ms")), color=NAVY, ls="--", lw=2)
axes[1].set_title("(b)", loc="left", fontsize=13, fontweight="bold", color=NAVY)
axes[1].set_xlabel("spatial monosemanticity\n(1 = localized to a position)")
axes[1].legend(fontsize=9)
axes[1].spines[["top", "right"]].set_visible(False)

# Panel 3: median scatter — the axis shift
labels = ["CIFAR-10\n(natural)", "Wafer maps\n(abstract)"]
class_meds = [np.median(alive_vals(dn, "class_ms")),
              np.median(alive_vals(dw, "class_ms"))]
spat_meds = [np.median(alive_vals(dn, "spatial_ms")),
             np.median(alive_vals(dw, "spatial_ms"))]
cols = [ORANGE, NAVY]
# Draw the shift arrow first (so points sit on top)
axes[2].annotate("", xy=(class_meds[1], spat_meds[1]),
                 xytext=(class_meds[0], spat_meds[0]),
                 arrowprops=dict(arrowstyle="-|>", color="gray", lw=2.2, ls="--"))
# Points
axes[2].scatter(class_meds[0], spat_meds[0], s=320, color=ORANGE,
                edgecolor="white", zorder=3)
axes[2].scatter(class_meds[1], spat_meds[1], s=320, color=NAVY,
                edgecolor="white", zorder=3)
# Labels placed clear of points and the arrow (no overlap):
#   CIFAR point is bottom-right -> label below-left of it
#   Wafer point is top-left    -> label above-right of it
axes[2].annotate("CIFAR-10\n(natural images)",
                 (class_meds[0], spat_meds[0]),
                 fontsize=10, ha="right", va="top", color=ORANGE,
                 fontweight="bold",
                 xytext=(-12, -10), textcoords="offset points")
axes[2].annotate("Wafer maps\n(abstract data)",
                 (class_meds[1], spat_meds[1]),
                 fontsize=10, ha="left", va="bottom", color=NAVY,
                 fontweight="bold",
                 xytext=(14, 8), textcoords="offset points")
axes[2].set_xlabel("median class monosemanticity\n(semantic axis)")
axes[2].set_ylabel("median spatial monosemanticity\n(spatial axis)")
axes[2].set_title("(c)", loc="left", fontsize=13, fontweight="bold", color=NAVY)
axes[2].set_xlim(0.05, 0.62); axes[2].set_ylim(0.05, 0.68)
axes[2].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = "outputs/figures/domain_axis_shift.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)

print("\n=== WBM vs CIFAR (alive features) ===")
for key in ["class_ms", "spatial_ms", "combined_ms"]:
    wv, nv = alive_vals(dw, key), alive_vals(dn, key)
    print(f"{key:<12}  WBM median={np.median(wv):.3f}   "
          f"CIFAR median={np.median(nv):.3f}")
