# -*- coding: utf-8 -*-
"""Slide-B figure: real MixedWM38 wafer bin maps, one clean example per defect
type, plus normal and a mixed example. For the Problem Definition slide.

Visualizes the genuine data (contrast with the abstract black-box on slide A).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

NAVY = "#0D326F"

# 8 basic single-defect dim names (index 0..7)
DIM_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
             "Loc", "Near-Full", "Random", "Scratch"]

d = np.load("data/raw/MixedWM38.npz", allow_pickle=True)
arr0, arr1 = d["arr_0"], d["arr_1"]
n_faults = arr1.sum(axis=1)

# categorical colormap: 0=blank(white-ish), 1=pass(light blue), 2=fail(navy/orange)
cmap = ListedColormap(["#F2F5FA", "#A9C0E8", "#E4572E"])

# Pick: 1 normal + 8 single-defect (one clean example each) + 1 mixed
picks = []  # (title, index)

# normal
normal_idx = np.where(n_faults == 0)[0]
picks.append(("Normal", int(normal_idx[0])))

# one clean single-defect per dim
rng = np.random.default_rng(0)
for dim in range(8):
    cand = np.where((n_faults == 1) & (arr1[:, dim] == 1))[0]
    picks.append((DIM_NAMES[dim], int(cand[0])))

# one mixed (3 faults) — show the "Mixed" nature
mixed_idx = np.where(n_faults == 3)[0]
mi = int(mixed_idx[0])
n_mixed = int(n_faults[mi])
picks.append((f"Mixed ({n_mixed} defects)", mi))

# Layout: 2 rows x 5 cols = 10 panels
fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
for ax, (title, idx) in zip(axes.flatten(), picks):
    ax.imshow(arr0[idx], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    # small in-panel label (top-left) instead of a title — no overlap when scaled
    ax.text(0.04, 0.96, title, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=NAVY, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                      alpha=0.75))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#C5D2E8")

# legend
from matplotlib.patches import Patch
legend = [Patch(facecolor="#F2F5FA", edgecolor="#C5D2E8", label="blank"),
          Patch(facecolor="#A9C0E8", label="pass die"),
          Patch(facecolor="#E4572E", label="fail die (defect)")]
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
           fontsize=11, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
out = "outputs/figures/wbm_gallery.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)

# also report what was picked
for title, idx in picks:
    print(f"  {title:<22} raw_idx={idx}  faults={int(n_faults[idx])}")
