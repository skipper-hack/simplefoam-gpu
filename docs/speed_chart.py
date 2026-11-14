# SPDX-License-Identifier: MIT
"""Draw docs/images/speed.png (measured times; see the README's Speed section)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ITER = 1500
runs = [("OpenFOAM, 7 CPU processes", 6.11, "#9aa0a6"), ("simplefoam-gpu, RTX 3090", 0.29, "#2a78d6")]

fig, ax = plt.subplots(figsize=(8, 2.4), dpi=150)
for i, (label, s_per_it, color) in enumerate(runs):
    minutes = s_per_it * ITER / 60
    ax.barh(i, minutes, height=0.5, color=color)
    ax.text(minutes + 2, i, f"{minutes:.0f} min  ({s_per_it:.2f} s per iteration)", va="center", fontsize=10,
            color="#202124")
ax.set_yticks(range(len(runs)), [r[0] for r in runs], fontsize=10, color="#202124")
ax.invert_yaxis()
ax.set_xlim(0, 240)
ax.set_xlabel(f"minutes for {ITER} iterations (2.5M-cell car model, k-omega SST, i7-10700K)", fontsize=9, color="#5f6368")
ax.set_title("About 20x faster, same result", loc="left", fontsize=12, color="#202124")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#dadce0")
ax.tick_params(axis="x", colors="#5f6368", labelsize=9)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#eeeeee")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "speed.png"), facecolor="white")
