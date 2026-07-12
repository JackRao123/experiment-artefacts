#!/usr/bin/env python3
"""MoE expert-routing imbalance vs per-GPU memory at S=163840 (dropless Ultra).

Left: per-physical-GPU MoE routing load (tokens dispatched to that GPU's 64
experts, summed over all MoE layers) for 6 datasets - 3 ramp variants + 3 random
seeds - each sorted ascending. Right: per-GPU memory.used at 163840 from the U1
sweep (sorted). Shows routing imbalance is mild (~1.2-1.3x) and data-independent,
while the memory curve has a sharper top (stage-specific + long-context effects).
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib import cm

# per-physical-GPU routing load (millions of token-routes), physical rank 0..31
ROUTING = {
    "syn1 ramp": [
        5.49,
        5.16,
        5.15,
        6.45,
        4.85,
        5.30,
        5.56,
        5.30,
        5.05,
        6.23,
        5.49,
        5.46,
        5.56,
        5.03,
        5.23,
        5.21,
        5.54,
        5.26,
        5.67,
        5.45,
        4.88,
        5.53,
        5.34,
        5.57,
        5.52,
        5.78,
        6.04,
        5.14,
        6.47,
        5.46,
        6.26,
        6.20,
    ],
    "syn2 ramp+off": [
        5.26,
        5.15,
        5.57,
        6.53,
        4.77,
        5.40,
        5.33,
        5.25,
        5.03,
        6.25,
        5.44,
        5.50,
        5.62,
        4.92,
        5.32,
        5.17,
        5.54,
        5.32,
        5.71,
        5.36,
        4.92,
        5.49,
        5.34,
        5.57,
        5.53,
        5.80,
        6.04,
        5.13,
        6.47,
        5.45,
        6.21,
        6.23,
    ],
    "syn3 ramp+stride": [
        5.30,
        5.19,
        5.21,
        6.61,
        4.84,
        5.40,
        5.31,
        5.39,
        4.95,
        6.27,
        5.47,
        5.50,
        5.66,
        4.94,
        5.25,
        5.21,
        5.54,
        5.31,
        5.71,
        5.36,
        4.92,
        5.50,
        5.33,
        5.59,
        5.54,
        5.77,
        6.05,
        5.14,
        6.49,
        5.52,
        6.20,
        6.14,
    ],
    "rand seed0": [
        5.26,
        5.22,
        5.32,
        6.42,
        4.85,
        5.56,
        5.43,
        5.20,
        5.25,
        6.33,
        5.04,
        4.97,
        5.87,
        5.42,
        5.12,
        5.25,
        5.17,
        5.25,
        5.62,
        5.76,
        4.98,
        5.64,
        5.47,
        5.37,
        6.06,
        5.05,
        5.37,
        5.56,
        7.01,
        5.14,
        6.14,
        6.53,
    ],
    "rand seed1": [
        5.47,
        5.06,
        5.24,
        6.24,
        5.01,
        5.50,
        5.33,
        5.39,
        5.25,
        6.34,
        5.05,
        4.97,
        5.84,
        5.42,
        5.13,
        5.24,
        5.23,
        5.22,
        5.65,
        5.74,
        4.97,
        5.68,
        5.42,
        5.34,
        6.09,
        5.05,
        5.46,
        5.56,
        6.96,
        5.14,
        6.08,
        6.52,
    ],
    "rand seed2": [
        5.38,
        4.78,
        5.44,
        6.67,
        5.14,
        5.53,
        5.12,
        5.18,
        5.25,
        6.31,
        5.05,
        4.90,
        5.88,
        5.54,
        5.11,
        5.22,
        5.19,
        5.30,
        5.63,
        5.77,
        4.90,
        5.70,
        5.46,
        5.31,
        6.03,
        4.99,
        5.34,
        5.59,
        7.07,
        5.30,
        6.05,
        6.48,
    ],
}
# per-GPU memory.used (GiB) at 163840 from U1 sweep (physical rank 0..31)
MEM_163840 = [
    119.6,
    114.9,
    110.1,
    157.0,
    111.7,
    115.9,
    116.4,
    126.5,
    116.8,
    126.3,
    123.5,
    119.7,
    125.9,
    115.3,
    118.1,
    117.7,
    114.4,
    107.2,
    121.2,
    108.5,
    106.7,
    113.2,
    111.2,
    115.7,
    110.9,
    118.6,
    120.5,
    109.2,
    133.1,
    111.8,
    133.5,
    127.2,
]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
x = list(range(32))
names = list(ROUTING)
colors = cm.viridis([i / (len(names) - 1) for i in range(len(names))])
for name, c in zip(names, colors):
    ys = sorted(ROUTING[name])
    dashed = name.startswith("rand")
    axL.plot(
        x, ys, marker="o", ms=3, lw=1.8, ls="--" if dashed else "-", color=c, label=name
    )
axL.set_xlabel("GPU rank (sorted ascending)")
axL.set_ylabel("MoE routing load (millions of token-routes)")
axL.set_title(
    "MoE routing load per GPU (dispatch volume)\n"
    "mild (~1.2-1.3x max/median) and ~identical for ramp vs random "
    "-> intrinsic, not data-driven"
)
axL.grid(True, alpha=0.3)
axL.legend(fontsize=8)

axR.plot(x, sorted(MEM_163840), marker="o", ms=4, lw=2, color="crimson")
axR.axhline(178.35, color="red", ls=":", lw=1.2, label="B200 cap (178 GiB)")
axR.set_xlabel("GPU rank (sorted ascending)")
axR.set_ylabel("GPU memory.used (GiB)")
axR.set_title(
    "Per-GPU memory.used at 163840 (U1 sweep)\n"
    "sharper top: stage-specific + long-context dropless amplification"
)
axR.grid(True, alpha=0.3)
axR.legend(fontsize=8)

fig.tight_layout()
out = "examples/ptt-nemotron3-super-sft/moe_routing_vs_memory_163840.png"
fig.savefig(out, dpi=150)
print("wrote", out)
