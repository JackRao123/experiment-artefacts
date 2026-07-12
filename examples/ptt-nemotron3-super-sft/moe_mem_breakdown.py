#!/usr/bin/env python3
"""Decompose a CUDA memory snapshot set into per-category peak composition.

For each rank snapshot (memory.rankN.pickle), reconstruct the allocation
timeline from device_traces, find the moment of maximum live memory, and bucket
the live blocks at that moment by allocation-site stack into categories
(expert activation, expert GEMM, MoE dispatch, attention, mamba, embedding,
other/backward). Reports the hottest rank's peak composition and the tallest
single allocation per category. Run with the trainer venv python (needs torch).

Usage: python moe_mem_breakdown.py <snapshot_dir> [label]
"""

import glob
import os
import sys
import pickle

CATS = [
    ("expert_act", ("experts.py", "bias_act")),
    ("expert_gemm", ("grouped_linear.py", "grouped_gemm", "groupedgemm")),
    (
        "moe_dispatch",
        (
            "mappings.py",
            "all_to_all",
            "_gather_along",
            "token_dispatch",
            "moe_layer.py",
        ),
    ),
    ("attention", ("attention", "dot_product", "flash", "core_attn", "fused_attn")),
    ("mamba", ("mamba", "mixer", "ssm", "causal_conv")),
    ("embedding", ("embedding",)),
]


def categorize(frames):
    if not frames:
        return "other/backward(no-frame)"
    blob = " ".join(
        f"{f.get('filename', '')}:{f.get('name', '')}".lower()
        for f in frames
        if isinstance(f, dict)
    )
    for cat, keys in CATS:
        if any(k in blob for k in keys):
            return cat
    return "other"


def analyze_rank(path):
    with open(path, "rb") as fh:
        s = pickle.load(fh)
    dt = max(s["device_traces"], key=len)  # the populated device
    live = {}  # addr -> (size, category)
    total = 0
    peak = 0
    peak_live = None
    maxcat = {}  # category -> largest single alloc seen
    for e in dt:
        a = e.get("action")
        if a == "alloc":
            cat = categorize(e.get("frames"))
            live[e["addr"]] = (e["size"], cat)
            total += e["size"]
            if e["size"] > maxcat.get(cat, 0):
                maxcat[cat] = e["size"]
            if total > peak:
                peak = total
                peak_live = dict(live)
        elif a == "free_completed":
            v = live.pop(e.get("addr"), None)
            if v:
                total -= v[0]
    comp = {}
    for sz, cat in (peak_live or {}).values():
        comp[cat] = comp.get(cat, 0) + sz
    return peak, comp, maxcat


def main():
    d = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(d.rstrip("/"))
    files = glob.glob(os.path.join(d, "memory.rank*.pickle"))
    best = None
    for f in files:
        peak, comp, maxcat = analyze_rank(f)
        r = int(os.path.basename(f).split("rank")[1].split(".")[0])
        if best is None or peak > best[0]:
            best = (peak, comp, maxcat, r)
    peak, comp, maxcat, r = best
    G = 2**30
    print(
        f"\n===== {label}: hottest rank = r{r}, reconstructed peak = {peak / G:.1f} GiB ====="
    )
    print("  peak composition (live GiB by category):")
    for cat, v in sorted(comp.items(), key=lambda kv: -kv[1]):
        print(f"    {cat:<26} {v / G:7.2f}")
    print("  tallest single allocation per category (GiB):")
    for cat, v in sorted(maxcat.items(), key=lambda kv: -kv[1]):
        print(f"    {cat:<26} {v / G:7.2f}")


if __name__ == "__main__":
    main()
