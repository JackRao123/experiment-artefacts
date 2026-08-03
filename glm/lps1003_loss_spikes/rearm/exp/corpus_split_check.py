#!/usr/bin/env python3
"""Test the TensorIterator-split theory against the full adj2 mask corpus.

Predictions if the fill is split into F1=[0, N/2), F2=[N/2, N) and F2 is the
race loser:
  P1: every event's bad region starts at row >= R/2 (=7957); destroyers start
      at EXACTLY R/2; nothing ever starts below R/2.
  P2: per rank, numel*4 > 2^31 <=> that rank can produce the boundary-7957
      destroyer. Ranks with numel*4 < 2^31 (single fill kernel) should show a
      different signature (or none).
  P3: partial events live inside the F2 region and their extents are NOT
      128-col aligned (store-race granularity), unlike causal-structure cuts.
"""
import glob, json, collections, os, sys
import torch

DIR = sys.argv[1]
INT32MAX = 2**31 - 1

rank_geom = {}   # rank -> (R, C)
rank_events = collections.defaultdict(list)

for f in sorted(glob.glob(f"{DIR}/rank*/evt*.pt")):
    rank = int(f.split("/")[-2].replace("rank", ""))
    d = torch.load(f, weights_only=False)
    R, C = d["shape"]
    rank_geom[rank] = (R, C)
    re_ = d["row_extents"]
    cnt = re_[:, 2]
    bad = (cnt > 0).nonzero().flatten()
    if len(bad) == 0:
        continue
    lo, hi = int(bad[0]), int(bad[-1])
    n = len(bad)
    whole = (lo == R // 2) and n >= (R - R // 2) - 64
    rank_events[rank].append({
        "lo": lo, "hi": hi, "n": n, "verdict": d["verdict"],
        "whole": whole, "first_col_lo": int(re_[lo, 0]),
    })

# Also scan the jsonl call logs for per-rank shapes (covers ranks with no events)
for f in sorted(glob.glob(f"{DIR}/adjudicate2_rank*.jsonl")):
    rank = int(f.split("rank")[-1].split(".")[0])
    if rank in rank_geom:
        continue
    with open(f) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            sh = d.get("out_shape") or d.get("shape")
            if sh:
                rank_geom[rank] = tuple(sh)
                break

print(f"{'rank':>4} {'R':>6} {'C':>6} {'bytes':>14} {'>2^31?':>7} {'#evt':>5} "
      f"{'#lo==R/2':>8} {'#lo<R/2':>8} {'lo range (partials)':>22}")
for rank in sorted(set(list(rank_geom) + list(rank_events))):
    R, C = rank_geom.get(rank, (0, 0))
    nbytes = R * C * 4
    evs = rank_events.get(rank, [])
    at_half = sum(1 for e in evs if e["lo"] == R // 2)
    below = sum(1 for e in evs if e["lo"] < R // 2)
    partial_los = sorted(e["lo"] for e in evs if e["lo"] != R // 2)
    rng = f"[{partial_los[0]},{partial_los[-1]}] x{len(partial_los)}" if partial_los else "-"
    print(f"{rank:>4} {R:>6} {C:>6} {nbytes:>14,} {str(nbytes > INT32MAX):>7} "
          f"{len(evs):>5} {at_half:>8} {below:>8} {rng:>22}")

# Global invariant check
viol = [(r, e) for r, evs in rank_events.items() for e in evs
        if e["lo"] < rank_geom[r][0] // 2]
print(f"\nP1 events starting below R/2: {len(viol)}")
tot = sum(len(v) for v in rank_events.values())
half = sum(1 for evs in rank_events.values() for e in evs
           if e["lo"] == rank_geom[list(rank_events).pop()][0] // 2)
print(f"total events: {tot}")
