#!/usr/bin/env python3
"""DSA top-k under-write hunt at REAL production geometries.

test_flood.py ruled out the candidate-flood/spill edge, but used constant
per-row windows and power-of-two-ish column counts. The measured production
calls (boot-1 audit, GLM-5.2 CP16 THD forward) differ in two ways that matter
for a tile-boundary bug — the family #814 belonged to:

  * `sk` is odd / unaligned: 39501, 48906, 58311, 60192, 20488, ...
    The kernel skips vector lanes past `aligned_size` filled by `_fill_oob`
    (indexer_top_k_varlen_util.py:654), so unaligned tails take that path.
  * per-row windows VARY inside one launch (seq_lens spans e.g. 4192..27174 at
    rows=6656), mixing rows below top_k (trivial full-write + -1 padding) and
    rows above it (radix) in the same CTA schedule, with shared smem counters
    reset per row.

Each case runs with the allocator-staging detector: an (rows, k) int32 block is
sentinel-filled and freed immediately before the wheel's torch.empty of exactly
that shape, so any slot still holding the sentinel afterwards is an unwritten
slot. The per-shape positive control reports whether the staged block was
actually handed over; without it a zero count proves nothing.

usage: python3 test_prodgeom.py [--rows 6656] [--reps 2]
"""

from __future__ import annotations

import argparse
import os

import torch

SENT = 200003
# (rows, sk, sl_min, sl_max) taken from the boot-1 audit of a real GLM-5.2 run.
PROD_CASES = [
    (6656, 39501, 4192, 27174),
    (6656, 39501, 11914, 39501),
    (2602, 39501, 19824, 36225),
    (4096, 58311, 1, 58311),
    (5120, 48906, 1, 48906),
    (4608, 52668, 26000, 52668),
    (8192, 32768, 1, 32768),
    (15360, 17408, 15361, 17408),
    (4608, 20488, 1, 2048),
]


def make_windows(rows: int, sk: int, lo: int, hi: int, kind: str, dev):
    if kind == "arange":
        w = torch.linspace(lo, hi, rows, device=dev).round().to(torch.int32)
    elif kind == "interleaved":
        # b independent batches, each with its own increasing causal window,
        # reshaped (b, sq) -> b*sq exactly as seq_lens_b does in the glue.
        b = 4
        sq = rows // b
        rowsb = b * sq
        per = torch.linspace(lo, hi, sq, device=dev).round().to(torch.int32)
        w = per.repeat(b)
        if rowsb < rows:
            w = torch.cat([w, per[-1].repeat(rows - rowsb)])
    elif kind == "const":
        w = torch.full((rows,), hi, dtype=torch.int32, device=dev)
    else:
        raise SystemExit(f"bad window kind {kind}")
    return w.clamp(1, sk).contiguous()


def build_scores(rows: int, sk: int, windows, pattern: str, n_pos: int, dev):
    s = torch.zeros(rows, sk, dtype=torch.float32, device=dev)
    if pattern == "relu_flood":
        if n_pos > 0:
            c = (torch.rand(rows, n_pos, device=dev) * windows.to(torch.float32).unsqueeze(1))
            s.scatter_(1, c.long().clamp(min=0), torch.rand(rows, n_pos, device=dev) + 1.0)
    elif pattern == "all_tied":
        s.fill_(1.0)
    elif pattern == "normal":
        s.normal_()
    pos = torch.arange(sk, device=dev).unsqueeze(0)
    s.masked_fill_(pos >= windows.to(torch.long).unsqueeze(1), float("-inf"))
    return s


def run(ns, rows, sk, lo, hi, k, wkind, pattern, n_pos, dev, reps):
    windows = make_windows(rows, sk, lo, hi, wkind, dev)
    scores = build_scores(rows, sk, windows, pattern, n_pos, dev)
    sl = windows.to(torch.long)
    n_short = int((sl <= k).sum())
    exp = torch.minimum(sl, torch.full_like(sl, k))
    print(
        f"\nrows={rows} sk={sk} k={k} windows={wkind}[{lo},{hi}] pattern={pattern} "
        f"sk_odd={sk % 8 != 0} short_rows={n_short} radix_rows={rows - n_short}"
    )
    for rep in range(reps):
        blk = torch.full((rows, k), SENT, dtype=torch.int32, device=dev)
        del blk
        probe = torch.empty(rows, k, dtype=torch.int32, device=dev)
        ctrl = float((probe == SENT).float().mean())
        del probe

        blk = torch.full((rows, k), SENT, dtype=torch.int32, device=dev)
        del blk
        out = ns.indexer_top_k_wrapper(scores, windows, top_k=k, next_n=1, return_val=False)
        idx = out["indices"]

        unwritten = idx == SENT
        n_un = int(unwritten.sum())
        valid = (idx >= 0) & (idx != SENT)
        written = valid.sum(dim=1)
        deficit = (exp - written).clamp(min=0)
        n_def = int((deficit > 0).sum())
        oor = int(((idx >= sl.unsqueeze(1)) & valid).sum())
        big = torch.iinfo(torch.int32).max
        srt, _ = torch.where(valid, idx, torch.full_like(idx, big)).sort(dim=-1)
        dup = int(((srt[:, 1:] == srt[:, :-1]) & (srt[:, :-1] != big)).sum())
        neg = int((idx < 0).sum())

        flag = ""
        if n_un or n_def or oor or dup:
            flag = "  <<< ANOMALY"
        print(
            f"  rep{rep}: ctrl={ctrl:.4f} unwritten={n_un} short_rows_deficit={n_def} "
            f"oor={oor} dup={dup} neg={neg}{flag}"
        )
        if n_un or n_def:
            r = int((unwritten.any(dim=1) | (deficit > 0)).nonzero()[0])
            print(f"    row {r}: window={int(sl[r])} expected={int(exp[r])} "
                  f"written={int(written[r])} unwritten={int(unwritten[r].sum())} "
                  f"branch={'trivial' if int(sl[r]) <= k else 'radix'}")
        if ctrl <= 0.99:
            print("    !! staging control failed — zero counts prove nothing here")
        del out, idx, unwritten, valid, srt
    del scores, windows
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels as dk

    dk._ensure_dsa_namespace()
    ns = dk._cudnn_dsa
    dev = "cuda"
    print(f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '-')} sentinel={SENT}")

    for rows, sk, lo, hi in PROD_CASES:
        for wkind in ("arange", "interleaved"):
            for pattern, n_pos in (("relu_flood", 64), ("all_tied", 0)):
                try:
                    run(ns, rows, sk, lo, hi, args.k, wkind, pattern, n_pos, dev, args.reps)
                except Exception as e:  # noqa: BLE001 — keep going past a bad shape
                    print(f"  rows={rows} sk={sk} {wkind}/{pattern} FAILED: "
                          f"{type(e).__name__}: {e}")
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
