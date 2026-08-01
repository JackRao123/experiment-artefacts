#!/usr/bin/env python3
"""Standalone check that the staged-block poison actually reaches the wheel.

Runs the real cuDNN DSA indexer top-k at prod-like shapes with a sentinel-filled
block staged immediately before the call, and reports whether the kernel's
output_indices came back carrying the sentinel (i.e. whether the caching
allocator hands our block to the wheel's torch.empty, and whether any slot is
left unwritten). Cheap: single GPU, no model, seconds — run this before
spending a 20-minute trainer boot on the same mechanism.

usage: python3 test_staging.py [--rows 8192] [--cols 32768] [--k 2048]
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--cols", type=int, default=32768)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--sentinel", type=int, default=200003)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels as dk

    dk._ensure_dsa_namespace()
    ns = dk._cudnn_dsa
    dev = "cuda"
    SENT = args.sentinel
    print(f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '-')}")
    print(f"shape rows={args.rows} cols={args.cols} k={args.k} sentinel={SENT}")

    # Prod-like scores: finite band inside each row's causal window, -inf past it
    # (that is what the wheel's own pre-fill guarantees; verified at runtime).
    scores = torch.empty(args.rows, args.cols, dtype=torch.float32, device=dev)
    scores.normal_(mean=0.0, std=1.0)
    seq_lens = torch.arange(1, args.rows + 1, dtype=torch.int32, device=dev)
    seq_lens = seq_lens.clamp(max=args.cols)
    pos = torch.arange(args.cols, device=dev).unsqueeze(0)
    scores.masked_fill_(pos >= seq_lens.to(torch.long).unsqueeze(1), float("-inf"))

    for rep in range(args.reps):
        # 1. control: does a fresh empty() of the wheel's exact shape get our block?
        block = torch.full((args.rows, args.k), SENT, dtype=torch.int32, device=dev)
        del block
        probe = torch.empty(args.rows, args.k, dtype=torch.int32, device=dev)
        ctrl = float((probe == SENT).float().mean())
        del probe

        # 2. stage for the real call — nothing may allocate in between
        block = torch.full((args.rows, args.k), SENT, dtype=torch.int32, device=dev)
        del block
        out = ns.indexer_top_k_wrapper(
            scores, seq_lens, top_k=args.k, next_n=1, return_val=False
        )
        idx = out["indices"]

        unwritten = idx == SENT
        n_un = int(unwritten.sum())
        sl = seq_lens.to(torch.long)
        expected = torch.minimum(sl, torch.full_like(sl, args.k))
        written = (idx != SENT).sum(dim=1)
        short = int((expected < written.new_full((), args.k)).sum())
        neg = int((idx < 0).sum())
        oor = int(((idx >= sl.unsqueeze(1)) & (idx >= 0) & (idx != SENT)).sum())

        print(
            f"rep{rep}: control_sentinel_frac={ctrl:.4f} "
            f"unwritten_slots={n_un} rows_unwritten={int(unwritten.any(dim=1).sum())} "
            f"neg={neg} out_of_window={oor} rows_with_window<k={short}"
        )
        if n_un:
            r = int(unwritten.any(dim=1).nonzero()[0])
            print(f"   e.g. row {r}: window={int(sl[r])} expected={int(expected[r])} "
                  f"written={int(written[r])} unwritten={int(unwritten[r].sum())}")
        if ctrl <= 0.99:
            print("   !! control failed: staged block was NOT handed back — "
                  "a zero unwritten count here proves nothing")


if __name__ == "__main__":
    main()
