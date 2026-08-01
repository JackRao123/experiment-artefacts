#!/usr/bin/env python3
"""Hunt the DSA top-k radix under-write edge: candidate flood past smem capacity.

Reading the wheel (indexer_top_k_varlen_util.py:40-95, 597-820) gives a sharp
prediction. For fp32 scores the per-row smem candidate buffer holds
`indexer_topk_smem_input_size = min(max_smem_input_size, max_num_cols)` entries,
where max_smem_input_size is 32768 (num_cols bucketed to <= 65536) or 16384
(above that). Ties at the threshold coarse bin are only stored
`if pos < indexer_topk_smem_input_size` (:664, :680, :706, ...) and the refine
pass then runs on `num_input = min(s_num_input[r_idx], smem_input_size)` (:805).
So when more scores tie at the threshold bin than the buffer holds, the excess
is dropped and the kernel cannot fill all min(K, window) slots — leaving slots
unwritten in a `torch.empty` output.

That requires num_cols > smem capacity, i.e. bucketed num_cols > 32768. A
32768-column run has capacity == columns and cannot drop. Production GLM-5.2
CP-THD forwards reach sk = 60192 (measured), which buckets to 65536 -> capacity
32768 < 60192. The DSA indexer also relu()s head scores, piling mass at exactly
0.0, so a row whose positive-score count is below K puts the threshold on the
zero bin with tens of thousands of ties.

This script builds that geometry directly and reports unwritten slots, with the
allocator-staging positive control on every shape.

usage: python3 test_flood.py [--cols 60192 ...] [--pattern relu_flood]
"""

from __future__ import annotations

import argparse
import os

import torch

SENT = 200003


def build_scores(rows: int, cols: int, window: int, pattern: str, n_pos: int, dev):
    """Scores as the indexer would deliver them: -inf past the window."""
    s = torch.zeros(rows, cols, dtype=torch.float32, device=dev)
    if pattern == "normal":
        s.normal_(mean=0.0, std=1.0)
    elif pattern == "relu_flood":
        # relu output: a few strictly positive scores, everything else exactly 0.
        # n_pos < K forces the top-k threshold onto the zero bin, so every
        # remaining in-window column becomes a tie candidate.
        if n_pos > 0:
            cols_idx = torch.randint(0, window, (rows, n_pos), device=dev)
            vals = torch.rand(rows, n_pos, device=dev) + 1.0
            s.scatter_(1, cols_idx, vals)
    elif pattern == "all_tied":
        s.fill_(1.0)
    else:
        raise SystemExit(f"unknown pattern {pattern}")
    pos = torch.arange(cols, device=dev).unsqueeze(0)
    s.masked_fill_(pos >= window, float("-inf"))
    return s


def smem_capacity(cols: int) -> tuple[int, bool]:
    """Mirror the wheel's fp32 sizing: (candidate capacity, gmem spill enabled)."""
    bucketed = 1 << max(0, (cols - 1).bit_length())
    max_smem = 32 * 1024 if bucketed <= 65536 else 16 * 1024
    cap = min(max_smem, bucketed)
    return cap, bucketed > cap


def run_case(ns, rows, cols, k, window, pattern, n_pos, dev, reps):
    cap, spill = smem_capacity(cols)
    ties = max(0, min(window, cols) - n_pos)
    pred = "DROP EXPECTED" if ties > cap else "no drop possible"
    print(
        f"\ncols={cols} rows={rows} k={k} window={window} pattern={pattern} n_pos={n_pos}\n"
        f"  wheel sizing: candidate_capacity={cap} gmem_spill={spill} "
        f"tie_candidates~{ties} -> {pred}"
    )
    scores = build_scores(rows, cols, window, pattern, n_pos, dev)
    seq_lens = torch.full((rows,), window, dtype=torch.int32, device=dev)
    sl = seq_lens.to(torch.long)
    expected = int(min(k, window))

    for rep in range(reps):
        block = torch.full((rows, k), SENT, dtype=torch.int32, device=dev)
        del block
        probe = torch.empty(rows, k, dtype=torch.int32, device=dev)
        ctrl = float((probe == SENT).float().mean())
        del probe

        block = torch.full((rows, k), SENT, dtype=torch.int32, device=dev)
        del block
        out = ns.indexer_top_k_wrapper(scores, seq_lens, top_k=k, next_n=1, return_val=False)
        idx = out["indices"]

        unwritten = idx == SENT
        n_un = int(unwritten.sum())
        rows_un = int(unwritten.any(dim=1).sum())
        neg = int((idx < 0).sum())
        oor = int(((idx >= sl.unsqueeze(1)) & (idx >= 0) & (idx != SENT)).sum())
        valid = (idx >= 0) & (idx != SENT)
        big = torch.iinfo(torch.int32).max
        srt, _ = torch.where(valid, idx, torch.full_like(idx, big)).sort(dim=-1)
        dup = int(((srt[:, 1:] == srt[:, :-1]) & (srt[:, :-1] != big)).sum())
        written = int(valid[0].sum())

        flag = "  <<< UNDER-WRITE" if n_un else ""
        print(
            f"  rep{rep}: ctrl={ctrl:.4f} unwritten_slots={n_un} rows={rows_un} "
            f"row0_written={written}/{expected} neg={neg} oor={oor} dup={dup}{flag}"
        )
        if ctrl <= 0.99:
            print("    !! staging control failed — zero counts prove nothing here")
        del out, idx, unwritten, valid, srt
    del scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the capacity boundary (default: single case)")
    ap.add_argument("--cols", type=int, default=60192)
    ap.add_argument("--window", type=int, default=0, help="0 = full cols")
    ap.add_argument("--pattern", default="relu_flood",
                    choices=["relu_flood", "normal", "all_tied"])
    ap.add_argument("--n-pos", type=int, default=64)
    args = ap.parse_args()

    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels as dk

    dk._ensure_dsa_namespace()
    ns = dk._cudnn_dsa
    dev = "cuda"
    print(f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '-')} sentinel={SENT}")

    if not args.sweep:
        w = args.window or args.cols
        run_case(ns, args.rows, args.cols, args.k, w, args.pattern, args.n_pos, dev, args.reps)
        return

    # Straddle the capacity boundary: below it no drop is possible, above it is.
    for cols in (32768, 40960, 49152, 60192, 65536, 98304):
        for pattern, n_pos in (("relu_flood", 64), ("all_tied", 0)):
            try:
                run_case(ns, args.rows, cols, args.k, cols, pattern, n_pos, dev, args.reps)
            except Exception as e:  # noqa: BLE001 — keep sweeping past a bad shape
                print(f"  cols={cols} pattern={pattern} FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
