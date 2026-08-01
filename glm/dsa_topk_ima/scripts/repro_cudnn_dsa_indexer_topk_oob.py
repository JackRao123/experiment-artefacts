#!/usr/bin/env python3
"""Reproduce the cuDNN-frontend 1.26.0 DSA indexer top-k OOB (incident e3m916q).

The CuTe-DSL radix top-k kernel (cudnn.deepseek_sparse_attention
indexer_top_k_decode_varlen) fills the out-of-bounds lanes of a row's final
vector tile with -inf and counts those phantom lanes as real elements in its
radix histogram and candidate-collection passes. When a row's top-k threshold
lands in the fp16 -inf coarse bin — the row has fewer than top_k values above
~-65504 — phantom lanes flood the threshold bin's candidate list:

* >148 rows (large_occupancy compile): candidate buffers (512 smem entries +
  num_cols gmem per row) overflow -> out-of-bounds shared/global writes ->
  "CUDA error: an illegal memory access was encountered" (Xid 43);
* any configuration: phantom lanes may be selected as winners, yielding
  out-of-range indices (silent corruption).

This script builds a synthetic batch that deterministically floods the
threshold bin and checks the result against torch.topk. On unpatched
nvidia-cudnn-frontend 1.26.0 it prints out-of-range index failures (and the
GLM-5.2 incident shape crashes outright); with the carried patch applied
(server/patches/cudnn-frontend-1.26.0-dsa-indexer-topk-oob.patch) it passes.

Run inside the trainers worker image on an SM90+ GPU:

    python server/scripts/repro_cudnn_dsa_indexer_topk_oob.py
"""

from __future__ import annotations

import torch

_NUM_ROWS = 149  # > 148 selects the large_occupancy kernel compile
_NUM_COLS = 4310  # incident front-segment width (rank 4, 55168/64 padded doc)
_TOP_K = 2048
_TRIGGER_LEN = 4122  # partial final tile -> 4080 phantom -inf lanes
_NUM_ABOVE = 1122  # values above the fp16 -inf coarse bin (< top_k)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("requires a CUDA GPU (SM90+)")
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        raise SystemExit("indexer top-k kernel requires SM90+")

    scores = torch.empty(_NUM_ROWS, _NUM_COLS, dtype=torch.float32)
    # Benign rows: strictly increasing ramp; thresholds far above the -inf bin.
    scores[1:] = torch.arange(_NUM_COLS, dtype=torch.float32).expand(_NUM_ROWS - 1, -1)
    # Trigger row: _NUM_ABOVE small values, the rest fp16-negative-overflow.
    row0 = torch.empty(_NUM_COLS, dtype=torch.float32)
    row0[:_NUM_ABOVE] = torch.arange(_NUM_ABOVE, dtype=torch.float32)
    row0[_NUM_ABOVE:_TRIGGER_LEN] = torch.linspace(
        -66000.0, -200000.0, _TRIGGER_LEN - _NUM_ABOVE
    )
    row0[_TRIGGER_LEN:] = float("-inf")  # beyond seq_len; kernel must not read
    scores[0] = row0

    seq_lens = torch.full((_NUM_ROWS,), _NUM_COLS, dtype=torch.int32)
    seq_lens[0] = _TRIGGER_LEN

    # GPU-only wheel; absent from the CPU typecheck env.
    from cudnn.deepseek_sparse_attention.indexer_top_k.api import (  # ty: ignore[unresolved-import]  # pyright: ignore[reportMissingImports]
        indexer_top_k_wrapper,
    )

    out = indexer_top_k_wrapper(
        scores.cuda(), seq_lens.cuda(), top_k=_TOP_K, next_n=1, return_val=True
    )
    torch.cuda.synchronize()
    indices = out["indices"].cpu()
    values = out["values"].cpu()

    failures = 0
    for r in range(_NUM_ROWS):
        L = int(seq_lens[r])
        k = min(_TOP_K, L)
        got = indices[r]
        valid = got >= 0
        if int(valid.sum()) != k:
            print(f"row {r}: expected {k} valid indices, got {int(valid.sum())}")
            failures += 1
            continue
        if int(got.max()) >= L:
            print(
                f"row {r}: out-of-range index {int(got.max())} >= seq_len {L} "
                "(phantom OOB tile lane selected)"
            )
            failures += 1
        got_vals = values[r][valid].sort(descending=True).values
        ref_vals = torch.topk(scores[r, :L], k).values.sort(descending=True).values
        if not torch.allclose(got_vals, ref_vals, rtol=1e-5, atol=1e-6):
            print(f"row {r}: selected values diverge from torch.topk")
            failures += 1

    if failures:
        print(f"FAIL: {failures} row checks failed (kernel has the OOB bug)")
        raise SystemExit(1)
    print("PASS: top-k results match reference")


if __name__ == "__main__":
    main()
