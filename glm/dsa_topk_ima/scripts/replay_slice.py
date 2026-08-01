#!/usr/bin/env python3
"""Run indexer top-k on a row/col slice of a dumped crashing input.

Usage: python replay_slice.py <dump.pt> <row_lo> <row_hi> [col_width]
Exit 0 if the slice completes, 1 if it crashes.
"""

from __future__ import annotations

import sys

import torch


def main() -> None:
    path, row_lo, row_hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    col_width = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    blob = torch.load(path, map_location="cpu")
    scores = blob["scores_flat"][row_lo:row_hi].contiguous()
    seq_lens = blob["seq_lens"][row_lo:row_hi].contiguous()
    if col_width:
        scores = scores[:, :col_width].contiguous()
        seq_lens = seq_lens.clamp(max=col_width).contiguous()
    print(
        f"slice rows=[{row_lo},{row_hi}) cols={scores.shape[1]} "
        f"seq_lens[{int(seq_lens.min())},{int(seq_lens.max())}]",
        flush=True,
    )
    from cudnn.deepseek_sparse_attention.indexer_top_k.api import indexer_top_k_wrapper

    out = indexer_top_k_wrapper(
        scores.cuda(), seq_lens.cuda(), top_k=blob["topk_k"], next_n=1, return_val=False
    )
    torch.cuda.synchronize()
    idx = out["indices"]
    print(
        f"OK idx_min={int(idx.min())} idx_max={int(idx.max())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
