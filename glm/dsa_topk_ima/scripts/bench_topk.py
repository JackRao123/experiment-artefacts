#!/usr/bin/env python3
"""Microbenchmark: indexer top-k latency on the incident shape."""

from __future__ import annotations

import time

import torch


def main() -> None:
    from cudnn.deepseek_sparse_attention.indexer_top_k.api import indexer_top_k_wrapper

    torch.manual_seed(0)
    rows, cols, top_k = 862, 51720, 2048
    scores = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((rows,), cols, dtype=torch.int32, device="cuda")

    for _ in range(3):
        indexer_top_k_wrapper(scores, seq_lens, top_k=top_k, next_n=1, return_val=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        indexer_top_k_wrapper(scores, seq_lens, top_k=top_k, next_n=1, return_val=False)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"topk (862, 51720) k=2048: {dt * 1e3:.3f} ms/call")


if __name__ == "__main__":
    main()
