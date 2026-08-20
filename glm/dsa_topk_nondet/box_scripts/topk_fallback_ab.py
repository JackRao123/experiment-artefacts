#!/usr/bin/env python3
"""A/B: cuDNN radix indexer_top_k_wrapper vs the masked torch.topk fallback
(the odd-K path in _indexer_top_k_one_chunk) at gate-like shapes.

Checks: (1) torch.topk path determinism x20 on fixed input (set and order),
(2) wall-clock per call for both paths.
"""
import time

import torch
from cudnn import DSA as dsa

dev = torch.device("cuda:0")
TOP_K = 2048
N_DET = 20
N_TIME = 50


def torch_topk_path(scores_scratch, scores_src, seq_lens, key_positions):
    # mirrors _indexer_top_k_one_chunk's odd-K fallback, on a scratch buffer
    scores_scratch.copy_(scores_src)
    scores_scratch.masked_fill_(
        key_positions.unsqueeze(0) >= seq_lens.unsqueeze(1), float("-inf")
    )
    topk_scores, topk_indices = torch.topk(scores_scratch, TOP_K, dim=-1, sorted=False)
    valid = topk_indices < seq_lens.unsqueeze(1)
    return topk_indices.to(torch.int32).masked_fill_(~valid, -1)


for sk in (4096, 12288):
    n_rows = 4096
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    scores = torch.randn(n_rows, sk, generator=gen, device=dev, dtype=torch.float32)
    scores = scores.to(torch.bfloat16).to(torch.float32).contiguous()
    seq_lens = torch.linspace(1, sk, n_rows, device=dev).round().to(torch.int32).clamp(1, sk).contiguous()
    key_positions = torch.arange(sk, device=dev)
    scratch = torch.empty_like(scores)

    # determinism of torch.topk path
    runs = [torch_topk_path(scratch, scores, seq_lens, key_positions).clone() for _ in range(N_DET)]
    torch.cuda.synchronize()
    ref_u, ref_s = runs[0], torch.sort(runs[0], dim=-1).values
    ord_diff = sum(0 if torch.equal(r, ref_u) else 1 for r in runs[1:])
    set_diff = sum(0 if torch.equal(torch.sort(r, dim=-1).values, ref_s) else 1 for r in runs[1:])
    print(f"sk={sk}: torch.topk path  order-diff {ord_diff}/{N_DET-1}  SET-diff {set_diff}/{N_DET-1}")

    # timing
    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_TIME):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / N_TIME * 1e3

    t_cudnn = bench(lambda: dsa.indexer_top_k_wrapper(scores, seq_lens, top_k=TOP_K, next_n=1, return_val=False))
    t_torch = bench(lambda: torch_topk_path(scratch, scores, seq_lens, key_positions))
    print(f"sk={sk}: cudnn radix {t_cudnn:.3f} ms/call   torch.topk fallback {t_torch:.3f} ms/call   "
          f"ratio {t_torch/t_cudnn:.2f}x")
