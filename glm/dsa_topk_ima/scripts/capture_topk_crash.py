#!/usr/bin/env python3
"""Capture the exact indexer top-k inputs that trigger the illegal memory access.

Monkeypatches ``_indexer_top_k_one_chunk`` so that when the CuTe-DSL top-k
kernel faults, the offending ``(scores_flat, seq_lens, topk_k)`` are saved to
``/tmp/topk_crash.pt`` before the error propagates.

Run with CUDA_LAUNCH_BLOCKING=1 so the error is attributed to the faulting call:

    CUDA_LAUNCH_BLOCKING=1 python /tmp/capture_topk_crash.py --seed 1234 --std 8.0
"""

from __future__ import annotations

import sys

import torch
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

sys.path.insert(0, "/tmp")
from repro_glm52_dsa_sparse_backward import main as _repro_main  # noqa: F401

_DUMP_PATH = "/tmp/topk_crash.pt"
_original = dsa_cudnn_kernels._indexer_top_k_one_chunk
_calls = {"n": 0}


_DUMP_RANGE = range(
    int(__import__("os").environ.get("DUMP_FROM", "39")),
    int(__import__("os").environ.get("DUMP_TO", "43")),
)


def _capturing_chunk(scores_flat, seq_lens, topk_k, return_topk_scores):
    _calls["n"] += 1
    n = _calls["n"]
    print(
        f"topk call #{n}: scores={tuple(scores_flat.shape)} topk_k={topk_k}",
        file=sys.stderr,
        flush=True,
    )
    if n in _DUMP_RANGE:
        path = f"/tmp/topk_call_{n}.pt"
        torch.save(
            {
                "scores_flat": scores_flat.detach().cpu(),
                "seq_lens": seq_lens.detach().cpu(),
                "topk_k": topk_k,
                "return_topk_scores": return_topk_scores,
                "call_index": n,
            },
            path,
        )
        print(f"dumped call #{n} -> {path}", file=sys.stderr, flush=True)
    return _original(scores_flat, seq_lens, topk_k, return_topk_scores)


dsa_cudnn_kernels._indexer_top_k_one_chunk = _capturing_chunk

if __name__ == "__main__":
    _repro_main()
