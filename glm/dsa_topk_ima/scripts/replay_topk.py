#!/usr/bin/env python3
"""Replay a dumped indexer top-k input in a fresh CUDA context.

Usage: python replay_topk.py /tmp/topk_call_41.pt [--bisect-rows]
"""

from __future__ import annotations

import argparse

import torch


def _load(path: str):
    blob = torch.load(path, map_location="cpu")
    print(
        {
            "scores_flat": tuple(blob["scores_flat"].shape),
            "dtype": str(blob["scores_flat"].dtype),
            "seq_lens": tuple(blob["seq_lens"].shape),
            "seq_lens_min": int(blob["seq_lens"].min()),
            "seq_lens_max": int(blob["seq_lens"].max()),
            "topk_k": blob["topk_k"],
            "call_index": blob["call_index"],
            "scores_min": float(blob["scores_flat"].min()),
            "scores_max": float(blob["scores_flat"].max()),
            "scores_neginf_frac": float(
                (blob["scores_flat"] == float("-inf")).float().mean()
            ),
            "scores_nan": int(blob["scores_flat"].isnan().sum()),
        }
    )
    return blob


def _run(scores_flat: torch.Tensor, seq_lens: torch.Tensor, topk_k: int) -> str:
    from cudnn.deepseek_sparse_attention.indexer_top_k.api import indexer_top_k_wrapper

    scores_flat = scores_flat.cuda()
    seq_lens = seq_lens.cuda()
    out = indexer_top_k_wrapper(
        scores_flat, seq_lens, top_k=topk_k, next_n=1, return_val=False
    )
    torch.cuda.synchronize()
    idx = out["indices"] if isinstance(out, dict) else out[0]
    return f"OK indices={tuple(idx.shape)} idx_min={int(idx.min())} idx_max={int(idx.max())}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument(
        "--bisect-rows",
        action="store_true",
        help="Binary-search the smallest row prefix that still crashes.",
    )
    args = parser.parse_args()
    blob = _load(args.dump)
    scores = blob["scores_flat"]
    seq_lens = blob["seq_lens"]
    topk_k = blob["topk_k"]

    if not args.bisect_rows:
        print(_run(scores, seq_lens, topk_k))
        return

    lo, hi = 1, scores.size(0)
    # Invariant: full range crashes; find smallest crashing row prefix.
    while lo < hi:
        mid = (lo + hi) // 2
        print(f"trying rows [0, {mid}) ...", flush=True)
        try:
            _run(scores[:mid].contiguous(), seq_lens[:mid].contiguous(), topk_k)
            print(f"rows [0, {mid}): OK")
            lo = mid + 1
        except Exception as exc:
            print(f"rows [0, {mid}): CRASH ({type(exc).__name__})")
            return  # CUDA context poisoned; rerun with the prefix directly.
    print(f"minimal prefix: {lo}")


if __name__ == "__main__":
    main()
