#!/usr/bin/env python3
"""Test perturbations of the crashing tensor to characterize the top-k trigger.

Usage: python replay_perturb.py <dump.pt> <mode>
modes: orig | noise | sorted | scaled | rowonly | shuffled
"""

from __future__ import annotations

import sys

import torch

ROW = 673
LO, HI = 646, 795  # known-crashing 149-row window


def main() -> None:
    path, mode = sys.argv[1], sys.argv[2]
    blob = torch.load(path, map_location="cpu")
    scores = blob["scores_flat"][LO:HI].clone()
    seq_lens = blob["seq_lens"][LO:HI].clone()
    topk_k = blob["topk_k"]
    r = ROW - LO
    L = int(seq_lens[r])

    if mode == "orig":
        pass
    elif mode == "noise":
        g = torch.Generator().manual_seed(0)
        scores[r, :L] += torch.randn(L, generator=g) * 1e-3
    elif mode == "sorted":
        scores[r, :L] = scores[r, :L].sort(descending=True).values
    elif mode == "scaled":
        scores[r, :L] *= 0.5
    elif mode == "shuffled":
        g = torch.Generator().manual_seed(0)
        scores[r, :L] = scores[r, :L][torch.randperm(L, generator=g)]
    elif mode == "rowonly":
        # keep trigger row, replace all others with a benign row pattern
        benign = blob["scores_flat"][700, : int(seq_lens[0])]
        for i in range(scores.size(0)):
            if i == r:
                continue
            Li = int(seq_lens[i])
            scores[i, :Li] = benign[:Li]
            scores[i, Li:] = float("-inf")
    else:
        raise SystemExit(f"unknown mode {mode}")

    from cudnn.deepseek_sparse_attention.indexer_top_k.api import indexer_top_k_wrapper

    out = indexer_top_k_wrapper(
        scores.cuda(), seq_lens.cuda(), top_k=topk_k, next_n=1, return_val=False
    )
    torch.cuda.synchronize()
    print(f"mode={mode} OK", flush=True)


if __name__ == "__main__":
    main()
