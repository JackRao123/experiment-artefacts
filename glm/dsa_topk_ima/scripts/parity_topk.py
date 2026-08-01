#!/usr/bin/env python3
"""Randomized parity test: cudnn indexer top-k vs torch.topk reference.

Covers both kernel compiles (rows <= 148 and > 148), unaligned/odd row
lengths, and value distributions including fp16-overflow magnitudes and
exact-tie populations (the conditions from incident e3m916q).
"""

from __future__ import annotations

import torch


def _check(scores: torch.Tensor, seq_lens: torch.Tensor, top_k: int, tag: str) -> bool:
    from cudnn.deepseek_sparse_attention.indexer_top_k.api import indexer_top_k_wrapper

    out = indexer_top_k_wrapper(
        scores.cuda(), seq_lens.cuda(), top_k=top_k, next_n=1, return_val=True
    )
    torch.cuda.synchronize()
    indices = out["indices"].cpu()
    values = out["values"].cpu()
    ok = True
    for r in range(scores.size(0)):
        L = int(seq_lens[r])
        k = min(top_k, L)
        ref_vals = torch.topk(scores[r, :L], k).values.sort(descending=True).values
        got = indices[r]
        got_vals = values[r]
        valid = got >= 0
        if int(valid.sum()) != k:
            print(f"{tag} row {r}: valid {int(valid.sum())} != {k}")
            ok = False
            continue
        if int(got.max()) >= L:
            print(f"{tag} row {r}: idx {int(got.max())} >= L={L}")
            ok = False
        gs = got_vals[valid].sort(descending=True).values
        # exact-tie rows may pick different equal-valued keys; compare sorted values
        if not torch.allclose(gs, ref_vals, rtol=1e-5, atol=1e-6):
            print(f"{tag} row {r}: values diverge (got {gs[:4]} ref {ref_vals[:4]})")
            ok = False
    return ok


def main() -> None:
    assert torch.cuda.is_available()
    g = torch.Generator().manual_seed(0)
    top_k = 2048
    all_ok = True
    for trial in range(12):
        num_rows = [37, 149, 149, 862, 149, 300, 862, 149, 862, 149, 200, 862][trial]
        num_cols = [4310, 4310, 51720, 4310, 8192, 4097, 4310, 4311, 51720, 4310, 4310, 4310][trial]
        dist = trial % 4
        scores = torch.empty(num_rows, num_cols, dtype=torch.float32)
        seq_lens = torch.randint(
            low=100, high=num_cols + 1, size=(num_rows,), generator=g, dtype=torch.int32
        )
        for r in range(num_rows):
            L = int(seq_lens[r])
            if dist == 0:
                row = torch.randn(L, generator=g) * 0.02
            elif dist == 1:
                row = torch.randn(L, generator=g) * 8.0
            elif dist == 2:
                # fp16-negative-overflow heavy: 60% of values below -65504
                row = torch.randn(L, generator=g)
                mask = torch.rand(L, generator=g) < 0.6
                row[mask] = -70000.0 - torch.rand(int(mask.sum()), generator=g) * 100000.0
            else:
                # exact-tie heavy: 80% zeros (ReLU-floor pattern)
                row = torch.randn(L, generator=g)
                row[torch.rand(L, generator=g) < 0.8] = 0.0
            scores[r, :L] = row
            scores[r, L:] = float("-inf")
        tag = f"trial{trial}(rows={num_rows},cols={num_cols},dist={dist})"
        ok = _check(scores, seq_lens, top_k, tag)
        print(f"{tag}: {'ok' if ok else 'MISMATCH'}", flush=True)
        all_ok &= ok
    print("ALL PASS" if all_ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
