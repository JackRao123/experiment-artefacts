#!/usr/bin/env python3
"""Minimal repro probe: is cudnn.DSA.indexer_top_k_wrapper (TRT-LLM CuTe-DSL
radix select) run-to-run nondeterministic on a FIXED input?

For each (sk, content) config:
  - build ONE fp32 scores tensor [n_rows, sk] and int32 causal seq_lens
  - call indexer_top_k_wrapper 20x on the SAME tensors
  - verify the input was not mutated (bitwise)
  - compare indices across calls: unsorted (ordering) and per-row-sorted
    with -1 padding kept (selection set)

Interpretation:
  - selection SET differs across calls  -> kernel nondeterministic, changes
    which keys attention sees => THE minimal repro.
  - only ORDER differs                  -> kernel emits nondeterministic
    ordering; whether that propagates depends on flash_mla reduction.
  - nothing differs                     -> kernel deterministic in isolation;
    suspect upstream scores or cross-stream interaction => re-plan.
"""
import hashlib
import itertools
import time

import torch
from cudnn import DSA as dsa

torch.cuda.init()
dev = torch.device("cuda:0")
TOP_K = 2048
N_CALLS = 20


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def make_scores(n_rows, sk, content, gen):
    if content == "random":
        # trainer scores come out of a bf16 GEMM pipeline in fp32; mimic
        s = torch.randn(n_rows, sk, generator=gen, device=dev, dtype=torch.float32)
        s = s.to(torch.bfloat16).to(torch.float32)
    elif content == "ties":
        # heavy quantization -> large exact-tie plateaus around the k-th value
        s = torch.randn(n_rows, sk, generator=gen, device=dev, dtype=torch.float32)
        s = (s * 2).round() / 2  # ~13 distinct levels over +-3 sigma
    elif content == "zeros":
        s = torch.zeros(n_rows, sk, device=dev, dtype=torch.float32)
    else:
        raise ValueError(content)
    return s.contiguous()


def run_config(sk, content, n_rows=4096):
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    scores = make_scores(n_rows, sk, content, gen)
    # causal-style per-row candidate lengths spanning the >2048 regime
    seq_lens = torch.linspace(1, sk, n_rows, device=dev).round().to(torch.int32).clamp(1, sk).contiguous()
    scores_ref = scores.clone()

    idx_runs = []
    for _ in range(N_CALLS):
        out = dsa.indexer_top_k_wrapper(scores, seq_lens, top_k=TOP_K, next_n=1, return_val=False)
        torch.cuda.synchronize()
        idx_runs.append(out["indices"].clone())

    assert torch.equal(scores, scores_ref), "input scores MUTATED by kernel!"

    ref_u = idx_runs[0]
    ref_s = torch.sort(ref_u, dim=-1).values
    n_order_diff = 0
    n_set_diff = 0
    set_diff_rows = set()
    for r in idx_runs[1:]:
        if not torch.equal(r, ref_u):
            n_order_diff += 1
        rs = torch.sort(r, dim=-1).values
        if not torch.equal(rs, ref_s):
            n_set_diff += 1
            rows = (rs != ref_s).any(dim=-1).nonzero().flatten()
            set_diff_rows.update(rows[:50].tolist())

    tag = f"sk={sk:6d} content={content:6s} rows={n_rows}"
    print(f"{tag}: order-diff {n_order_diff}/{N_CALLS-1}  SET-diff {n_set_diff}/{N_CALLS-1}"
          + (f"  set-diff rows(sample)={sorted(set_diff_rows)[:12]}" if set_diff_rows else "")
          + f"  idx0 sha={sha(ref_u)}")

    if set_diff_rows:
        row = sorted(set_diff_rows)[0]
        a = torch.sort(idx_runs[0][row], dim=-1).values
        for r in idx_runs[1:]:
            b = torch.sort(r[row], dim=-1).values
            if not torch.equal(a, b):
                only_a = sorted(set(a.tolist()) - set(b.tolist()))[:8]
                only_b = sorted(set(b.tolist()) - set(a.tolist()))[:8]
                print(f"    example row {row} (seq_len={int(seq_lens[row])}): "
                      f"only-in-run0 {only_a}  only-in-other {only_b}")
                sl = int(seq_lens[row])
                vals_a = [round(float(scores[row, i]), 4) for i in only_a if 0 <= i < sl]
                vals_b = [round(float(scores[row, i]), 4) for i in only_b if 0 <= i < sl]
                print(f"    scores of those keys: run0-only {vals_a}  other-only {vals_b}")
                break
    return n_set_diff, n_order_diff


print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}")
print(f"top_k={TOP_K}, calls per config={N_CALLS}\n")

results = {}
for sk, content in itertools.product([2049, 4096, 12288], ["random", "ties", "zeros"]):
    results[(sk, content)] = run_config(sk, content)

print("\n=== VERDICT ===")
any_set = [k for k, v in results.items() if v[0] > 0]
any_ord = [k for k, v in results.items() if v[1] > 0]
if any_set:
    print(f"SELECTION-SET nondeterminism on fixed input: {any_set}")
elif any_ord:
    print(f"ordering-only nondeterminism on fixed input: {any_ord}")
else:
    print("fully deterministic in isolation across all configs — suspect upstream/concurrency")
