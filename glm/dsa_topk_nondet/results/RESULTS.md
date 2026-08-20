# Results — localization + kernel micro-repro (2026-08-19, session kirchhoff)

All runs on tj-w5yd8r3 (B200), venv `/root/.cache/user_artifacts/trainers_main/server/.venv`
(torch 2.11.0+cu128), trainer at exact nightly pins f93eb052 / 20fcf2ea / 57efae08.

## 1. Fingerprint localization run (probe `--repeats 3 --arms padded-random-big`)

Probe result (failure fired during the instrumented run):

```
{"arm": "padded-random-big", "per_datum_max_spread": [1.797923, 0.0, 1.92671, 1.8382, 0.0, 0.0],
 "max_spread": 1.9267101287841797, "bitwise_identical": false}
```

Fingerprint analysis (`analyze_fprints.py`; raw jsonl in `../fprints_localization_run/`;
78 records/rank = 26/repeat: 5 dsa_topk + 12 dsa_sparse_fwd + 9 moe_router):

```
rank0: FIRST DIVERGENCE global call 0 = dsa_topk call #0   (indices sha: b4b679da | b4b679da | 04bf7426)
rank1: FIRST DIVERGENCE global call 2 = dsa_topk call #1
rank2: FIRST DIVERGENCE global call 0 = dsa_topk call #0
rank3: FIRST DIVERGENCE global call 0 = dsa_topk call #0
verdict: first-divergence site = dsa_topk on every rank; dsa_sparse_fwd and moe_router
diverge only at later global call indices (downstream propagation).
```

Note: the `out_flat` field of dsa_sparse_fwd records is a constant `ERR:TypeError`
(bf16 tensor → numpy in the hasher); the `lse` hash carried that site's comparison.
dsa_topk hashes (indices + length) are valid.

## 2. Kernel micro-repro (`topk_micro_repro.py`) — THE minimal repro

Single GPU, no model, seconds. Fixed fp32 scores tensor [4096 rows × sk], causal-style
seq_lens 1..sk, `cudnn.DSA.indexer_top_k_wrapper(scores, seq_lens, top_k=2048, next_n=1,
return_val=False)` called 20× on the SAME tensors (input verified unmutated bitwise):

```
sk=  2049 random: order-diff 19/19  SET-diff  0/19
sk=  2049 ties  : order-diff 19/19  SET-diff  0/19
sk=  2049 zeros : order-diff 19/19  SET-diff 19/19  (only row with seq_len>2048)
sk=  4096 random: order-diff 19/19  SET-diff 19/19  divergent rows start at seq_len≈2049
sk=  4096 ties  : order-diff 19/19  SET-diff 19/19
sk=  4096 zeros : order-diff 19/19  SET-diff 19/19
sk= 12288 random: order-diff 19/19  SET-diff 19/19
sk= 12288 ties  : order-diff 19/19  SET-diff 19/19
sk= 12288 zeros : order-diff 19/19  SET-diff 19/19
```

Every observed set swap exchanges keys with **bitwise-identical scores** (e.g. run0-only
key score -2.3594 vs other-run key score -2.3594): a tie-breaking race at the k-th
boundary. "random" content ties because scores are bf16-quantized (≈256 distinct values
per octave) — exactly the trainer regime. Rows with seq_len ≤ 2048 never diverge.
Output ORDER is nondeterministic in all configs (radix select emits arbitrary order).

## 3. Deterministic fallback A/B (`topk_fallback_ab.py`)

Masked `torch.topk` path (the odd-K fallback body from `_indexer_top_k_one_chunk`,
dsa_cudnn_kernels.py:467), same fixed inputs, 20×:

```
sk=4096 : torch.topk order-diff 0/19  SET-diff 0/19   cudnn 0.045 ms/call  torch 0.461 ms/call  (10.2x)
sk=12288: torch.topk order-diff 0/19  SET-diff 0/19   cudnn 0.107 ms/call  torch 1.120 ms/call  (10.5x)
```

Bitwise deterministic, ~10× slower per call, ~1 ms absolute at gate shapes (4096 rows).

## 4. Existing dense tie-break bias is INERT for real content

`_TOPK_TIE_BREAK_EPS = 1e-12` spread over sk columns → per-column step ~8e-17, below
fp32 ulp (~2e-7 at |score|≈2). Empirical (sk=4096, bias applied then cudnn kernel 20×):

```
random: bias changed 0.00% of elements; SET-diff 19/19  (bias rounds away → no fix)
zeros : bias changed 99.98% of elements; SET-diff 0/19  (bias survives near 0 → fixes)
```

So enabling `_use_dense_indexer_topk_tie_break` on CUDA would NOT fix the gate failure.
