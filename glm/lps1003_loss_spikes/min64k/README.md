# min64k/ — minimal-config repro attempt: cp=1 (2026-08-01, run 0731_234442)

**Question (Jack):** can the LPS-1003 rep0 destruction be reproduced on a
much smaller f — no CP, pp8/ep2 @ 64k, synthetic repetitive data, 5 fresh
`/forward` reps at the boot window?

**Answer: NO. 5/5 reps clean. cp>1 is part of minimal f.**

## Setup

- Devbox `tj-3y0gjkq` (2×8 B300), same venv/stack as the proven cp16 repro.
- `trainer-config.min64k.json`: tp1 / **pp8 / ep2 / cp1** (→ dp2), etp1,
  max_seq_len 65536, flash, lora32. pp8 works on 78 layers via the explicit
  uneven layout in `_glm52_dsa.py` `_GLM52_DSA_PIPELINE_LAYOUTS[(78, 8)]`
  (repo also ships `(78, 16)`; pp16 needs 32 ranks).
- Prod-faithful env: `parity/run_trainer_node_prodenv.sh` (conn UNSET) +
  `BT_SKIP_FULL_WARMUP=1` — the exact trigger conditions that fire 10/10 on
  the cp16 golden config.
- Payload `payload_synth2x50k.json`: 2 datums × 50,000 tokens, prefix 32,
  one per dp rank (builder: `build_synth.py`, seeds fixed):
  - datum 0: "ABCDEFEDCBA" palindrome text repeated (Jack's literal ask);
  - datum 1: one random 256-token chunk tiled to 50k (position-sensitive —
    pure repetition can mask wrong-KV reads because wrong-position content
    is identical content).
- Driver `min64k_run.sh` (mirrors parity/devbox_d2_prodenv.sh dispatch):
  boot → 5 × `/forward` of the identical payload at `/health`, full
  per-position logprobs dumped per rep. No optim steps.

## Result (boot healthy at 777s; reps ~97s each)

| datum | rep0 | rep1 | rep2 | rep3 | rep4 | max&#124;ΔNLL&#124; vs rep0 |
|---|---|---|---|---|---|---|
| palindrome | 0.000115 | 0.000121 | 0.000113 | 0.000117 | 0.000131 | 0.000016 |
| random-tile | 0.16747 | 0.16750 | 0.16429 | 0.16274 | 0.16192 | 0.0056 |

Proven bug signature = rep0 destroyed at 5–11 nats per datum, healed from
rep1. Worst delta here is 0.0056 nats — 3 orders below — and rep0 is NOT an
outlier: pairwise per-token divergence (>1 nat) is 15–165 tokens for EVERY
rep pair (rep2-vs-rep3 = 15; rep0-vs-rep4 = 165), concentrated at two fixed
intra-chunk offsets (213 and 102 mod 256 — near-tie tokens). Means drift
monotonically downward with reps 2–4 agreeing tightest → warm-up + ambient
numeric wobble, not the window bug.

## Why this is the expected structural outcome

At cp=1 the trainer never takes the packed-THD multi-doc path: THD packing
is gated on `context_parallel_size > 1` in `megatron_controller.py`
(`thd_cp = packed_seq_params is not None`); cp1 goes through padded-BSH
`_pack_batch`. The proven corruption enters DSAttention only in
second-zigzag-chunk bins — geometry that exists only under CP.

## Minimization state

- cp1 (this run): CLEAN → CP is load-bearing.
- Next smallest candidate: **cp2** — smallest THD-zigzag-preserving config
  (fits a single node: e.g. tp1/pp1/cp2, ep to fit, multi-doc packed batch).
- Data note: synthetic data has NOT yet been tested on a cp>1 config; if a
  cp2 boot with synthetic multi-doc data fires, both the data and the
  hardware footprint minimize together.

## Files

| file | role |
|---|---|
| `trainer-config.min64k.json` | the cp1/pp8/ep2 trainer config |
| `build_synth.py` | payload builder (run on devbox; tokenizer from HF cache) |
| `payload_synth2x50k.json` | exact submitted /forward body (2×50k datums) |
| `min64k_run.sh` | boot + 5-rep window probe driver |
| `compare_reps.py` | per-datum NLL table + per-position bands + delta dump |
| `min64k_0731_234442/` | run: driver.log, rep{0..4} full logprobs (.json.gz), `positional_deltas.json.gz` |

Devbox originals: `/root/.cache/user_artifacts/lps1003/min64k/`. Devbox left
idle after the run (squeue empty, GPUs 0 MiB — verified).
