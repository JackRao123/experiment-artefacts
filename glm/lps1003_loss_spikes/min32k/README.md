# min32k/ — minimization step 2: cp16 golden config @ 32k, 2 docs (2026-08-01, run 0801_003030)

**Question (Jack):** does the bug fire on the golden cp16/ep16 leaf with only
`max_seq_len` dropped to 32768 and a 2-doc ~30k payload — i.e. is CP alone
(with the proven trigger) enough, or do prod-scale rows/packing matter?

**Answer: NO repro. 3/3 window reps clean.**

## Setup

- Devbox `tj-3y0gjkq` (2×8 B300), same stack as the 10/10 cp16 repro.
- `trainer-config.cp16s32k.json`: golden B300 leaf (tp1/pp1/**ep16/cp16**,
  dp1, flash, lora32) with **max_seq_len 32768** (golden = 262144).
- Trigger identical to the proven repro: `parity/run_trainer_node_prodenv.sh`
  (conn UNSET) + `BT_SKIP_FULL_WARMUP=1`, 3 identical `/forward` reps fired
  at `/health` (boot window). Boot healthy at 642s; reps 15s/4s/4s.
- Payload (`build_min32k.py` → `payload_min32k.json`, meta in
  `payload_min32k.meta.json`):
  - datum 0 `random-tile`: one random 256-token chunk (seed 1003100) tiled
    to 30,000 tokens, prefix 32;
  - datum 1 `mudith-b21i13`: the longest real datum ≤30k tokens in
    `train_bundle_0_31.jsonl.gz` — batch 21 idx 13, 29,985 tokens, prefix
    29,682 (303 supervised tokens, sitting at the row TAIL = the danger
    zone of the positional law).
- Note the partition structure: two ~30k docs under a 32k cap can't share a
  row → each doc is a SINGLE-DOC partition. Zigzag chunks exist (cp16), but
  there are no partition-tail *neighbour* docs like prod's multi-doc rows.
- CAVEAT (discovered 2026-08-01, next session block): /forward returns
  logprob **0.0 at every weight-0 position** — so this run's dumps only
  measure datum 0 fully and datum 1's 303-token supervised tail; the
  29,681 'prompt' entries are zero-fill, NOT measurements. Probe with
  uniform weights to record true full-position logprobs.

## Result

| datum | rep0 | rep1 | rep2 | max&#124;ΔNLL&#124; vs rep0 |
|---|---|---|---|---|
| random-tile | 0.121787 | 0.123366 | 0.121124 | 0.00158 |
| mudith-b21i13 | 0.895125 | 0.915860 | 0.902474 | 0.02074 |

Signature is rep0 destroyed at 5–11 nats, healing later; here worst
per-datum delta is 0.021 (and rep0 is the LOWEST rep, not the highest). No
rep0-outlier structure per-token either — pairwise tokens >1 nat:
random-tile 34/8/31 (rep01/02/12), ALL at intra-chunk offset 175 mod 256
(one near-tie token in the repeated chunk; max 4.99 nats — cp16 wobble is
larger than the cp1 run's, unsurprising); mudith 5/4/6 — all inside the 303-token supervised band, the only region
actually measured (see caveat above: the 'quiet prompt region' is zero-fill
artifact, not quiet). Ambient nondeterminism, not the window bug. A uniform-
weights resend of this payload on the NEXT boot measured the full doc:
~5,470/29,984 tokens >1 nat across ALL rep pairs (see x3x10k run + README
section below-dated entries in INVESTIGATION.md).

## Reading

cp16 + proven trigger + 30k single-doc partitions ≠ f. Remaining deltas vs
the 10/10 repro, in likely order of relevance:

1. **Row scale**: prod rows were 64–131k packed tokens (per-CP-rank local
   length ~8k vs ~1.9k here) — shape thresholds inside the DSA kernels
   (e.g. tile/scheduler occupancy) are not reached at 32k.
2. **Multi-doc packing**: proven events destroyed partition-TAIL docs of
   multi-doc rows; single-doc partitions may not express (or not reveal)
   the corruption the same way.
3. max_seq_len config value itself (buffer/warmup allocation shapes).

Next rung on the ladder: keep cp16, raise max_seq_len (64k+) and pack MANY
shorter docs per row (restore partition-tail structure) — synthetic first,
batch-0 as fallback (batch-0 at 262144 is already the proven 10/10 repro).

## Files

| file | role |
|---|---|
| `trainer-config.cp16s32k.json` | golden cp16/ep16 leaf @ max_seq_len 32768 |
| `boot_min32k.sh` | dispatch-only boot (prodenv + BT_SKIP_FULL_WARMUP=1) |
| `build_min32k.py` | payload builder (bundle scan + random tile) |
| `payload_min32k.json` / `.meta.json` | exact submitted /forward body + provenance |
| `min32k_0801_003030/` | probe.log, rep{0..2} full logprobs, positional_deltas.json.gz |

Analysis driver: `../min64k/compare_reps.py` (generalized: any tag,
labels arg). Devbox originals: `/root/.cache/user_artifacts/lps1003/min32k/`.
Devbox torn down after the run (squeue empty, GPUs ≤4 MiB — verified).
