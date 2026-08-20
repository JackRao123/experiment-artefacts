# PARITY RUNBOOK — padded vs unpadded partitions (volta, for gibbs, post-first-light)

Goal: prove the pad-to-131k tail-fill is inert end-to-end — loss(padded) ==
loss(unpadded) within tolerance — before real data runs through PP2.

Driver: `../tools/parity_driver.py` (self-contained; same HTTP pattern as
profile_driver_new.py). Fixed datum set: 9 mixed-length datums (8k–64k, all
non-16-aligned except one), 262,032 real tokens → 3 partitions unpadded
(padded totals 92,960 / 98,704 / 70,448 → tail fills 38k / 32k / 60k when
padding is on). Each datum carries `weights` with a zero-weight span in the
middle third. **No optim_step in any leg** — weights never move, all legs see
the same model.

## Leg 1 — padding effect at fixed parallelism (CP8/PP1)

NOTE (2026-08-12, borel-ordered fallback): the reference topology is ONE NODE,
PP1/CP8/EP8/DP1 on 8 GPUs (`start_trainer.sh --num-nodes 1`) — the 16-GPU DP2
boot crashes at the cross-node DP grad all-reduce (NCCL nvlink peer-memory,
logged as its own finding). DP1 is the identity case of the topology-invariance
argument: the comparison surface is unaffected, and no DP collective exists in
either leg to confound drift.

1. Server A: CP8/PP1, default env (no BT_PACK_PAD_TO_MAX).
   `python3 parity_driver.py --label pp1-unpadded`
2. Server B: same config + `BT_PACK_PAD_TO_MAX=1` in the server env.
   `python3 parity_driver.py --label pp1-padded`
3. `python3 parity_driver.py --compare \
   /root/.cache/user_artifacts/lps1062_bench/parity_pp1-unpadded.json \
   /root/.cache/user_artifacts/lps1062_bench/parity_pp1-padded.json`

## Leg 2 — PP effect under padding (the real PP2 gate)

WEIGHTS VALIDITY (load-bearing): the PP gate is only valid between FRESH boots
(LoRA B is zero-init, so a fresh boot's forward == the base model exactly).
The original leg A JSON (parity_pp2-padded.json) was collected on the sweep
trainer at STEP 18 — its adapter had moved; do NOT compare it to B/C at the
tight bars. After the PP2 relaunch (clean init_trainer_server), collect a
fresh leg: `parity_driver.py --label pp2-padded-fresh`, then compare against
pp1-padded.

1. Server C: CP8/PP2 on the MERGED branch (gibbs's + volta's), FRESH boot.
   PP2 pads by gate, no env needed.
   `python3 parity_driver.py --label pp2-padded-fresh`
2. `python3 parity_driver.py --compare \
   /root/.cache/user_artifacts/lps1062_bench/parity_pp1-padded.json \
   /root/.cache/user_artifacts/lps1062_bench/parity_pp2-padded.json`

## Pass criteria

- `loss`: rel diff ≤ 1e-6 (fp32 all-reduced token-weighted mean — tight).
- per-token `logprobs` in loss_fn_outputs: max abs diff ≤ 1e-3 (bf16 forward
  through a different THD row layout; kernel reduction order drifts).
- Datum count / lengths in the outputs must match EXACTLY (the script checks;
  a mismatch means datum accounting broke — see suspect (a) below).

## If it fails — suspect order (from PACKING_MEMO)

1. A code path reading `cu_seqlens_padded` as if it were real boundaries.
2. `max_seqlen` heuristics in the DSA kernel (tail-fill makes it 131072 for
   every partition — the memo's open watch item).
3. Loss-mask sentinel drift on the tail (weights/advantages not zeroed).

## Notes

- Expected wall clock: each leg is one forward_backward over 262k real tokens
  (padded leg processes 393,216 tokens = 3×131072 — ~1.5× the unpadded work on
  this datum set). No tracing; this is correctness, not perf.
- The padded leg's `fb_elapsed_s` vs the unpadded leg's is a free byproduct —
  a first direct measurement of the padding waste on a realistic mix (memo
  Part 3 estimated ~8-12% on the customer histogram; this synthetic mix is
  heavier-padded by design).
- If the box is memory-tight at PP2, remember the merged branch makes the boot
  warmup a full-size 131k fwd-bwd; `BT_SKIP_WARMUP=1` is the lever.
