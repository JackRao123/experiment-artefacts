# PRE-DRAFTED PR BODY — F2 phantom partitions (trainers)

**Repo:** basetenlabs/trainers · **Branch:** `jackrao/lps-1062-f2-main-rebase`
(rebased onto main; see runs/overnight_20260810_round3/f2/REBASE_NOTES.md) · **Base:** main
**STATUS: OPENED AS DRAFT 2026-08-10 — basetenlabs/trainers#1001** (open
condition met: (a) PASS, (a3) PASS, (b) PASS, (c1) vacuous-by-construction
recorded, (c2) PASS max |dloss| 1.65e-3/21 windows). The opened body is the
filled version of this draft (evidence slots + honest scope line + the
main-layout test port f4f5d332); this file is kept as the pre-draft record.

---

## What

Fix the DP>1 THD-CP deadlock by equalizing the per-replica partition count
and padding short replicas with **phantom partitions** (env-gated
`BT_F2_PHANTOM_PARTITIONS`, default ON-with-flag; kill switch
`BT_F2_PHANTOM_PARTITIONS=0` reproduces the deadlock on demand).

**Problem.** The controller's per-partition loop carries an explicit
invariant — "no data-parallel collective may run inside this loop" — that
covers only the collectives the *controller* issues. The *model* issues MoE
expert-parallel collectives per partition (per-MoE-layer `all_gather` on the
TP-EP group + dispatch/combine A2As), and the EP group spans DP replicas
whenever `ep_size > ranks-per-replica` (EP16/CP8/DP2: the EP16 group covers
both replicas). THD-CP packs a **data-dependent** number of partitions per
replica (greedy bin-pack of each replica's datum slice), so when replica
counts disagree — boot warmup pass-1 sends exactly 1 datum → dp_rank1 packs
**0** partitions; heterogeneous customer data → 3 vs 2 — the longer replica
enters partition N's first EP all_gather while the shorter has exited the
loop and sits in post-loop DP grad finalize: permanent NCCL deadlock,
silent (warmup lifts the pg timeout to 120 min). DP>1 was unshippable on
real THD data.

**Fix.** (1) Immediately after `partition_thd_cp_datums` in
`_pack_thd_cp_microbatches`: one `all_reduce(MAX)` of the local partition
count over the **pure-DP group** — once per `forward_backward` op, on every
rank, at the same point in the op sequence (no new desync vector; one 4-byte
reduce per op). (2) Replicas below the max append phantom partitions as a
**strict suffix**: synthetic `pad_multiple`-token datums (16 tokens at CP8,
32 at CP16) with zero-masked loss — their contribution to loss and grads is
exactly zero.

## Why (evidence)

- **(a) boot-deadlock repro → fix: PASS.** With the patch, default env, the
  server reaches READY/health OK at EP16/CP8/DP2 (pre-fix: >35 min hang in
  warmup pass-1; py-spy: node0 in token_dispatcher all_gather, node1 in
  finalize_model_grads).
- **(a3) kill-switch control: PASS.** Reboot with
  `BT_F2_PHANTOM_PARTITIONS=0` reproduces the exact deadlock — py-spy
  signatures matched — proving the flag and the causal mechanism, not a
  fluke boot. **(a)+(a3) ⇒ causality proven.**
- Evidence files: box-2 bench dir `lps1062_bench/wxlgv5w/` (fix-ON DP2 clean
  boot log; flag-OFF deadlock repro log + py-spy dumps); fleet scorecard
  row: [runs/overnight_20260810_round3/overlap/SCORECARD_MORNING.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/runs/overnight_20260810_round3/overlap/SCORECARD_MORNING.md).
- **(b) B-custmix heterogeneous-shape repro** (20 datums, 10,240–63,488 tok,
  customer histogram): *result pending — fills before opening.*
- **(c) Loss canary vs DP1 golden** (same total tokens, same seed-fixed
  synthetic data, matched `--warmup-datums`; bar: per-window |dloss| ≤ ~5e-3
  cross-config reduction-order tolerance, grad-norm series in band): *result
  pending — fills before opening.*

## Explicitly NOT in this PR (ship-gated follow-ups)

- **(d) perf non-regression bench** (equal-length DP2 131k×d4; phantom
  overhead at equal counts is one 4-byte all_reduce per step — expected
  unmeasurable; must be measured before ship).
- **1a/1b NaN×0 grad-corruption hardening canaries** (fibonacci's blocking
  review additions): phantom-only microbatch backward grad-finiteness/zero
  assertions + phantoms-on vs phantoms-off checksum soak. Zero loss seeds
  give exact-zero grads ONLY if no phantom intermediate goes inf/NaN
  (NaN×0 = NaN would silently corrupt accumulated real grads) — these gates
  are required before the flag defaults ON anywhere customer-facing.

## Value

Not a 2-node throughput lever (patched golden ≈ CP8/DP2 within 1 %). The
value is **correctness for any DP>1 customer** and **unlocking 4+-node
scaling** (EP16/CP8/DP4 ≈ 2× aggregate tok/s) — Jack's explicit pivot when
lever returns diminish.

## Design / review / tests

- Design: [runs/overnight_20260810_round3/f2/DESIGN_F2_bohr.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/runs/overnight_20260810_round3/f2/DESIGN_F2_bohr.md)
  (incl. the DPO two-datum phantom note and the strict-suffix argument)
- Review: [runs/overnight_20260810_round3/f2/REVIEW_F2_helmholtz.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/runs/overnight_20260810_round3/f2/REVIEW_F2_helmholtz.md)
- On-box validation recipe (the (a)–(d) + 1a/1b frames this PR's gates come
  from): [runs/overnight_20260810_round3/f2/ONBOX_VALIDATION_F2.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/runs/overnight_20260810_round3/f2/ONBOX_VALIDATION_F2.md)
- Tests: *fill from the patch's test file at open time (packing/controller
  unit coverage per the diff).*

## Status

Prepared for review — **do not merge** (ship go/no-go is Jack's). Opens only
after (b)+(c) PASS; (d) and 1a/1b are tracked ship-gated follow-ups.
