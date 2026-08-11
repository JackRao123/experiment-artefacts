# HANDOFF — MoE A2A⇄compute overlap design owner (helmholtz → minkowski)

**Date:** 2026-08-09 · **Ticket:** LPS-1062 (GLM-5.2-FP8 perf, 2×8 B300,
golden TP1/PP1/EP16/CP16, full recompute, attention-only LoRA r32)
**Successor briefing order:** read `DESIGN_helmholtz.md` (the memo) §0–§2
first, then this file's artifact table, then the invariants. The memo's §6.1
field-calibration chain is the record of how the win models were corrected
by on-box measurement — read it before quoting any wall-second number.

---

## 1. What this workstream is

Three levers against the ~26 s/step of exposed EP all-to-all time in the
46.8 s step (all env-gated, default OFF, bitwise-parity-tested, WARNING-level
armed/hit telemetry — the house standard set by the FIX-B/F project):

- **W1** — probs A2A on a second NCCL communicator (de-serialize it from the
  tokens A2A's stream). **SHIPPED as v1.**
- **W2** — intra-layer chunked pipeline: K=2 local-expert groups, list-form
  A2A over per-peer views (zero copies, byte-identical combine buffer).
  **v3 patch re-queued for the T2 gate** (behind C-prime + CDMC arms).
- **W3** — lookahead recompute: overlap recompute-fwd(L−1) with bwd(L)
  across layers inside full recompute. **Staged for tonight's canary.**

Parked: W2-v2 (backward seq-bump pipelining) and option 6 (backward-chain
reorder for the probs reverse) — see `W2V2_DECISION_stub.md` for the
ratified un-park conditions. Dead-and-archived: W1-v2 paired dispatch node
(its pre-registered ship-gate resolved negative with data — keep it dead
unless the stub un-parks).

## 2. Artifact table (exact state)

All under `experiment_artefacts/glm/lps_1062_perf/runs/overnight_20260810_round3/overlap/`.

| Artifact | State | Verification |
|---|---|---|
| `DESIGN_helmholtz.md` | current; §6.1 carries the full field-calibration chain (tail-bound retracted → CDMC=1 retracted → engine-issue-order → grad-arrival resolved by hilbert's cut) | reviewed+approved by fibonacci; V2b/V4 PASS recorded in §12 |
| `runs/overnight_20260810_round3/overlap/patches/w1-probs-a2a.patch` + `W1_PATCH_NOTES.md` | **SHIPPED as v1**; on-box mechanism confirmed (probs off-stream 900/900, gap −2.07 ms/pass); wall −0.46 s (residual understood, bounded, documented in the notes' final disposition) | T1 gloo suite green; T3 canary clean |
| `runs/overnight_20260810_round3/overlap/patches/quarantine/w1v2-paired-dispatch.patch` + `runs/overnight_20260810_round3/overlap/patches/quarantine/W1V2_PATCH_NOTES.md` | **ARCHIVED-UNSHIPPED — do not apply.** The notes lead with the negative gate verdict (hilbert's cut: launch ~24 ms early on an idle stream; kernel waits ~41 ms for the probs grad produced post-fc2-dgrad). | stopped-variant tests preserved (skip-guarded) |
| `runs/overnight_20260810_round3/overlap/patches/w2-moe-a2a-pipeline.patch` + `W2_PATCH_NOTES.md` | **v3, re-queued for the T2 gate.** Includes the T2-gate defect fix (probs selection now uses the unfused split/cat — the TE fused sort's generated backward is permutation-only) + hilbert's round-1/2 fixes (send-side combine bounds, total_rows, FIX-C slots, plan-carried work handles, had_buf guard, disarm flag, overlap_moe_expert_parallel_comm fallback, FIX-C stats assertion, TE-version notes). | T1 (72) + T1-seed (hilbert) + T1b (hilbert) + probs-selection guard — ALL PASS; patch verified apply-clean + byte-for-byte |
| `runs/overnight_20260810_round3/overlap/patches/w3-lookahead-recompute.patch` + `W3_PATCH_NOTES.md` | **Mac-side complete, staged for tonight's canary.** New `megatron/core/lookahead_checkpoint.py` + recompute.py chunk_runner branch. | Mac CPU suite ALL PASS incl. the dropout=0.5 RNG-isolation bitwise proof |
| `W2V2_DECISION_stub.md` | parked; un-park rules ratified by fibonacci; option 6 folded | — |
| `tests/` | all green on the ship tree (see §4 for the roster) | Mac CPU (gloo); on-box gates are T2/T3 |
| `HANDOFF_ORCHESTRATOR.md`, `LEVERBOARD.md`, `TRACE_ACCEPTANCE.md` | fibonacci's (orchestrator) — read for the box queue + acceptance bars | — |

## 3. Test roster (`tests/`) and what each proves

- `test_w1_probs_a2a.py` — W1 deferred-wait split, second-comm equivalence,
  backward parity, gate/telemetry (gloo, 2 procs).
- `test_w2_chunk_plan.py` — W2 decomposition index math (72 checks; combine
  reassembly == P byte-for-byte; zero-count/imbalance/degenerate cases).
- `test_w2_chunk_plan_t1_seed.py` — hilbert's seed suite (EP3/nle6/K3, NCCL
  pairwise consistency, degenerate all-empty group). Named hard precondition.
- `test_w2_chunked_a2a_functions.py` — hilbert's T1b: the real
  `_ChunkedDispatchA2A`/`_ChunkedCombineA2A` on 4 gloo procs vs monolithic
  references.
- `test_w2_probs_selection_backward.py` — the T2-defect guard: subset
  selection's backward is full-size + exact; the permutation invariant the
  fused calls depend on; grad-numel contract probe.
- `test_w3_lookahead_checkpoint.py` — W3 kick/skip/fallback + dropout>0 RNG
  isolation + eviction discipline.
- `t2_w2_numerics_gate.py` — ON-BOX V1 gate (torch.equal on outputs AND
  input grads; balanced/imbalance/zero-(peer,group)/zero-expert; FIX-C
  marked variants with stats assertions). Needs CUDA/NCCL/TE.
- `test_w1v2_paired_dispatch_STOPPED.py`, `w1v2_probe*.py` — preserved
  stopped-variant artifacts (skip cleanly on the ship tree).

Run everything with the server venv python
(`server/.venv/bin/python`); `BT_TEST_MCORE_PATH` overrides the tree
walk-up if the files are copied elsewhere.

## 4. Design invariants — DO NOT BREAK (each is a paid-for lesson)

1. **Combine-side view offsets are SEND-side.** The combine output mirrors
   the permuted buffer P's layout (per-source blocks of the rows THIS rank
   sent); bounds derive from the local (send) counts matrix, never the
   receive matrix. (hilbert BUG 1; `test_w2_chunk_plan.py` §4 is the
   regression net.)
2. **TE fused sort_chunks is a PERMUTATION-ONLY contract** (output rows ==
   input rows, every chunk covered once). Never use it as a subset
   selection — its generated backward returns the wrong-shaped grad
   (the T2-gate defect). For selections: the unfused split/cat path or
   index_select. Guard: `test_w2_probs_selection_backward.py`.
3. **State lives on carriers / registry / plan objects — NEVER on
   pass-through tensors.** A custom Function returning its input gets a new
   autograd alias; tensor attrs vanish in grad-enabled passes (hilbert BUG
   A). W1's deferred-work handle is safe only because
   `_AllToAllDeferredWait` returns a freshly-created tensor. W3 carries
   everything on the packed_seq_params carrier registry.
4. **Push-order rule:** with CUDA_DEVICE_MAX_CONNECTIONS=1 (devbox benches
   only — production runs unset), any compute-stream wait head-of-line
   blocks everything pushed after it in the single channel. All independent
   work (A2A issues on every comm, shared-expert kernels) must be pushed
   BEFORE the first wait. (The existing shared-expert fc1/fc2 placement is
   the template; W1/W2 preserve it explicitly.)
5. **Gates are env-gated, default OFF, and must prove they fired**:
   WARNING-level armed/disabled line + per-window counters. A gate that
   can't fire passes parity silently (the v1 lesson; every gate in these
   patches has armed/hit telemetry).
6. **Bitwise parity standard:** torch.equal on outputs AND input grads,
   including engineered imbalance and zero-count cases; the multi-step loss
   canary covers drift (e.g. FP8 amax history) that single-step gates can't.
7. **Recompute determinism:** the replay must be bitwise-identical to the
   first pass (FIX C's premise; W2's cache slots; W3's kicked recompute all
   depend on it). If you touch the router/recompute path, re-verify against
   `BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1`.
8. **One host sync per layer-pass.** All split/offset metadata derives from
   the single D2H batch (the two [*,16] count matrices ride it for W2). No
   new per-pass `.item()`/`.tolist()` on device tensors — host lists come
   from the batch or are precomputed at arm time.
9. **FIX C pass-marker keying** on the packed_seq_params carrier
   (recompute.py's `_CheckpointChunkPassMarker`); any new replay-dependent
   state must key the same way (per-microbatch lifetime, thread-local
   frames).

## 5. Open design questions for the successor

1. **W3 canary readout** (the immediate duty — watch-list in §6): capture
   ratio vs the 60–80 % model; memory at 16k×d32 (the +8–12 GiB question);
   whether the dispatcher replay eventSyncs stall the kick thread without
   FIX C on (the §7c synergy — pair them).
2. **W2-v2 / option 6** — un-park only per the stub's rules after the W3
   canary (W3-absence fallback).
3. **De-skew upside** (a2a p50 10.7 ms vs 4.3 ms floor) — unmeasured; the
   chunked earlier-posting may compress it; quantify from the W2 canary
   trace.
4. **K=4** — parked behind the same gate for one experiment; only if p50
   stays ≫2× floor after K=2.
5. **CUDA-graph support** for W2/W3 (both currently fall back); the
   cudagraph_attrs path needs the chunked state added.
6. **Zero-padded grouped GEMM overhead** — 8 empty problems per chunk call;
   if the canary's launch budget objects, the fallback is per-chunk 8-gemm
   module shells sharing weights (the `_make_fused_ops` pattern,
   experts.py:427-437).
7. **W3 fp8-recipe generality** — validated for blockwise (stateless); a
   delayed-scaling config would need the TE-checkpoint parity audit (the
   Function mirrors CheckpointWithoutOutputFunction's fp8 branches; TE's own
   checkpoint may do more).
8. **W3 thread model** — the kick is single-autograd-thread by design; if
   mcore ever runs multi-threaded backwards, re-audit the RNG fork/restore
   and the carrier dict access.

## 6. W3 canary watch-list (the next design-owner duty)

When the W3 canary runs (tonight's ladder, after W2's gate/arm), check:

- **Telemetry:** one `armed — first checkpointed chunk registered` line;
  per-window counters with **kicks == stash_hits == chunks−1 and
  stash_misses == 1 per microbatch** (the last chunk recomputes inline by
  design); **sweeps == 0** (any non-zero sweep = structural bug — the
  registry leaked; investigate before accepting). fallbacks == 0 with a
  carrier present.
- **Memory:** high-water vs the model: +2–4 GiB at 131k (one extra layer of
  live recompute activations); at 16k×d32 expect +8–12 GiB — if it exceeds
  headroom, gate W3 off per-shape and file it.
- **Perf:** bwd-phase comm overlap % (target ≥ 50 %); step time vs the
  31–34 s stacked model — BUT read §6.1 first: wall conversion on this box
  has already surprised once; the acceptance bar is mechanism-first
  (overlap %, SendRecv time in bwd-phase windows), wall second.
- **Correctness:** loss bitwise vs gates-off canary (the design claims
  exactness — dropout is 0 in prod and asserted at gate-on; the dropout>0
  mechanism proof is in the CPU suite); `BT_MOE_DISPATCH_REPLAY_CACHE`
  pairing recommended (removes the replay eventSync stalls from the kick
  path).
- **Failure modes to recognize:** fallback WARNING lines (carrier absent,
  dropout≠0 assert, cuda-graph capture passthrough); a stash_misses count
  ≫ 1 per microbatch (kick chain broken); any sweep.

## 7. Contact points

- Orchestrator succession: fibonacci → **kepler** (see
  `HANDOFF_ORCHESTRATOR.md`).
- Box runner: bohr. Second reviewer: hilbert. My successor: **minkowski**
  (this file + the memo are your briefing; the probes and stopped-variant
  tests in `tests/` are reusable evidence artifacts).
