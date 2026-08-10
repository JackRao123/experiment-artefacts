# Verification handoff: hilbert → boltzmann (2026-08-09 night)

You (boltzmann) inherit the LPS-1062 adversarial-verification estate: the trace
acceptance checker, the straggler analyzer, the FIX C/C′ evidence chain, the
FIX A-v3 package, and the review standards that caught tonight's bugs. This file
+ the orchestrator handoff (`overlap_design/HANDOFF_ORCHESTRATOR.md`) +
`overlap_design/TRACE_ACCEPTANCE.md` are your authoritative state.

**Immediate duty:** verify the timed C′ capture when fourier lands it (ARM:
timed C′ arm, in flight at handoff). Acceptance below (§3).

## 0. How to run things (the two venvs)

- Trace analysis (perfetto/pandas/numpy):
  `/var/folders/1m/bllgmvfs6t7czgc4w3l_h7f00000gn/T/opencode/perf-venv/bin/python`
  — trace_processor **v56.1 is checker-authoritative** (see §1).
- Mac-CPU test suites (torch 2.11 CPU):
  `/var/folders/1m/bllgmvfs6t7czgc4w3l_h7f00000gn/T/opencode/fixc-test/bin/python`
  — run every suite with `BT_TEST_MCORE_PATH=<tree>` pointing at the vendored
  mcore under test (default walk-up finds the shared tree).

Test suites live in `dispatcher_opt/tests/` (A/B/F), `dispatcher_opt/fixc/`
(FIX C + C′), `dispatcher_opt/fixa_v3/` (A-v3), `overlap_design/tests/` (W1/W2).

## 1. check_acceptance.py (dispatcher_opt/) — profiles + calibration state

Mechanical A/B acceptance checker over one-step kineto captures. Usage:
`python check_acceptance.py TRACE --profile PROFILE [--set metric.bound=v]
[--json out.json]`. `--json` dumps ALL measured metrics (calibration tool).

**The v56.1 rule (binding):** trace_processor v56.1 is authoritative. v≥57
keeps ~5,405 overlapping complete events v56.1 drops (kernel_count shifts by
that amount). The baseline tolerance absorbs it; where two numbers exist, the
checker's own measurement on the baseline trace is authoritative.

**Profiles and their calibration state:**
- `baseline-exp05d`, `baseline-4mb131k` — measured-calibrated unpatched
  references (counts exact, times ±2-3%; the 4mb one carries the M-invariance
  scaling-law lessons in its header).
- `post-patch-A/-B/-AB/-ABF(+4mb variant)` — the round-2 gates; ABF-4mb is
  validated ALL PASS on gated-v2-4mb-steady.
- `post-patch-BFC-4mb131k` — FIX C/C′ ship config (B+F+C, A parked). Sharp
  rows: `eventsync_dispatcher_replay_calls` 300→~0; `eventsync_dispatcher_fwd_calls`
  300 unchanged; `dispatcher_allgather_calls` 600→300 (GPU-side Long AllGather
  on EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP — reads exactly 600 on the
  baseline; the CPU-side `_GatherFromSequenceParallelRegion` walk CONFLATES
  DSA CP gathers at 696/696 — do not use it).
- `post-patch-W1-4mb131k` — W1 (probs a2a on a second comm). Sharp rows:
  `probs_a2a_off_token_stream` ==900; `dispatch_combine_gap_avg_ms` ≤7.0;
  `token_a2a_gpu_s` 20.90±5%; `step_wall_s` ≤43.18.
- `post-patch-W2-4mb131k` — W2 v1 chunked pipeline. Sharp rows:
  `token_a2a_calls` ~3600 (2×); `token_a2a_in_nelems_p50` 150–280M (half
  payload); `a2a_compute_overlap_pct` ≥15; `step_wall_s` ≤44.68.

**The A-off re-seating (ratified):** FIX A is PARKED (default-off). The
BFC/W1/W2 profiles carry the A-off reality rows: the 312 DSA-bwd nonzero
drains remain (`nonzero_calls` ≤700, `nonzero_cpu_s` ≤21.0,
`nonzero_gt5ms` ≤330, `streamsync_calls` ≤1000, `streamsync_cpu_s` ≤21.0).
The ABF profile keeps the tight A-on rows. If A-v3 ever ships, re-tighten.

**Metric definitions validated against the baseline trace (reproduce
laplace's published numbers exactly):** the dispatch→combine gap uses
consecutive-pairing of token kernels on the comm stream (reproduces
10.85/16.64 ms); a2a↔compute overlap = union-intersection of SendRecv
intervals with non-SendRecv kernel intervals, as a fraction of total SendRecv
residency (reproduces 0.395 s / 1.5%).

## 2. straggler_analysis.py (overlap_design/) — two-rank skew

Usage: `python straggler_analysis.py RANK_A.json RANK_B.json --label-a rank0
--label-b rank8 --out summary.md`; `--self-test` (same trace twice) must give
~0 skew. Deps pinned+guarded in the header (`perfetto==56.1` pandas numpy).

**The median-centering rationale (binding):** cross-host CPU clocks are not
comparable — the raw signed skew carries a constant artifact (~127 ms on the
rank0/rank8 pair). The PRIMARY table is median-centered (per-series);
§1b is the duration-floor sanity check (a genuine X ms posting skew would
inflate the waiting rank's a2a durations to ≥X ms; ~4.3 ms floors on both
ranks rule out the constant); §1c demotes the raw offsets to a caveated
footnote. The hot-expert test (Spearman + hit rate) median-centers end deltas
(constant offsets are not straggler signal; correlations are shift-invariant,
hit rates are not).

## 3. FIX C / C′ evidence chain + timed-arm acceptance

The chain (all artefacts under `dispatcher_opt/fixc/`):
1. ARM-1 verify failure (deterministic 2/2): `num_tokens_per_local_expert_dev`
   mismatch at routing_shape (8192,256) — the replay's routing_map is not
   bitwise vs the first pass on this stack.
2. Instrumentation (`fixc_verify_instrumentation.patch`, regenerated against
   the full w1/w2 stack): per-field mismatch forensics + routing_map XOR.
3. Probe (`probe_gradmode_divergence.py`): GEMM grad-mode mechanism REFUTED
   (all four GEMM rows torch.equal); the cuDNN DSA attention forward is the
   remaining candidate (kill-shot case 5 queued for an idle box — design-doc
   completeness, not a blocker).
4. One-mechanism synthesis (CONFIRMED, zero counterexamples): integer counts
   can't be order-nondeterministic; `input_splits_dev` is the ONLY purely-local
   field (num_tokens_per_local_expert_dev sums the gathered matrix —
   allgather-derived); a peer flip shows up locally as "XOR: 0 flips yet
   output_splits differ by 1". Flip statistics: 1 row/raise, 2 flips (a paired
   ±1 swap), max|diff|=1.
5. C′ (`fixc_prime_routing_force.patch` + `test_fixc_prime_routing_force.py`
   24 checks + `DESIGN_FIXC_PRIME.md`): save only the routing_map on the
   carrier; replay masks its recomputed logits with −inf on unsaved pairs →
   selection forced to the saved set (bitwise), probs recompute
   grad-connected (the activation-grad term preserved — the rejected
   detached-probs variant would have dropped it). Soak already GREEN on-box
   (20/20, zero raises, 6300/6300 hits, +3.7 GiB cache, canary in band).

**Timed C′-arm acceptance (your immediate duty):** trace →
`--profile post-patch-BFC-4mb131k` must PASS (replay eventsync ~0, fwd 300,
allgather 600→300, B+F invariant rows); log → the C′ window lines show
`stashes=312, forces=312, misses=0` per step, `verify_asserts=312/step` with
zero raises, and the in-flight stash bytes PLATEAU (~600 MB peak, no growth
across steps = no leak); canary in band at identical `--warmup-datums`; wall
delta is informational (compounding value, no hard bar).

## 4. FIX A-v3 package (conditional ARM 5)

`dispatcher_opt/fixa_v3/`: `fixa_v3.patch` (first-pass-anchored probe;
gate `BT_DSA_BWD_ASYNC_NONEMPTY_V3`, verify mode `..._VERIFY`),
`test_fixa_v3_probe.py` (33 checks), `DESIGN_FIXA_v3.md`, `ONBOX_VALIDATION.md`.
**Conditional slot: only after C′ timed passes** — v3 measures whether the
312 DSA-bwd drains become exposed once the dispatcher's replay throttle is
gone. A ~0 result is a valid publishable answer (drains still overlapped).
The checker's `eventsync_a_*` rows split A-flag events by the
FusedSparseAttentionFuncBackward parent (312 calls, µs-scale, ≤1.5 s, ≤25
>1 ms for a healthy A-active capture).

## 5. The falsification-cascade method (house method, worth preserving)

When a mechanism is theorized, run the cheap discriminators in order and let
each falsification narrow the space — the MEASUREMENT stands while mechanisms
fall. Tonight's W1-residual cascade (all from one capture): channel
head-of-line falsified by CDMC-unset (bohr's /proc proof); engine issue order
falsified by the EARLY launch (p50 −24.5 ms before the paired dispatch);
comm2 queueing falsified by the idle stream (previous probs kernel ended
~48 ms prior); peer-tail falsified by corr(probs_dur, token_dur)=0.075. The
surviving gate: the comm stream's input-readiness wait on the compute stream
(the probs-grad production point in the backward chain) — kernel launches
early, starts +41 ms later. Consequence: the W1-v2 seq-bump was correctly
stopped (launch already early); the bwd-reorder lever parked as a W3-absence
fallback. The general form: pair every kernel with its launch via
args.correlation; launch-vs-start separates CPU-side from GPU-side gating;
duration-correlation with a coupled collective separates peer-bound from
launch-side.

## 6. Review standards — the five bug classes caught tonight

Apply these to every diff you review (they are all silent-corruption or
silent-inertness classes that parity-green hides):

1. **Symmetric-fixture blindness.** The W2 chunk-plan combine_recv_bounds was
   computed from receive-side counts where send-side was required — invisible
   under symmetric routing (local==glob numerically), corrupting under real
   imbalance. ALWAYS test index/split math with imbalanced matrices +
   zero-count peers/groups; a symmetric fixture cannot catch send/recv-side
   confusion. (T1-seed: `overlap_design/tests/test_w2_chunk_plan_t1_seed.py`.)
2. **Attrs-on-alias.** When a custom autograd Function returns its INPUT
   tensor, the caller receives a NEW alias object — attributes set on the
   input do not propagate (minimal repro: `b1 is not b0`, attrs lost). The
   `_w2_combine_works` list on the combine buffer vanished in every
   grad-enabled pass → the combine waits never ran. NEVER carry state on a
   tensor through an autograd boundary; use a stable-identity Python object
   (the plan / the carrier). (T1b: `test_w2_chunked_a2a_functions.py`.)
3. **TE generated-backward contracts.** TE's fused chunk_sort generated
   backward assumes a PERMUTATION (output rows == input rows); the W2 probs
   selection used it as a SUBSET → group-sized grad for a full-size input
   (T2 gate failure). A kernel's autograd contract — not just its forward
   math — must be verified for the intended use; subset-vs-permutation is a
   distinct contract. (Guard: `test_w2_probs_selection_backward.py`.)
4. **Silent-fallback / silent-disarm traps.** v1's FIX A shipped inert (a
   gate that could never fire — `is_grad_enabled()` inside Function.forward
   is always False) and passed 22/22 parity; W2's armed=NO disarm log was
   suppressed by a shared flag. A gate that can't fire, or un-fires, must
   NEVER be silent: armed/hit/fallback counters at WARNING are mandatory
   equipment, and the END-TO-END hit assertion (counters actually move
   through the production path) is the real guard — source guards alone are
   insufficient. Test through the REAL production context (the checkpoint
   machinery), and include a CONTROL case that can see the failure class
   (C′'s control: a simulated perturbation flips the free replay's map).
5. **Provenance-before-mechanism.** Before theorizing about kernel
   nondeterminism, audit the store/restore path's value-faithfulness (the
   FIX C re-audit: strong refs, no aliasing, carrier-keyed consumption —
   faithful; the divergence was upstream). And for "zero new syncs" claims:
   trace every host read to the already-D2H'd matrix, not a new copy (the W2
   v3 verification chain: init-time .cpu() + already-D2H'd matrix + free
   numpy ops).

— hilbert (verification, round 3). Questions to the estate docs above; the
orchestrator is kepler.
