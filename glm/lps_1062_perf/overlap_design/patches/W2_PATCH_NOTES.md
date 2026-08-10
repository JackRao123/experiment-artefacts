# W2 — BT_MOE_A2A_PIPELINE patch notes (helmholtz, 2026-08-09)

## What

`w2-moe-a2a-pipeline.patch` — intra-MoE-layer all-to-all ⇄ compute pipelining
for the mcore alltoall dispatcher: the 16 local experts are split into K=2
groups of 8 (group g = local experts [8g, 8g+8) on EVERY rank); the dispatch
and combine A2As become K list-form `torch.distributed.all_to_all` calls over
per-peer contiguous **views** (zero copies), so chunk 1's dispatch overlaps
chunk 0's expert GEMMs and chunk 0's combine overlaps chunk 1's. Design:
`../DESIGN_helmholtz.md` §2–§3 (variant (ii); variant (i) row-chunking is
unsound — see §2.2). Expected win: ~3.0 s/step v1 (fwd+replay pipelining; bwd
neutral), ~4.4 s with the v2 backward seq-bumps (not in this patch).

## Base (apply over)

- `w1-probs-a2a.patch` applied first (this patch's `token_dispatcher.py`
  hunks have the W1 gate block as context). Underlying tree: vendored mcore
  @ `57efae08b` + FIX A/B/F + hilbert's FIX C (frozen 2026-08-09).
- Files: `megatron/core/transformer/moe/token_dispatcher.py` (base = post-W1
  working tree), `megatron/core/transformer/moe/moe_layer.py` and
  `megatron/core/transformer/moe/experts.py` (base = clean @ `57efae08b`).
- Apply: `cd <mcore> && patch -p1 < w2-moe-a2a-pipeline.patch`. Verified:
  applies clean and reproduces the author's working tree byte-for-byte.

## Gate + telemetry (WARNING level)

- `BT_MOE_A2A_PIPELINE=K` (K must divide num_local_experts; K=2 validated).
  Default OFF; gate off ⇒ byte-identical upstream path.
- One-time gate-state line; one-time `armed — K=2, L=8 experts/group` (only
  after BOTH validation stages: dispatcher/config checks at layer init, and
  the frozen-experts check at the first forward — LoRA freezing can happen
  after construction, so the requires_grad check is deferred on purpose).
- `armed=NO — <reason>` fallbacks (loud, status-quo path): K ∤
  num_local_experts; EP=1; drop_and_pad; expert-TP>1; moe_permute_fusion off;
  cuda graphs; fused TEGroupedMLP impl; activation offloading; paged stash;
  moe_act recompute; overlap_dispatch_backward_with_experts_wgrad;
  moe_apply_probs_on_input; expert weights require grad.
- Per-window counters every 300 gated layer-passes (~1 step):
  `{dispatch_issues, combine_issues, waits, fallback_passes}`.

## Composition

- **W1 (`BT_MOE_PROBS_A2A_COMM`)**: with W1 armed, the single full probs A2A
  rides the second communicator (concurrent with the token chunks); without
  W1 it rides the EP comm issued FIRST (before the token chunks) so it does
  not trail them. Per-group probs are selected on-device by one fused
  sort_chunks call (`probs_keep_idxs`, precomputed at arm time). W1's own
  counters do not advance under W2 (the W2 counters cover the same passes).
- **FIX C (`BT_MOE_DISPATCH_REPLAY_CACHE`)**: the two count matrices W2 D2H's
  per pass (in the SAME single D2H batch + single event as today — no new
  syncs) are cached across the recompute replay via two W2-gated slots on FIX
  C's entry (`w2_local_counts_host` / `w2_global_counts_host`), populated in
  `_store_replay_entry` and compared in `_verify_host_metadata` (hilbert's
  BUG-3 review points, implemented).
- **Shared experts**: fc1 still launches between the issues and the waits;
  fc2 after all combine issues, before the waits (CDMC=1 push order
  preserved).

## Numerics (the parity argument, §5 of the design memo)

- Every expert's row block stays whole in one grouped-GEMM call: chunk g runs
  the SAME TEGroupedMLP module with a 16-entry counts list zero-padded outside
  the group (`forward_expert_group` in experts.py) — unchanged per-expert
  (M, N, K), unchanged FP8 blockwise scales (stateless per-call; per-expert
  Fp8Padding unchanged). **The one residual risk (R1: kernel-schedule
  dependence on zero-padded num_gemms) is what T2/V1 settles.**
- The combine writes each group's rows into a shared combine buffer at the
  unchunked path's per-source offsets — the final unpermute consumes a
  byte-identical buffer in byte-identical order (the combine output mirrors
  the permuted buffer P's layout; the recv bounds are SEND-side counts —
  hilbert's BUG-1/BUG-2 review points, implemented).
- Backward: per-group reverse list-A2As, issued+waited inline (v1 =
  status-quo bwd timing); dispatch-side reverses write into views of ONE grad
  buffer (no cross-group accumulation kernel; exact by disjointness).

## Second-review fixes landed (hilbert, 2026-08-09 round 2)

- **BUG A (severe, silent corruption):** combine work handles were carried as
  attrs on the combine-buffer tensor, but a custom Function returning its
  input gets a NEW autograd alias — attrs don't propagate, so in every
  grad-enabled pass the combine waits would never run. Fixed: works ride in
  `_W2ChunkPlan.combine_works` (stable-identity plan object), waited via
  `self._w2_pass["plan"].combine_works`.
- **BUG B (chain-breaking crash):** `_ChunkedCombineA2A.backward` returned a
  non-None grad for group 0's `None` combine_buf input → RuntimeError on the
  first backward. Fixed with the `ctx.had_buf` guard.
- **3a:** the trainable-experts disarm WARNING moved to a separate one-time
  flag (`_A2A_PIPELINE_DISARM_LOGGED`) — the shared armed flag suppressed it
  (silent disarm).
- **5:** `overlap_moe_expert_parallel_comm` added to the fallback list.
- **6:** T2 asserts FIX-C replay-cache stats (hits>0, misses==0) on the
  W2-on run in the use_fixc branches — a silent fallback passing torch.equal
  is the v1 trap.
- **4:** T2 header documents the two TE-version notes (Fp8Padding(0) boundary,
  M=0 amax inertness) with the 20-step loss canary named as the covering gate
  for multi-step amax drift.
- Adopted hilbert's `tests/test_w2_chunked_a2a_functions.py` (T1b: real
  `_ChunkedDispatchA2A`/`_ChunkedCombineA2A` on 4 gloo procs, imbalanced +
  zero-count fixtures, dispatch/combine fwd+bwd vs monolithic references) —
  ALL PASS after the fixes, as are test_w2_chunk_plan.py (72),
  test_w2_chunk_plan_t1_seed.py, and the W1 T1 suite.

## T2-gate defect + fix (2026-08-09, box run on 318g61w)

**Signature:** `GeneratedBackwardFor_te_moe_chunk_sort_fwd_defaultBackward
returned an invalid gradient at index 3 — got [33216] but expected shape
compatible with [65440]` (group-size vs full-size), simplest case, all ranks.

**Root cause (fibonacci/hilbert localization, confirmed):** the per-group
PROBS SELECTION in `dispatch_postprocess` used the TE fused
`sort_chunks_by_idxs` as a SUBSET of chunks (group rows out of the full probs
buffer), but the fused kernel's generated backward assumes a PERMUTATION
(output rows == input rows) and returned a group-sized grad for the
full-size input. The per-group TOKEN sorts are true permutations and were
clean (audited: `sort_input_chunk` / `restore_output_chunk` are full
permutations — now asserted by the guard test across geometries).

**Fix:** the probs selection now uses the UNFUSED `sort_chunks_by_idxs`
(split/cat) for that one call — autograd-correct via native view/cat
backward (group grads scatter into the full-size buffer, disjoint across
groups, engine-add exact), with host split/index lists precomputed at arm
time (`_W2PipelineConfig.probs_keep_idxs_host` + the already-D2H'd counts
matrix) so no new syncs. Values are bitwise-identical (a gather either way).

**Why it escaped (requirement a):** the Mac suites never execute TE fused
kernels' generated backwards (no TE / no GPU on the Mac); the T1
decomposition tests validate the index MATH, not the autograd contract of
the kernel consuming those indices; hilbert's T1b covers the hand-written
Functions on gloo. The fused-subset misuse is invisible at the plan level by
construction — it needed the on-box T2 gate, which caught it exactly as
designed.

**Off-box guard added (requirement b):**
`tests/test_w2_probs_selection_backward.py` (Mac CPU, pure-torch unfused
path) — subset-selection backward is full-size + lands exactly at selected
rows (bitwise vs manual scatter); the two groups' grads are disjoint and sum
to the full scatter; the permutation invariant the fused calls depend on
(sort/restore idxs cover every chunk exactly once) is asserted across
(EP,LE,K) geometries; and a grad-numel==input-numel contract probe covers
every sort_chunks call shape used in the W2 path. ALL PASS.

**Re-verification:** T1 (72) + T1-seed + T1b + W1-T1 + W3 suites ALL PASS
after the fix; patch regenerated (v3, 967 lines) and verified
(apply-clean + byte-for-byte).

## Tests

- **T1 decomposition (CPU, single-proc)** — `../tests/test_w2_chunk_plan.py`:
  72 checks, ALL PASS: split conservation; dispatch send views carve P
  exactly (content + partition); recv restrictions; **combine reassembly ==
  P byte-for-byte**; tpe slices; sort/restore/probs-keep index metadata;
  zero-(peer,group), zero-expert, 90%-imbalance, all-empty-group cases.
- **T1 seed suite (hilbert)** — `../tests/test_w2_chunk_plan_t1_seed.py`:
  ALL PASS (EP3/nle6/K3 configs, NCCL pairwise consistency, degenerate
  all-empty group). Named hard precondition for the box (TRACE_ACCEPTANCE
  W2).
- **T2 on-box V1 numerics gate** — `../tests/t2_w2_numerics_gate.py`
  (torchrun, EP=world): one real MoELayer (TEGroupedMLP, FP8 blockwise,
  frozen experts, checkpointed = first-pass + replay), in-process A/B:
  W2-off vs W2-on (±W1) must be `torch.equal` on the layer output AND input
  grad, for balanced / 90%-imbalance / zero-(peer,group) / zero-whole-expert
  routing, plus FIX-C-marked checkpoint variants of two cases. Not run on
  Mac (needs CUDA/NCCL/TE) — for bohr.
- **T3-style canary**: 20-step 16k×d8, gates off vs W1+W2 on, step-by-step
  loss must be bitwise identical (same recipe as W1's T3; see
  W1_PATCH_NOTES.md §T3). Watch the W2 telemetry lines for armed + counters.

## Known limits / follow-ups

- v1 backward is status-quo timing (issue+wait inline). v2 (NOT in this
  patch): seq-bump the combine/dispatch Functions so both reverse A2As issue
  before the MLP backwards — same mechanism as the shared-expert overlap.
- cuda-graph capture of the MoE layer falls back to the monolithic path
  (assert + loud); cudagraph support is follow-up work.
- The zero-padded grouped-GEMM launches 8 empty problems per chunk call
  (~16 extra launches/layer-pass) — measured in the canary; if the launch
  budget objects, the follow-up is per-chunk 8-gemm module shells sharing
  weights (the `_make_fused_ops` pattern, experts.py:427-437).
- Host cost per pass: two small numpy cumsums + list building from the
  precomputed plan structure — target <100 µs/pass (R10); measure in canary.
