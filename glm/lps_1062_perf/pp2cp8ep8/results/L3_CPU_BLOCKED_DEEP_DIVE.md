# L3 CPU-BLOCKED DEEP DIVE — call-site attribution + cache-pair prediction (gauss, 2026-08-13 ~01:5x CDT)

**Traces:** `traces/l3_d16_rank{0,8}.pt.trace.json` on box A (doppler's pulls;
one full d16 step, driver step-2 steady state, BT_PROFILE_RANKS=0,8, +4.0%
kineto tax: traced 136.3/138.1s vs untraced controls ~131.4s). Cross-checks:
`mn_d4_rank{0,8}` (M=N d4) and `diag_d4_rank{0,8}` (pre-M=N convoy d4).
Method: perfetto trace_processor on box (v56.1 shell), parent-chain
attribution seeded from each blocking call's `parent_id`; class separations
are the Aug-9-validated ones from `check_acceptance.py`. Scripts:
`lps1062_pp2/l3_deepdive.py` (+ probes 2–6), JSON outputs beside the traces.

## 0. Headline

**The raw host-blocked mass at d16 is ~97 s/step/rank — and ~all of it is
GPU-covered (absorbed by M=N runahead), confirming borel's §3b bucket read
(0.94 s sync-anchored critical path). BUT the op-window attribution finds a
second exposure channel his sync-anchored bucket cannot see: ~5.3 s/step on
rank 0 of long GPU-empty gaps (10–212 ms) that start *inside the DSA
layout-builder op windows*, plus ~2.4 s/step of RoPE host-read exposure —
both FIX B / FIX F targets. On-record prediction for the d16 B+F A/B below
(§4): +3–6% if the builder-window mechanism is causal (evidence says it is),
floor ~+1% if phase-confounded. NOT the Aug-9 +11–13% class either way at
d16 — the caches' big regime remains 16k/customer and convoy.**

## 1. Where the absorbed ~97 s hides (per step, per rank, d16)

API-level host-blocked time by call-site class (cudaStreamSynchronize +
blocking cudaMemcpyAsync + cudaEventSynchronize + cudaDeviceSynchronize):

| class | rank 8 | rank 0 | lever |
|---|---:|---:|---|
| RoPE host reads (aten::to/item under CheckpointFunction*) — tolist DtoH + per-seq items | **35.45 s** (24,320 calls) | **37.88 s** (23,296) | **FIX F** |
| DSA-bwd nonzero drains (under FusedSparseAttentionFuncBackward) | 31.23 s (1,280) | 26.88 s (1,216) | FIX A (parked) |
| Dispatcher replay eventSync (CheckpointFunctionBackward) | 5.72 s (640) | 6.31 s (560) | FIX C |
| Dispatcher fwd eventSync (CheckpointFunction) | 6.38 s (640) | 7.30 s (560) | (untouched) |
| Layout-builder nonzeros (aten::nonzero ← aten::index) | 1.28 s (115,264 API calls) | 6.44 s (109,440) | **FIX B** |
| Other item reads (metrics/packing etc.) | 9.05 s | 1.53 s | — |
| Step-level: 35 cudaDeviceSynchronize + step-level to-copies | 8.51+6.09 s | 10.29+? s | unresolved origin (§5) |
| **Total** | **≈97.9 s** | **≈96.7 s** | |

(borel's 61.3 s raw figure = streamSync 40.67 + eventSync 12.09 + deviceSync
8.51 — same trace, excluding the blocking-memcpy class, which adds ~36 s.
The two conventions reconcile exactly.)

Count-scaling cross-check (the Aug-9 M-law holds): every per-pass class count
scales exactly 4× from d4 to d16 (layout nonzeros 28,816→115,264 API calls;
RoPE 6,080→24,320; DSA-bwd drains 320→1,280; kernels 253k→1.01M), and
per-call waits are ~unchanged (queue depth per pass is M-independent), so
class totals scale ~4× with the 4× tokens/step.

**Proof of absorption:** GPU-union idle measured *inside* the blocking-call
windows ≈ 0.0 s for every class, both ranks. The host waits; the GPU has
queued work throughout. The 16.0 s (rank 8) / 25.8 s (rank 0) of GPU idle is
elsewhere: 10.8/10.3 s in 957k/902k sub-0.1 ms micro-gaps (pure launch drag
over 1.01M/946k kernels), and 10.5 s on rank 0 in 173 long (>10 ms) gaps.

## 2. The second channel: long gaps inside layout-builder windows (rank 0)

The sync-anchored bucket (borel's 0.94 s) measures idle *during CUDA API
sync calls*. Anchoring instead on the whole aten op window finds more:

- **Rank 0: 173 gaps >10 ms = 10.46 s; 95 of them (4.82 s) start strictly
  inside an aten::index(boolean-mask) window that parents a nonzero** — the
  packed-CP layout builder. Chance level from window coverage ≈ 3.6%;
  observed 55% of long gaps → ~15× over chance. Bracket kernels are the
  nonzero's own cub reduce → compact pair: the GPU finishes the count kernel
  and then sits empty until the compact launch arrives — the gap is the
  host-side path between them (the blocking calls inside the gaps sum to
  only ~1.3 s; the rest is host-path latency: python/aten bookkeeping +
  926 cudaEventQuery polls ≈ 1.06 s). 108/173 long gaps start within 50 ms
  after a pipeline p2p handoff — the shallow-queue phase at pass boundaries,
  where a long host path maps 1:1 to GPU idle.
- Rank 8 shows almost none of this (FIX B strict exposure 0.04 s) — and its
  layout blocks are 5× cheaper (1.28 s vs 6.44 s). Stage asymmetry is real:
  rank 0's builders run at deeper-queue/higher-pressure points.
- RoPE strict exposure (gap starts inside the tolist/item op windows):
  1.79 s rank 0 + 0.58 s rank 8 ≈ 2.4 s/step.
- Under M=N the stages overlap, so the step wall sees ~the max of the two
  stages' exposures, not the sum: B+F-addressable exposure ≈ **7–8 s/step**
  (rank-0-dominated) ≈ 5–6% of the step.

Mechanism caveat, stated honestly: a host thread "inside" the builder op for
200 ms could be (a) the builder's own long host path (45 sequential
nonzero+sync+alloc rounds per layer-pass), or (b) the launch thread
descheduled under CPU contention while nominally inside the op. The
cudaEventQuery polling inside the gaps shows the host is active, not
descheduled — favors (a). FIX B removes the entire builder call tree on
cache hit either way; if (b) dominates, the A/B will underperform the
prediction — which is exactly what the A/B is for.

## 3. The M/runahead story (borel's reframe questions)

**At what M does runahead stop absorbing?** Not between d4 and d16 at 131k.
mn_d4 (M=N, 4 mb): host-blocked 27.2 s on a 36.5 s step (75% of the step!),
GPU idle 3.89 s (10.7%) — absorption holds. d16: 16.0/25.8 s idle on
136/138 s (11.8/18.7%) — holds. Per-mb host-block cost and per-mb GPU work
are both M-independent at fixed seqlen, so absorption is structurally
M-independent in steady state; what M changes is the fill/drain fraction
(shrinks) and the step-level fixed syncs (amortize). **The exposure variable
is sequence length, not M:** at 16k the per-pass GPU work is ~8× smaller
while the RoPE/layout host costs per pass do not shrink (they scale with doc
count, which grows) — that is why the Aug-9 win was +18.5% at 16k×d32 vs
+11–13% at 131k×d4 on the golden topology. The convoy regime (pre-M=N) was
the other exposure regime: diag_d4 shows ~2× the host-adjacent idle of mn_d4
(e.g. rank 0 RoPE-attributed 1.93 s vs 0.07 s) on top of the structural
parks.

**Does it cap M-scaling above d16?** No. Block totals and GPU work both
scale with tokens/step; the ratio (hence absorption) is constant. What grows
with M is absolute launch count (~2M kernels/step at d32) — the micro-gap
launch drag grows proportionally, and FIX B's launch-storm removal
(~201k/1.01M = 20% of launches on rank 8, 190k/946k on rank 0, measured as
kernels inside the layout-builder windows) is worth ~1–2% there. The
step-level fixed costs (deviceSync cluster 8.5–10.3 s + step-level to-copies
6.4 s) *shrink* as a share with M. M-scaling is capped by memory/schedule,
not host syncs.

## 4. ON-RECORD PREDICTION — d16 B+F A/B (written before the A/B runs)

Arm: `BT_DSA_CP_LAYOUT_CACHE=1 BT_THD_ROPE_HOST_CACHE=1` vs unset, d16,
73c24b00 lineage (current box tree has the hooks inert).

**Predicted: +3% to +6% tok/s/GPU (1052 → ~1085–1115), central ~+4%.**

Decomposition: rank-0 builder-window exposure ~5.3 s + RoPE exposure ~2.4 s
≈ 7.7 s/step ≈ 5.7% of the untraced 131.4 s step as the addressable pool;
the caches realize it if the builder-window gaps are causal (evidence: 15×
over chance, bracket-kernel signature, host-active polling). Launch-drag
reduction from −20% launches adds at most ~+1% (speculative; drag per launch
is not uniform).

**Floor scenario ~+0.5–1%:** if the long gaps turn out phase-locked to the
pipeline handoffs rather than builder-caused, only the sync-anchored
exposure (borel's 0.94 s ≈ 0.7%) plus a sliver of drag is realized.

A result inside either band validates the model; a result ABOVE +6% would
suggest the launch-storm decompression compounds (Aug-9-style superlinearity)
and would justify re-examining the d16 priority; a result BELOW the floor
refutes the builder-window mechanism and points at CPU contention.

For the record, the §3b threshold question ("CPU-blocked ≥15% of step →
host-sync attack is top priority"): the sync-anchored share is 0.7%, the
op-window share is ~6% — neither crosses 15% at d16. **The caches are not
the top d16 lever; they remain the top 16k/customer-regime lever (+18.5%
measured Aug-9) and a convoy-regime lever, and they are nearly free to ship
(default-off, parity-proven, PR series staged).**

## 5. Loose ends flagged

- **35 cudaDeviceSynchronize/step (8.5–10.3 s at d16) — RESOLVED to source:
  mcore's `batch_isend_irecv` race workarounds in the dynamic-shape p2p
  path** (`p2p_communication.py:263` shape-exchange path, unconditional; and
  `:419` payload path under `batch_p2p_sync`). The THD packed path exchanges
  p2p shapes per handoff → ~2 full-device syncs per microbatch + 3 step-level
  (counts across boots: 11 at d4 M=N, 35 at d16 = 2M+3 exactly; the convoy
  diag_d4 had 17 costing 27–34 s/step — parks resident make each sync wait
  them out). Absorbed at d16 (gap attribution ~0), but it is a per-mb
  full-device sync in the hot path — pure exposure in any degraded-runahead
  regime and a constraint on any future schedule tightening. mcore's own
  comment says the sync is unneeded on modern PyTorch ("User should assert
  that we have a modern enough PyTorch"). Candidate morning lever: assert the
  torch version and drop/gate both syncs (upstream-fork patch; the :263 one
  is unconditional and needs a code change, :419 is config-gated). The 28
  step-level to-copies (6.4 s) remain unresolved (end-of-step metrics class
  vs memory_profile artifact — one control boot with profiling fully off
  resolves it).
- OTHER_ITEM (9.05 s rank 8): non-RoPE item reads — per-mb metrics/packing
  reads (packing.py:415 class). Small, local fix territory.
- The dispatcher eventSyncs (12.1–13.6 s/step combined fwd+replay) are the
  FIX C target — absorbed today, but they are the class that *inflates* when
  other blocks are removed (Aug-9 wait-relocation lesson). If the B+F A/B
  lands, watch the dispatcher eventSync CPU grow — benign, but expected.

## Appendix: data files

- `traces/l3_d16_rank{0,8}_deepdive.json`, `mn_d4_rank{0,8}_deepdive.json`,
  `diag_d4_rank{0,8}_deepdive.json` (box `lps1062_pp2/traces/`)
- Scripts: box `lps1062_pp2/l3_deepdive.py`, `l3_probe{2..6}.py`;
  analysis venv `lps1062_pp2/analysis-venv` (perfetto v56.1 shell).
- Method note: trace_processor v56.1 on-box; the v56/v57 overlapping-event
  difference noted in check_acceptance.py affects kernel_count at the ~1.5%
  level — immaterial to the class attributions here.
