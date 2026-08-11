# REVIEW_FIXA: stream-safety review + fallback frequency + A/B acceptance (laplace, 2026-08-09)

Scope: ramanujan's FIX A design (`CANDIDATE_FIXES_ramanujan.md` §FIX A). Code refs:
vendored mcore @ ef4ea4a8 (`.../experimental_attention_variant/dsa_cudnn_kernels.py`,
`dsa.py`, `dsa_layout.py`, `dsa_masking.py`; trainer packer
`server/src/trainers_server/dp_worker/backends/megatron_bridge/thd_cp.py`, `packer.py`).

## (2a) Stream-safety review of FIX A — verdict: sound, with 8 requirements

### 0. Attachment-point correction (matters for the diff)

The trace's 156 syncing nonzeros come from **`FusedSparseAttentionFunc`** (the
index-sharing class, `dsa_cudnn_kernels.py:2580-2640`): its `backward` (:2632) calls
`_run_sparse_attention_backward` **without** `all_rows_nonempty`, so it defaults to
`False` and the syncing path is *unconditional* — this, not data, is why the profiled
step shows 156/156. The other class (`FusedIndexerSparseAttnFunc`, :2394) passes a
static config predicate that is also False on the CP-THD path. FIX A must attach the
async flag where `topk_length_flat` is produced — `_run_sparse_attention_forward`
(:2037, shared by both classes) — and stash `(event, pinned_buf)` on the calling
class's ctx. Minimum viable: `FusedSparseAttentionFunc` only (the measured path).

### 1. Ordering spec (must follow exactly)

In fwd, immediately after `topk_length_flat` exists:
1. `flag_gpu = (topk_length_flat > 0).all()` on the **current (compute) stream**;
2. `side_stream.wait_stream(torch.cuda.current_stream())` — *after* the `.all()` enqueue
   (wait_stream snapshots already-enqueued work; this orders the copy after the flag);
3. on `side_stream`: `pinned_buf.copy_(flag_gpu, non_blocking=True)` (1 byte),
   `evt = side_stream.record_event()`;
4. `ctx.dsa_nonempty = (evt, pinned_buf)`.

In bwd: `evt.synchronize()` → read `pinned_buf` → if True,
`valid_row_indices = torch.arange(N, dtype=torch.int64, device=...)` and keep every
other op of the compaction dataflow byte-identical (dummy-row cat, index_selects,
wrapper call); if False/missing → original `torch.nonzero` path unchanged.
`nonzero(t>0).flatten()` == `arange(N)` bitwise when all rows are >0 (row-major, int64)
— the fast path is bitwise identical by construction, no ULP concerns.

### 2. Pinned buffer lifetime / GC

Per-call `torch.empty((), dtype=torch.bool, pin_memory=True)` referenced from ctx until
bwd reads it. That reference is what prevents the caching host-allocator from recycling
the block while the D2H is in flight — do **not** use a module-global ring buffer (156
kicks in flight per step; replay interleaving makes slot reuse racy). Per-call pinned
allocs recycle through the caching allocator after warmup; first-step `cudaHostAlloc`
stalls are warmup-only. Source-tensor lifetime is fine: `topk_length_flat` is a fresh
tensor (`.permute(1,0).reshape(-1)` copies) saved on `ctx.topk_length`, never mutated
in place after production. ✓

### 3. Side stream

ONE lazily-created module-level `torch.cuda.Stream` singleton (mirror the dispatcher's
`self.cuda_dtoh_stream`), never per-call streams. Precedent: the dispatcher runs this
exact pattern (side-stream D2H + `d2h_event.synchronize()`) 300×/step in the same step
today without incident (300 `cudaEventSynchronize` / 4.13s in the trace).

### 4. CUDA_DEVICE_MAX_CONNECTIONS

Trainer runs with it **unset** — verified in the captured `/proc` environ
(`runs/overnight_20260809_mfu_sweep/results/B_dp2_boot_deadlock/trainer_env_node0.txt`, no such var); the
LPS-1003 regression-test docstring confirms "unset (the production default)". Driver
default connections (>1) ⇒ the side-stream copy is genuinely concurrent with compute.
Under a hypothetical `=1` the design is still correct and deadlock-free: `wait_stream`
is a device-side ordering snapshot, not a host block; the 1-byte copy completes
mid-fwd; bwd holds nothing the side stream waits on (no circular wait). The LPS-1003
race class (concurrent writes to a tensor a DSL kernel reads) does **not** apply: the
side stream only reads a frozen tensor and writes private pinned host memory. The one
hazard to avoid in review: the side stream must never touch any tensor the compute
stream later mutates — satisfied here (§2).

### 5. Full-recompute subtlety (the one that bites)

- Under mcore full recompute the first fwd runs under `torch.no_grad()` and its ctx is
  discarded; the **replay** (inside `CheckpointFunctionBackward`, grad enabled) creates
  the ctx the bwd actually uses. Gate the flag kick on `torch.is_grad_enabled()` so the
  no-grad pass skips it (saves 156 wasted kicks; harmless if forgotten).
- So the D2H is enqueued during the replay, mid-layer, and the bwd's
  `event.synchronize()` fires tens of ms later, after the replayed layer fwd has
  finished executing on the compute stream. The event waits only on the side-stream
  copy (ordered after replay's `topk_length` production) — long complete ⇒
  **µs-scale, not a hidden drain** (it's `cudaEventSynchronize` on a completed event;
  it does not touch or drain the compute stream). Deadlock-free per §4.
- Determinism: replay recomputes `topk_length` bitwise (full-recompute determinism),
  and the flag only selects between two bitwise-identical index sets — no fwd/replay
  inconsistency even in principle.
- Post-patch trace check that proves this review: the 156 new event syncs must total
  <100µs and zero `aten::nonzero` slices >5ms may remain (§2c counters 1, 4).

### 6. CUDA-graph guard

If `torch.cuda.is_current_stream_capturing()` → original path (side-stream work during
capture is illegal). The trainer doesn't capture this path today; one-line guard.

### 7. Numerics / parity

Fast path bitwise-identical by construction; fallback byte-unchanged. Parity test must
include a **forced-fallback case** (a `topk_length` row zeroed artificially) so both
branches are exercised — see (2b) for why the fallback will otherwise never run.

### 8. Env gate

Separate gate from FIX B (different subsystem): e.g. `BT_DSA_BWD_ASYNC_NONEMPTY=1`,
default OFF, per HANDOFF correctness rule.

## (2b) Fallback frequency — fast path should fire ~100% on bench AND customer data

`topk_length[row] == 0` ⟺ the row's varlen key range is empty. Tracing the data path:

- The trainer's THD packer (`thd_cp.py:300-346`) pads **every doc** to a multiple of
  `2*cp_size` and records **padded** boundaries in `cu_seqlens_padded`; the microbatch
  length is exactly `cu_padded[-1]` — pad rows live *inside* their doc's padded
  segment, and there is no unrepresented tail region.
- Mask construction (`dsa_masking.py:68-80`, `generate_varlen_mask_params_for_positions`
  on the padded cu_seqlens): a pad row at position p in doc i's padded span gets bounds
  `[doc_i_start, p+1)` — always non-empty (contains at least its own position).
- All three packed top-k branches (`_indexer_topk_from_score_chunks`,
  `_indexer_topk_multi_packed_cp_thd`, `_indexer_topk_single_packed_cp_segments`) derive
  `seq_lens = ends.clamp(max=sk) > 0` for pad rows ⇒ ≥1 valid selection ⇒
  `topk_length ≥ 1`. Short-tail zigzag segments likewise keep full in-doc causal ranges.
- The customer histogram (70% ≤32K / 28.5% 32–64K / 1.4% 64–131K) changes how many docs
  and pad rows a 131K buffer holds, **not** per-row non-emptiness. P(all-rows-nonempty)
  ≈ 1 per microbatch in prod, same as the uniform benches.
- The trace's 156/156 nonzero rate is **not** evidence of empty rows — it's the
  unconditional default in `FusedSparseAttentionFunc.backward` (§0).

Caveats: (i) this is a code-level inference, not a measurement — the patch **must ship a
per-step fallback counter** (the host flag is already materialized; count + log every
step for the first 100 steps). gibbs reports it in the A/B; expected 0/156/step on the
bench shapes, and the first customer-shape (16k-d32) boot confirms the prod regime. If
routine fallbacks ever appear, the win claim is void for that shape — stop and report.
(ii) When the fallback does fire, the cost is today's status quo (one ~120ms drain per
layer-bwd) — no regression, just no win for that layer-mb.

## (2c) A/B acceptance criteria (for gibbs)

Counters derive from the exp05d baseline trace (CP16, 78 layers, 2 mb/step); they
transfer to expA131 unchanged — 156 = 78×2 and 85 = 17×5 are buffer-shape-independent.
Profiler runs are separate boots from timed runs (profilers stay out of timed windows).

From a one-step profiled capture on the patched boot (steady window, same rank 0):

| # | counter (trace_processor) | baseline | accept |
|---|---|---:|---:|
| 1 | `aten::nonzero` calls / CPU | 26,684 / 20.59s | ≤ ~200 / ≤ 0.2s |
| 1a | — of which slices >5ms (the bwd drains) | 160 | **0** (FIX A) |
| 1b | — layout-builder nonzeros (fwd+replay) | 26,520 | ~170 (FIX B: 85 × 2 mb first-layers) |
| 2 | `cudaStreamSynchronize` calls / CPU | 29,131 / 19.98s | ≤ ~500 / ≤ 0.6s |
| 3 | `cudaMemcpyAsync` CPU total | 15.01s | ≤ 1s **iff FIX F (THD RoPE host cache — site corrected 2026-08-09, see ATTRIBUTION.md errata) ships**, else ~14.8s remains — A/B must state which gates are ON |
| 3a | — blocking pageable-DtoH copies >1ms | ~590 | 0 if F ships |
| 4 | `cudaEventSynchronize` calls / CPU | 300 / 4.13s | ~456 / ≤ 4.5s; **no new slice >1ms** (FIX A's 156 events are µs — this proves §5) |
| 5 | patch fallback counter (log) | n/a | **0/156 per step**, logged first 100 steps |
| 6 | SendRecv GPU total | 24.48s | ±5% (unchanged) |
| 6a | total kernel count | 361,032 (v57.2) / 355,627 (v56.1 — checker-authoritative) | ≤ 230,000 (FIX B removes ~150k launches) |
| 6b | `Memcpy DtoH (Device -> Pinned)` count | 29,502 | ≤ ~3k |
| 7 | GPU-union idle in step | 4.74s | ~4.0–4.5s (the 0.70s sync-attributable idle gone) |

Correctness gates (HANDOFF standing rule): parity test bitwise green on **both**
branches (incl. forced-fallback case); loss canary with identical `--warmup-datums`
across boots, drift ≤ 5e-3 vs unpatched.

Throughput (profiler-free boots, steady windows): non-regression bound ≥ baseline −2%;
expected win **+1–4%** per the attribution bound (0.7–2s of the 48.8s step). A measured
"win" > +6% exceeds the trace-derived bound — treat as a measurement artifact (window
selection / warmup contamination / F5 shape) and re-measure before reporting.

## Errata to ATTRIBUTION.md

`cudaEventSynchronize`: I reported 150 calls / 2.11s — that was the main thread only.
Full count is **300 / 4.13s** (150 fwd on main + 150 recompute-replay on the autograd
thread). ramanujan's figure is correct; the dispatcher's per-MoE-layer event sync runs
in both passes.
