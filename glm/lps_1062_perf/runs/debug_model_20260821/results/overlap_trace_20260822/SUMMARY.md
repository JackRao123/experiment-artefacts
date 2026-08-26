# Four-layer offload overlap: trace findings and fix (2026-08-22)

Handoff: `HANDOFF_4L_OFFLOAD_OVERLAP.md`. Box `tj-wdpok4w`, worktree
`/root/.cache/user_artifacts/devboxes/qkpox9w/trainers` @ `c226338a`
(branch `lps1062-scale`, PR #1074). Model: 0D4M proxy, 8,192 tokens,
TP1/PP1/CP1/EP1, offload groups `core_attn, moe_act, qkv_linear, attn_proj`.

## Traces captured (first deliverable)

One-window kineto traces via `/runtime_profile/start|stop`, one warmup +
2 untraced windows before the traced window, fresh processes:

| arm | traced window fb | untraced neighbors | trace |
|---|---:|---:|---|
| 04-none (baseline) | 0.724 s (11,315 tok/s) | 0.710-0.731 s | `traces/trace-04-none-v1/` |
| 04-four-groups (offload) | 1.024 s (8,001 tok/s) | 0.875-0.941 s | `traces/trace-04-four-groups-v1/` |

Profiler perturbation on the baseline is negligible (0.724 traced vs
0.710-0.731 untraced). The offload arm is noisier; its traced window is at
the slow end of its own untraced spread and matches the layer-scaling
campaign's 7.2k median regime.

## Headline: the penalty is 100% idle time, not extra work

Main-stream (stream 7) GPU busy time is byte-identical across arms
(633.3 vs 634.3 ms; per-phase 65/243/312 ms in both). The entire ~300 ms
traced penalty is main-stream idle, split into three separate mechanisms:

| phase | baseline wall | offload wall | delta | mechanism |
|---|---:|---:|---:|---|
| forward (to first CE chunk) | 123.2 ms | 205.3 ms | +82 ms | dispatcher host readbacks queue behind bulk D2H on the copy engine |
| mid (CE1 → backward start) | 244.7 ms | 383.5 ms | +139 ms | python thread blocks 258 ms inside loss-path `aten::add` (lock-class stall, zero CUDA calls) |
| backward | 338.9 ms | 417.9 ms | +79 ms | H2D reloads issued just-in-time; offloaded layers' backward is copy-bound |

## Success-criteria numbers

**Copy volume and bandwidth.** D2H: 87 copies, 8.031 GiB, 150.8 ms busy,
**53.2 GiB/s**. H2D: 87 copies, 8.031 GiB, 155.2 ms busy, **51.8 GiB/s**.
Volume = 3 offloaded layers × 2.68 GiB exactly. Bandwidth is healthy: the
NUMA/PCIe hypothesis is ruled out.

**Overlap and uncovered time.**
- D2H runs [6.2, 190.1] ms (forward). Overlap with any compute: 44.9 ms
  (**30%**); uncovered 106.0 ms. At this proxy scale forward compute in that
  span is only ~65 ms busy, so full D2H hiding is arithmetically impossible
  at 8k — a proxy-scale artifact, not an implementation verdict.
- H2D runs [829.1, 1000.9] ms — the **last 172 ms of a 418 ms backward**.
  Overlap with any compute: 71.7 ms (**46%**); uncovered 83.4 ms.
- Main-stream idle gaps >1 ms: 410.5 ms total (offload) vs 65.5 ms (baseline).

**Prefetch hits / demand misses per group** (cumulative telemetry over 8
iterations; steady-state per-iteration deltas in parentheses):

| group | prefetch_hits (per iter) | demand_misses (per iter) |
|---|---:|---:|
| qkv_linear | 270 (+45) | 60 (+0) |
| core_attn | 126 (+21) | 28 (+0) |
| moe_act | 90 (+15) | 20 (+0) |
| attn_proj | 36 (+6) | 8 (+0) |

**Steady state has zero demand misses.** All misses date from warmup/early
iterations. The stalls are prefetch hits that arrive late: consumers block
on the group reload event. Valve (uncapped): commits +3/iter per group,
max_pending 4, zero drain firings.

## Mechanism detail

1. **Forward (+82 ms).** Three bare `cudaEventSynchronize` host blocks
   (42.5, 13.6, 14.2 ms) on the python thread, each immediately after
   `te_moe::chunk_sort_fwd` — the MoE dispatcher reading routing metadata
   back to the host. The first block ends at 59.3 ms, the exact completion
   time of a tiny 4-copy D2H batch on stream 28: the dispatcher's few-byte
   readbacks serialize on the D2H copy engine behind the offload stream's
   GiB-scale transfers. Baseline has one such wait (24 ms, compute-backlog
   bound); offload turns it into 70 ms of copy-engine queueing.
2. **Mid (+139 ms).** In the second LM-head chunk's loss chain
   (`LinearWithGradAccumulationAndAsyncCommunication → mul → to → add →
   CrossEntropy`), `aten::add` takes **257.9 ms vs 21 µs baseline**, with
   zero CUDA API calls until its two kernel launches at the very end. GPU
   goes fully silent for 139 ms inside it. No cudaMalloc / cuMem* / event
   calls from the blocked thread → a lock-class wait, most plausibly the
   CUDA caching-allocator device mutex held by an untraced thread
   processing the offload's record_stream-freed blocks. **Mechanism
   unresolved**; it is offload-specific and is the best candidate for the
   campaign's window-to-window variance (3,225-TPS outlier windows).
3. **Backward (+79 ms).** Reloads are issued one per group-start backward
   node, so the first H2D is issued by the CPU at 827.7 ms — after loss +
   LM-head-dgrad + the retained layer's backward (~245 ms into backward).
   All 8 GiB of H2D then crams into the last 172 ms while the three
   offloaded layers' backward consumes it just-in-time; their sync waits
   stretch (FusedSparseAttention backward 17-33 ms vs 8-10 ms baseline).
   Additionally `on_group_start_backward` chains the H2D stream behind
   *all main-stream work queued so far* (`h2d_stream.wait_stream(main)`),
   which is an ordering choice, not a lifetime edge.

## Fix under test: backward reload pipelining

Patch `bt_offload_backward_prefetch.patch` (this directory) on
Megatron-LM `d7a72ec1` adds two independently env-gated knobs, both
default-off (legacy behavior preserved bit-for-bit when unset):

- `BT_OFFLOAD_PREFETCH_DEPTH=K`: top up H2D so up to K groups are in
  flight ahead of their consumers, issued at every backward group boundary
  (earliest: the retained layer's first commit-backward node). Bounds the
  early re-residency to ~K groups of GPU memory.
- `BT_OFFLOAD_H2D_UNCHAINED=1`: drop `h2d_stream.wait_stream(main)` before
  each reload. Safety: pinned source guarded by the group offload event;
  destination is a fresh h2d-stream allocation (allocator pools are
  per-stream); consumers join via the reload event; pinned-buffer reuse
  waits the pool reuse event; `record_stream` at pop covers frees.

Results: see `PATCHED_RESULTS.md`.

## Can this scale to the 78-layer production topology?

Stated per the handoff's instruction to answer from the trace rather than
memory scaling:

- **Scheduling: yes.** With the two knobs armed the implementation hides
  the backward reloads completely at proxy scale (97% overlap, backward
  wall equal to no-offload baseline). No structural scheduling obstacle
  remains in the fine-grained offload path for the measured groups.
- **Bandwidth: the full 4-group set on all 75 MoE layers does not fit.**
  Envelope with campaign-measured numbers: per-rank per-layer one-way
  payload at 131k/CP8/EP8 scales to ≈5.5 GiB (attn-side ×2 for 16,384
  local tokens: core_attn ≈2.3 GiB, qkv ≈0.55, attn_proj ≈0.50; moe_act
  ×2 for 131,072 expert-token rows/rank ≈2.0 GiB). At the NUMA-local
  27.5-28.8 GB/s per-GPU budget that is ≈190 ms/layer of H2D against
  roughly ≈125 ms/layer of backward compute (918 tok/s/GPU d4 record) —
  ~1.5× over budget. Dropping `core_attn` (≈3.1 GiB → ≈110 ms) or
  offloading ~60-70% of layers fits the envelope. These are ±30%
  back-of-envelope numbers; the decision needs a production-shape probe,
  not this proxy.
- **Blocker to clear first:** the roaming 80-260 ms host stall
  (mechanism open, offload-correlated). At proxy scale it is worth
  10-25% of a window; its production-scale behavior is unknown.

## Artifacts

- `traces/trace-04-none-v1/`, `traces/trace-04-four-groups-v1/` — kineto
  traces + trainer logs
- `raw/` — driver JSONs, telemetry text, telemetry trainer log
- `bt_offload_backward_prefetch.patch` — the fix
- `../../tools/trace_driver_20260822.py`, `run_trace_capture_20260822.sh`,
  `run_telemetry_20260822.sh`, `run_patched_arm_20260822.sh`,
  `overlap_analysis_20260822.py` — reproducers
