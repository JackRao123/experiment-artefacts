# Backward-reload knobs: validation results (2026-08-22)

Patch: `bt_offload_backward_prefetch.patch` on Megatron-LM `d7a72ec1`
(deployed on `tj-wdpok4w`, worktree `qkpox9w`, trainers `c226338a`).
All arms: 0D4M proxy, 8,192 tokens, four-group offload set, fresh process
per arm, one warmup window.

## Traced single-window A/B (same patched file, env on vs off)

| metric | knobs off (legacy) | K=6 + unchained | no-offload baseline |
|---|---:|---:|---:|
| traced window fb | 0.877 s | 0.868 s | 0.724 s |
| H2D window (rel. ms) | [693, 852] | [638, 818] | — |
| H2D overlap w/ compute | 49% | **97%** | — |
| H2D uncovered | 78.4 ms | **4.3 ms** | — |
| backward wall / idle | 411.2 / 98.6 ms | **336.3 / 24.0 ms** | 338.9 / 27.0 ms |
| mid wall / idle | 244.5 / 1.5 ms | 244.6 / 1.5 ms | 244.7 / 1.5 ms |
| forward wall / idle | 204.2 / 138.6 ms | 270.7 / 206.1 ms* | 123.2 / 57.7 ms |

*The K6 traced window drew the roaming host stall in its forward phase
(81 ms blocked `cudaEventQuery` under `aten::silu` + 23 ms under
`aten::empty`); the knobs-off window escaped it. The stall roams — see
below.

**The knobs do exactly what the trace demanded: backward reload is fully
hidden (backward wall becomes baseline-equal).**

## 10-window statistical A/B (`stat_ab/`, bench_driver2c, untraced)

| arm | median fb | mean | stdev | min | max | median TPS | peak alloc |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline, expandable_segments (ES) on | 0.699 s | 0.712 | 0.045 | 0.691 | 0.839 | 11,721 | 108.52 GiB |
| offload legacy, ES on | 0.923 s | 0.966 | 0.143 | 0.848 | 1.301 | 8,871 | 100.05 GiB |
| offload K6+unchained, ES on | 0.905 s | 1.199 | 0.883 | 0.769 | **3.669** | 9,115 | 100.05 GiB |
| offload K6+unchained, ES off | **0.854 s** | 0.871 | **0.070** | 0.776 | 0.996 | 9,587 | 100.06 GiB |
| baseline, ES off | 0.695 s | 0.741 | 0.148 | 0.689 | 1.163 | 11,787 | 108.53 GiB |

Readings:

1. **Memory saving fully preserved.** Peak allocated is identical with
   and without the prefetch knobs (100.05 GiB, −8.46 GiB vs baseline =
   the full 3-layer saving). The allocation peak sits at end-of-forward,
   before any reload, so K-deep backward prefetch does not touch it.
2. **Knobs raise the fast-window ceiling** exactly by the traced −75 ms:
   best windows 0.769-0.832 s vs legacy floor 0.848 s.
3. **expandable_segments is implicated in the roaming stall.** With ES on,
   offload arms have a heavy tail (legacy max 1.301 s; K6 max 3.669 s —
   same class as the campaign's old 2.54 s outlier, which occurred on
   legacy code). With ES off the tail vanishes (max 0.996 s) and stdev
   collapses 0.883 → 0.070. Baselines are ES-insensitive (0.699 vs
   0.695) — the interaction is offload-specific. Sample is 10 windows per
   arm: treat the tail-rate comparison as strong signal, not proof.
4. **Best offload config (knobs + ES off): median 0.854 s = 9,587
   tok/s/GPU, −18.1% vs baseline** (legacy was −24.3%), with near-baseline
   variance.
5. **Parity:** max |loss delta| vs baseline at matched window index
   2.86e-6 across every arm/window (within the observed run-to-run
   nondeterminism band of identical configs); grad-norm deltas ≤ 1e-7.
   Two cross-run window pairs were bitwise identical. No OOM, no trainer
   failure in any arm.

## Three-way reference: full recompute vs no-recompute vs offload

Full-recompute arm added 2026-08-22 evening
(`stat_ab/stat-04-fullrecompute-es-v1.json`, config
`configs/layer_scaling_20260822/04-full.json`, full/uniform/1, no offload,
ES on, 10 windows):

| arm | median fb | median TPS | vs no-recompute | peak alloc | memory saved |
|---|---:|---:|---:|---:|---:|
| no recompute, no offload (ES on) | 0.699 s | 11,721 | — | 108.52 GiB | — |
| **full recompute** (ES on) | 0.782 s | 10,469 | **−10.7%** | **95.97 GiB** | **−12.55 GiB** |
| offload, fixed (knobs + ES off) | 0.850 s | 9,878 | −16.2%* | 100.06 GiB | −8.46 GiB |

*vs the ES-off baseline (11,787); −15.7% vs the ES-on baseline.

Full-recompute windows: 9689 9744 10167 10388 10447 10492 10535 10572
10610 10620 tok/s — tight, no outliers. Losses in the same ≤3e-6 band.

**Honest proxy-scale reading: full recompute currently beats offload on
both axes here — faster AND more memory saved.** Two reasons this does
not transfer to production as-is: (1) at 4 proxy layers the recompute
work is small next to the fixed LM-head/loss cost, so full recompute
costs only 10.7%; at the 78-layer production model it is measured at
~35% of the step — that ratio inverts with depth while offload's cost
does not scale with depth the same way. (2) The production plan is
selective-recompute + offload of what remains, not either alone. But at
proxy scale, this is the honest ranking, and any future proxy result
should be quoted against all three arms.

## Where the remaining −18% lives (trace-quantified)

- **Forward D2H exposure, intrinsic to the proxy's scale (~10-15%).**
  Forward compute in the D2H span is ~65 ms busy against 150.8 ms of
  copies — at 8,192 tokens the proxy simply lacks forward compute to hide
  3 layers × 2.68 GiB. This is the handoff's hypothesis 6 confirmed
  quantitatively; it shrinks as compute-per-layer grows toward the
  131k production shape.
- **MoE dispatcher host readbacks (~5%).** The dispatcher's few-byte D2H
  readback after `te_moe::chunk_sort_fwd` queues on the copy engine behind
  the offload stream's GiB-scale transfers (70 ms vs 24 ms baseline of
  bare `cudaEventSynchronize`). Mitigation candidates: chunked offload
  copies so the engine can interleave, or moving the readback off the
  D2H engine. Untouched in this pass.

## Recommendations

1. Arm `BT_OFFLOAD_PREFETCH_DEPTH` + `BT_OFFLOAD_H2D_UNCHAINED=1` for all
   offload work (defaults keep legacy behavior; K needs retuning at
   production scale — 6 groups ≈ 1.5 layers ahead here).
2. Treat the ES × offload host-stall interaction as a first-class blocker
   for production offload trials: either run offload ranks with
   `expandable_segments:False` (measure the fragmentation cost first) or
   root-cause the stall (suspect: ES map/unmap on an untraced thread
   holding the CUDA driver/allocator lock while pinned copies are in
   flight; the blocked calls are innocent `cudaEventQuery`/
   `cudaEventSynchronize`/allocations with zero CUDA activity of their
   own).
3. Do not use this proxy to pick the production offload fraction — after
   the fix the binding constraint is PCIe-bandwidth-vs-compute ratio,
   which the 8k proxy distorts against offload (see SUMMARY.md envelope).

## Fresh-subagent review (one-shot, per the standing review policy)

Four findings, all addressed in the final patch:

1. **Real (config-conditional) bug:** `BT_OFFLOAD_H2D_UNCHAINED` skipped
   the `wait_stream` during CUDA-graph capture too — but under capture that
   wait is the h2d stream's only join into the capture (the offload-event
   wait is deliberately skipped there). Fixed: the wait now always stays
   under `is_graph_capturing()`. Not exercised by this proxy or the current
   131k deployment, but latent.
2. **Memory-bound correction:** the early-residency bound was ~2K+1 groups,
   not ~K (the group-start node's unconditional reload ran at K+1, and once
   the chunk's pending list emptied, remaining start nodes dragged ~K
   next-chunk groups in early via `pre_reload_last_layer`). Fixed: with
   prefetch at depth, the start-node reload is gated on in-flight < K;
   bound is now ~K+1.
3. **Warmup hardening:** top-up is now disabled during warmup (backward
   group order unknown → K-deep blind LIFO prefetch would manufacture
   demand misses, each costing a full stream synchronize).
4. **Env parsing:** empty-string `BT_OFFLOAD_PREFETCH_DEPTH=` no longer
   crashes handler construction.

The reviewer confirmed: state machine sound (no double reloads, ordering
preserved via `_backward_group_order`), the eager-mode unchaining safety
argument correct leg by leg, multi-chunk/PP paths safe, and the
default-off path behavior-identical to legacy.

Re-validation after the review fixes (K=6 + unchained, ES off, 8 windows,
`stat_ab/reval-04-k6u-noes-v2.json`): 9,610 tok/s aggregate, median window
9,878 TPS, no outlier (worst window 0.96 s), peak_alloc 100.06 GiB
(−8.46 GiB saving intact), losses in the same ≤3e-6 parity band. No
regression from the review fixes; marginally better than the pre-fix arm.

## Artifacts

- `stat_ab/stat-04-*.json` + `stat_ab/stat_ab.log` — 10-window arms
- `traces/patched-04-K6-unchained-v1/`, `traces/patched-04-knobsoff-v1/`
- `raw/patched_*.csv`, `raw/knobsoff_*.csv` — trace slice dumps
- `bt_offload_backward_prefetch.patch` — final (CUDA-graph-capture-guarded)
- Box copies: `/root/.cache/user_artifacts/lps1062_traces/`,
  `/root/.cache/user_artifacts/lps1062_bench/stat-04-*.json`; patched file
  deployed in the `qkpox9w` worktree's vendored Megatron-LM (uncommitted;
  `git diff` saved as `lps1062_traces/bt_offload_backward_prefetch.patch`).
