# W3-v4 — lookahead recompute, capture-restoring design (SKELETON, 2026-08-10)

**Status: DESIGN SKELETON for morning review — not built, not booted.** W3
closed for the night of 08-10 (helmholtz's call): no surgical v4 patch
existed by the 13:00 UTC bar; v4 needs real design work plus the L3
prerequisite. This doc maps the path so Jack reads a mapped path, not a dead
end. Lineage: v2 (mechanism proven, win refuted — whole-backlog
`wait_stream`) → v3 (ordering fix correct, capture 0.9 % — the residual
couplings below) → v4 (this skeleton).

## 0. Where the lever stands (the evidence chain)

- **The territory is real and still there.** SM slack measured twice (gate +
  on-arm): ~35 s/step of bwd-phase SM slack, 83–84 % of kernel time <10 %
  occupancy, SendRecv comm windows at ~0 % occupancy (~24–27 s/rank/step).
- **v3 fixed what it aimed at.** The whole-backlog `wait_stream` chain is
  gone: kicks now release in their local neighborhood, not behind the phase
  tail (T6 evidence 2: launch-lag p50 39.65 ms with the side stream free
  ~11 ms of that — GPU-side dependency-gated, not host-late, not FIFO).
- **v3 did not reach the territory.** Capture 0.9 % (0.10 s of 11.29 s
  side-stream time concurrent; v2 was 0.1 %). Everything else was correct:
  99.5 % right-window localization, stash hits exact, numerics near-bitwise
  (w0 +0.1e-3, mains ≤0.7e-3), memory +24.9 GiB ≤ the +28 bar.
- **Chunk-count correction (curie):** the C′-ON stack runs **75 chunks/mb**
  (74 kicks + 1 structural inline), not the model-derived 78 — the frozen
  bar's constant conflated DSA-layer count with checkpoint-chunk count. The
  bar's STRUCTURE (kicks == hits == chunks−1, misses == 1) held exactly.
  Carried into DESIGN_W3V3.md §7.

## 1. T6's causal chain (verbatim anatomy, W3V3_T6_WAIT_ANALYSIS.md)

A reverted ⇒ DSA-bwd drains block the host ⇒ shallow main queues;
the (b)-edge (`last_bwd_end`) completion is **comm-tail-delayed** (the
compute stream's chunk-end marker sits behind the chunk's A2A tail — the
compute stream waits on the comm result — so the (b) event completes only
when the chunk's stream-79 comm tail drains; gate-release coincidence:
75 % NCCL-stream-79 ends, 24 % expert-GEMM ends, ~0 % stream-7 ends);
kick releases exactly when the GPU goes idle; the kick **fragments**
(~5 fragments per chunk kick, 1,579 gap-delimited clusters not 296 clean
kicks) re-gating on shared comm/expert FIFO slots (its own MoE A2As share
the comm-stream FIFO — v3's E7 "by design" — and its expert GEMMs share the
expert streams); each fragment runs solo in a drain gap ⇒ 0.9 % capture.

**The three residual couplings v4 must break:**
1. **(b)-edge comm-tail delay** — the allocator-safety edge's event is
   recorded at the chunk backward's stream tail, which sits behind the comm
   tail; the kick's start is gated on the previous chunk's comm draining.
2. **Shared FIFO slots** — the kick's comm (stream 79) and expert GEMMs
   (streams 51/161–164) queue behind the main backward's own comm/GEMM work
   instead of running in the 0 %-occupancy comm windows' SM slack.
3. **Empty main queues at release** — with A reverted, the 312 DSA-bwd
   nonzero drains/step block the host (run-ahead collapsed), so when a kick
   fragment releases there is no queued main work to overlap *with*.

## 2. The four levers (T6's L1–L4, labeled hypotheses — not measurements)

| # | Lever | Coupling it breaks | Design notes for v4 |
|---|---|---|---|
| L1 | **Dedicated/priority comm stream for the kick's A2As** (revisit E7) | 2 | v3 assumed comm-stream sharing was free ("stream 83/79 stays ~100 % busy either way") — true for bandwidth, FALSE for FIFO gating: the kick's A2A sits behind the main bwd's queued A2As. A dedicated kick comm (the W1 second-communicator pattern, now proven) lets the kick's comm run in the comm windows' slack. NB the W1 PR-review fix: multi-EP-group worlds need the guarded `new_group` path (ship-w1 branch carries it). |
| L2 | **`last_bwd_end` recorded at the last kernel that actually touches the freed blocks** — or replace the (b) edge with pool versioning / a side-scoped `torch.cuda.MemPool` | 1 | The (b) edge exists only for the allocator reuse edge (DESIGN_W3V3.md §3 E3/E5/E6). A side-scoped MemPool (spec §5's noted follow-up) makes the edge structural: side-pool blocks never shared with the compute stream ⇒ no cross-stream reuse edge ⇒ the (b) wait deletes entirely. That is the cleaner v4 form; the event-refinement form (record at last-touching-kernel) is the fallback if MemPool scoping misbehaves with the NCCL/TE allocators. |
| L3 | **Restore host run-ahead — the A-v3 re-arm is a PREREQUISITE experiment** | 3 | With the 312 DSA-bwd drains async (A-v3), the host stays deep ahead and gate-release meets queued main work. W3 capture likely depends on A-v3; the A-v3 re-arm runs tonight as the final experiment (its status upgrades from "tail item" to "W3-v4 prerequisite"). v4 design work should NOT final-lock L1/L2 until the A-v3 read shows the post-drain regime. |
| L4 | **Side-stream priority** (`torch.cuda.Stream(priority=−1)`) | 2 (partial) | Deferred in v3 as a controlled variable; in v4 it is a first-class knob to test AFTER L1/L2 land (priority helps a released kick win SM slots against main kernels; it cannot fix FIFO gating on its own). |

## 3. Tooling need (T3's hole, from curie)

The consume-stall row was NOT MEASURABLE in the pre-registered idle-gap form
because the trace carries NO `gpu_user_annotation` windows for the
Lookahead functions (cpu_op windows are host-skewed by run-ahead). **v4's
patch must emit GPU-track annotations for the LookaheadCheckpoint backward
window** (nvtx/`torch.profiler.record_function` on the device-side path or
the profiler's gpu_user_annotation instrumentation) so T3 is measurable on
the next capture. This is a patch-level requirement, not an analyzer
workaround.

## 4. v4 acceptance frame (delta from the v3 frame, rest carries over)

- T1 capture bar unchanged (≥50 % side-stream concurrency); the v3 arm's
  0.9 % is the baseline to beat.
- NEW bar: kick fragmentation — fragment count per kick ≈ 1 (v3: ~5);
  measured by the T6 cluster analysis (durable analyzer).
- T3 consume-stall measurable via the new GPU-track annotations (§3);
  bar unchanged (mean ≤10 ms idle-gap form).
- L-bars re-anchored to **75/mb** (kicks == hits == 74/mb, misses == 1/mb).
- Memory/numerics bars unchanged; 16k stays HARD OFF.
- Gate: v4 ships only if A-v3 (L3) has landed AND T1 + fragmentation bars
  pass on the same boot.

## 5. What v4 is NOT (scope guards)

- Not a re-litigation of the win model: −8…−12 s at 60–80 % capture stands
  as the model; the HBM term stays measured-not-assumed (unmeasurable at
  0.9 % capture — needs a capturing arm to measure).
- Not a memory redesign: the depth-2 bound and the +25 GiB class held under
  v3; L2's MemPool scoping changes WHERE the retention lives, not the
  intrinsic ~11.5 GiB.
- Not tonight's work: no patch exists until L3's read and helmholtz's
  go-ahead.
