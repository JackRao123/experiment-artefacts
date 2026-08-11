# T6 WAIT-BY-CAUSE — W3-v3 canary rank0 trace (the v4-or-close discriminator)
Author: curie · Date: 2026-08-10 · Trace: incoming/w3v3_rank0.pt.trace.json
(md5 8acd46a6175d7c7b0b3139756aee847f, verified both sides). 1.79M slices.
Pipeline validated: my capture cross-check reproduces boltzmann's 0.10s/0.9%
exactly (0.100s of 11.29s side-stream time overlapping any other GPU work).

## QUESTION (helmholtz): what does each kick's first kernel actually wait on —
## the input event (a), the last_bwd_end allocator edge (b), or something else?

## ANSWER: (b)-class GPU event gating — NOT (a), NOT host-lateness — with a
## measured refinement: the gate's completion is COMM-TAIL-DELAYED, and after
## release the kick FRAGMENTS on shared comm/expert stream slots.

Evidence (all GPU-timeline, host/GPU skew excluded by construction):
1. Side-stream (103) work fragments: 93,888 kernels / 11.29s drain as ~1,579
   gap-delimited clusters (>2ms), NOT 296 clean kicks — ~5 fragments per chunk
   kick. Fragment spans p50 3.5ms. The kick cannot run through once started.
2. 1,571/1,579 fragment first-kernels start at gap ≈ 0.00ms (p90 = 0.01ms)
   after the end of the latest other-GPU-work kernel. Only 8/1,579 start while
   other work is running. Launch-lag p50 = 39.65ms (host pushed ~40ms EARLY;
   the side stream was free for ~11ms of that) => the kernels sat GPU-side
   with UNSATISFIED dependencies until the other work ended. Not host-bound
   (4/1,579), not side-FIFO (prev fragment ended ~11ms prior).
3. Gate-release kernel identity (the work whose end coincides with fragment
   starts): 1,184/1,579 = NCCL SendRecv on stream 79 (token A2A);
   384 = expert-stream GEMMs (51/161-164); 4 = stream 7 (the compute stream
   proper); 4 unattributed.
4. Ruling on the design candidates:
   - (a) input_event: completes at end of fwd(X) — long before. NON-BINDING.
   - (b) last_bwd_end: recorded on the compute stream at chunk-backward end —
     but the compute stream's chunk-end marker sits BEHIND the chunk's A2A
     tail (the compute stream waits on the comm result), so the (b) event
     completes only when the chunk's stream-79 comm tail drains. Consistent
     with the measured gate-release coincidence (75% stream-79 ends, 24%
     expert-GEMM ends, ~0% stream-7 ends). BINDING, comm-tail-delayed.
   - "something else" — mid-kick re-gating: each fragment past the first
     re-waits on shared-stream slots (the kick's own MoE A2As share the comm
     stream FIFO with the main bwd's A2As — E7 "by design" — and its expert
     GEMMs share the expert streams). Same signature, per-fragment.
5. Why no overlap once released: at gate-release the main queues are EMPTY
   (host drain-throttled — A reverted, 312 DSA-bwd nonzero drains/step block
   the host; run-ahead collapsed). The fragment runs solo into the drain gap;
   main work arrives after it drains. Capture 0.89%.

## CAUSAL CHAIN (for the v4-or-close call)
A reverted => DSA-bwd drains block the host => shallow main queues;
(b)-edge completion is comm-tail-delayed => kick releases exactly when the
GPU goes idle; kick fragments re-gate on shared comm/expert FIFO slots =>
each fragment runs solo in a drain gap => 0.9% capture. v3's ordering fix
DID remove the whole-backlog wait_stream chain (kicks now release in their
local neighborhood, not behind the phase tail) — the residual serialization
is (i) the comm-tail-delayed (b) edge, (ii) shared comm/expert stream FIFO
for the kick's own comm/GEMMs, (iii) no queued main work at release time.

## V4 LEVERS (labeled hypotheses for fermi/helmholtz, not measurements)
- L1: break the comm-stream FIFO sharing — dedicated/priority comm stream or
  stream-79 priority for kick A2As (revisit E7).
- L2: record last_bwd_end at the last kernel that actually touches the freed
  blocks (not the chunk stream tail), or replace the (b) edge with pool
  versioning / a side-scoped MemPool (spec §5's noted follow-up).
- L3: restore host run-ahead — the A-v3 re-arm is a PREREQUISITE experiment:
  with drains async, the main queue stays deep and gate-release meets queued
  main work. W3 capture likely depends on A-v3 (tonight's invalidated arm).
- L4: side-stream priority (spec R2 lever, deferred in v3 as a controlled
  variable).

## CONTEXT ROWS (reported-under-every-outcome, from the same trace)
- T2 kick start-delay / localization: side work lands in the bwd phase
  (boltzmann localization 99.5%); fragments interleave with main bursts.
- T3 consume-stall: NOT MEASURABLE on this artifact in the pre-registered
  (A1.3 idle-gap) form — the trace has NO gpu_user_annotation windows for
  LookaheadCheckpointFunctionBackward (cpu_op windows are host-skewed by
  run-ahead; my probe computed n=8 garbage). Recommendation: next capture
  emits GPU-track annotations for the Lookahead functions. NOT a FAIL — the
  row is context; the verdict rode on T1.
- T5 HBM dilation: expected ~nil at 0.9% capture (nothing co-runs); quick
  matched-kernel check queued separately.
- Comm coverage (corrected, proper interval intersection): side work covers
  1.045s of 26.73s bwd NCCL wall = 3.9% (an earlier 30% figure from my
  first-pass script was a walk-back bug — discarded).
- L3: CLOSED by grothendieck's boot-41 counter line (sweeps==0, fallbacks==0).
  M2: trims==0 confirmed. L4: cumulative avg reached 59.7ms <60 at w7 (and
  the marginal recovery stands: steady band 43.5-47.8ms); the recurring
  ~2947ms max = the step-boundary structural wait documented in the verdict.

## CAVEATS
- rank8 trace unavailable (CUPTI thread-init); single-rank basis, margin ~55x.
- Launch-correlation matched for 987/1,579 fragment first-kernels (the rest
  launched outside the capture window or correlation reuse); classification
  does not hinge on launch data.
- Clustering threshold 2ms; results stable across 2-50ms thresholds
  (881-1,579 clusters, same gate signature).
