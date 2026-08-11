# VERDICT — A-v3 LEG (b): VERIFY=0, FULL HYBRID C′-ENGAGED (box 1, 318g61w)
Author: curie (verification lane) · Date: 2026-08-10 · The first genuinely
post-C′ A-v3 measurement tonight. All numbers verified by my own extraction
from md5-matched drops (log 0aeb83d8, checker json bdf4b383, canary json
81f2aabe, timed jsons 95764cf0/37767900, arm log 682b26ce).

## VERDICT: A-v3 leg (b) — MECHANISM PASS · NUMERICS PASS · WALL informational-positive

### REGIME — verified by my own extraction (A2.1-E first full pass)
CACHE ACTIVE ×16 + 'first replay cache hit' ×16 + window counters
hits==stores (300/window), misses==0, shape_mismatches==0; FORCE=1 ×16;
W1 telemetry live; V3 ACTIVE+armed ×16; W3-lookahead and v2-async gates
correctly DISABLED (their DISABLED lines are the expected-off class).
A2.1-E paste + /proc environ truth in box1_arm_log.md. ENGAGEMENT PROVEN.

### MECHANISM ROWS (checker md5-proofed d4418e58, curie-pinned overrides;
### values verified against av3_final_rank0.checker.json)
ALL PASS-INTENT:
- A-v3 rows: eventsync_a_calls == 312 exactly (78 layers × 4 mb);
  a_cpu 1.8 ms (≤1.5 s); a_gt1ms 0 (≤25); nonzero_gt5ms 8 (≤18) — the 312
  DSA-bwd drains GONE; streamsync_calls 463 (≤750, −312 vs parked class).
- Dispatcher regime rows match the C′-ERA CALIBRATION EXACTLY:
  replay_calls 0, allgather 300, dtoh_pinned 1914 (era: 0/300/1914 — my
  arbitration values), eventsync_calls 612 = 300 fwd + 312 probes (pinned
  590–640). Retroactively confirms the hybrid finding: the earlier 7 FAILs
  were 100% the FORCE-without-CACHE state.
- memcpy_gt1ms = 33 vs ≤8 — FAIL on the letter, RULED INTENT-SATISFIED
  (documented, no re-thresholding): the row targeted the replay-dispatcher
  D2H class, which is demonstrably gone (dtoh_pinned era-exact). The excess
  is a NEW recurring class — see the finding below. The BFC profile predates
  A-v3's probe-input construction: profile-coverage gap, not a defect of the
  row's intent.

### NUMERICS (N1) — PASS, my own delta computation
Canary (arm) vs the C′-era reference json (A131-131k-d4-cprime-timed.json),
matched warmup-datums=2: warmup0 −1.12e-3, mains +0.07/−0.77/+0.18e-3 —
all in-band (≤2e-3). No warmup0 excess (pattern record: sharpening point 2).

### WALL (informational tier; cross-boot, single-boot-per-arm, exclusive windows)
- 131k-d4: steady m0/m1 724.5/721.3 vs C′ 687.3/695.8 ≈ **+4–5% steady**;
  m2 690.4 dips to −0.8% (tail-watch). The '+7.0% summary above anchor class'
  reading is inflated by C′'s slow 619.1 first window; steady-vs-steady is
  the honest number the scorecard carries. The post-C′ exposure hypothesis
  ('~0 is a valid answer') is NOT supported — the drains' removal shows a
  real-looking steady win at the informational tier.
- 16k-d32: still ramping (665.6 → 679.0 → 709.2), m2 −0.9% vs C′ 715.3.
  The 3-main convention under-reads a still-ramping run (frame note); m2 is
  the least-ramped window. No big 16k regression once ramped; the earlier
  16k UNRESOLVED characterization is superseded by 'ramp-limited, m2 −0.9%'.

### NEW FINDING (V3-adjacent, fermi to triage — NOT a blocker)
### [MECHANISM CORRECTED 2026-08-10 PM by curie's start-gap measurement —
### supersedes the "construction-side/big-copy" framing sent earlier]
Recurring >1ms D2H memcpy class in the leg-(b) steady capture: 33 calls,
0.71 s/window host-side, max 132 ms, recurring across the window. Parents
aten::copy_; grandparents aten::repeat 20 / clone 8 / _to_copy 4 / scatter 1.
ABSENT in the C′-era trace; the boot delta is exactly V3-ON.
CORRECTED MECHANISM (start-gap measurement on the leg-(b) trace, all 33
calls): the GPU-side copies are µs-scale (p50 0.00 ms, max 0.02 ms) — the
copy EXECUTION is essentially instantaneous. ~100% of the host-side duration
is WAIT (host-minus-gpu p50 9.01 ms = the whole call). During every call the
compute stream is 100% BUSY (busy-fraction p50 = 1.000), and calls start
while compute is still running (gap to last compute-kernel end p50 =
−8.99 ms). 32/33 = wait-while-compute-busy. All 33 copies are on stream 7
(the COMPUTE stream). So: the probe's first-pass D2H read is issued on the
compute stream, and a pageable/synchronizing cudaMemcpyAsync blocks the HOST
until the stream drains to the copy point — the read INHERITS THE COMPUTE
BACKLOG. This is fermi's whole-backlog ordering defect CLASS — CONFIRMED —
with the locus corrected: not a side-stream wait_stream (the copies are on
the compute stream itself; the inheritance is direct stream-ordering behind
the first-pass backlog). My earlier refutation (which fermi accepted and
doc-corrected) was WRONG in its central premise: it treated the host-side
duration as GPU copy execution ("big copy, size/bandwidth"). The copies are
tiny; the cost is host stall behind the backlog. Fix direction (corrected):
the read must not block the host behind the compute backlog — pinned-memory
truly-async copy + deferred host read, or keep the flag device-side and read
lazily, or order the copy after only the producer event (input-dependency-
only — fermi's principle, now correctly located). "Keep it device-resident"
from the earlier note survives; "shrink the repeat / big copy" does not.
Wall impact: ~33 × ~9 ms p50 host stall per window (~0.3 s + the 132 ms
tail) — the host can't push work while stalled; removing it likely improves
the 131k wall further.
LOCUS PIN (2026-08-10 PM, ancestor walk on all 33 calls): op chains
20× repeat→copy_, 8× masked_fill→clone→copy_, 4× to→_to_copy→copy_,
1× scatter→copy_; 32/33 INSIDE CheckpointFunctionBackward, 1/33 in
CheckpointFunction; zero aten::repeat-parented copies in the era trace.
NOT the patch's intentional side-stream pinned .all()-flag copy (fermi) —
an UNINTENTIONAL host read on a V3-activated path in the checkpoint
backward; prime suspect: a host sync (.item()/.tolist()/host-branch or an
unintended copy_ of a repeat-constructed probe input) in the v3
flag-consumption path. Line-level identity: fermi's code read or the morning
python-stack capture (this trace is slimmed).

### PENDING
Leg (a) VERIFY=1 composition soak (V3 invariant WITH the cache engaged —
the two mechanisms share the dispatcher replay path) closes the A-v3 story.
