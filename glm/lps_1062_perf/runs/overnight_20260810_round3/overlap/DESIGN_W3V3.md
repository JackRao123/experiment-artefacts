# W3-v3 — lookahead recompute with input-dependency-only kick ordering

**Author:** fermi (design) · **Date:** 2026-08-10 · **Ticket:** LPS-1062
**Lineage:** W3 v2 (patch `6f08c5dc…`) → v3 (patch `05dda37f…`,
`runs/overnight_20260810_round3/overlap/patches/w3-lookahead-recompute-v3.patch`). Design base: `DESIGN_helmholtz.md`
§7; verdict record: `ESTATE_NOTES_minkowski.md` (2026-08-10 + post-succession
addendum).

---

## 0. TL;DR

W3-v2's mechanism was PROVEN on-box (stash hits, kicks in the right windows,
canary in band) and its win REFUTED as implemented: kicks 99.9 % serialized,
0 wall for +25.8 GiB. The gate measurement then came back POSITIVE: the bwd
phase carries ~35.9 s/step of SM slack (83.5 % of kernel time <10 %
occupancy; the ~27.5 s/rank of SendRecv comm windows at ~0 % occupancy —
SMs entirely free), so the serialization was **ordering, not hardware**.
v3 removes the one ordering defect: the kick's `wait_stream(current)`
(whole-compute-backlog) becomes a wait on exactly two events — the kicked
chunk's **input event** and the previous chunk-backward's **end event**.
Everything else about v2 (registry, stash, RNG discipline, telemetry bars,
eviction) is unchanged. Win model unchanged in structure: −8…−12 s/step at
60–80 % capture; the capture risk is now HBM-bandwidth contention, not SM
co-residency — the canary frame measures both.

## 1. Why v2 serialized (the defect being removed)

v2's kick did `side_stream.wait_stream(torch.cuda.current_stream())` — the
side stream ordered behind the compute stream's **entire pushed backlog** at
kick-push time. Two compounding effects:

1. **Transitive kick chain.** The consume path opens every chunk backward
   with `current_stream().wait_event(done_event(kick L))` — a wait sitting on
   the compute stream's tail. The next kick's `wait_stream(current)` covers
   that tail, so kick(L−1) is ordered after kick(L). Kicks formed a serial
   chain regardless of available slack.
2. **Host run-ahead.** The autograd thread pushes the backward phase faster
   than the GPU drains it (throttled only by the DSA-bwd drains), so "the
   current backlog" at kick time extends arbitrarily far into the remaining
   backward phase.

Result (canary, measured): 10.82 s of side-stream kernel time, 0.01 s
concurrent with other GPU work; wall flat (the kick *moves* the recompute's
kernels — total work is conserved — so serialized kicks move nothing on the
clock). boltzmann's `w3_side_stream`/`w3_overlap` analyzers +
`sm_slack`/`drain_attrib` (durable at `runs/overnight_20260809_dispatcher_hostsync/analyzers/`) are the
measurement tools; curie's independent confirmation:
`runs/overnight_20260810_round3/verdicts/SM_SLACK_RESULT_318g61w.md`.

## 2. v3 ordering rule

Per kick of chunk X (pushed inside `backward(X+1)`, before bwd(X+1) is
pushed), the side stream waits on **exactly**:

- **(a) `input_event(X)`** — recorded on the compute stream at X's first-pass
  registration (end of `forward()`). Covers the saved-input producers. For a
  microbatch's first kick this also transitively covers the previous
  microbatch's entire backward (the event was recorded in this microbatch's
  forward phase, which follows it on the FIFO compute stream).
- **(b) `last_bwd_end`** — recorded on the compute stream at the end of the
  most recent chunk backward (i.e. bwd(X+2)'s). Covers the **allocator reuse
  edge** (§3). Absent at a microbatch's first backward — no freed side-pool
  graph exists yet this microbatch, so there is no edge to cover.

Both events complete long before kick time in steady state → the wait is a
no-op and the kick starts immediately, co-running with bwd(X+1)'s window —
the intended overlap. The consume side is unchanged: `backward(X)` waits
`done_event(kick X)` (off the critical path when the kick fit the window) and
`record_stream()`s the stashed outputs.

Also moved in v3: the `distribute_saved_activations` gather now runs **inside**
the side-stream context (after the two waits). In v2 it ran on the compute
stream just before the wait; under the two-event rule a compute-stream gather
pushed at kick time is not covered by either event, while the kick reads its
output on the side stream. Inside the context it is stream-ordered (and its
own input — the saved split chunk — is covered by (a)). Inert at the golden
TP1; correct-by-construction at TP>1.

## 3. Cross-stream edge audit (helmholtz consideration 2 — the place "pure
scheduling" could silently stop being true)

v2's backlog wait made every cross-stream edge safe by brute force. v3 keeps
exactly the edges that exist, enumerated:

| # | Edge | Covered by |
|---|------|-----------|
| E1 | Kick **reads** saved inputs (compute pool, produced in fwd) | (a) input_event |
| E2 | Kick reads the distribute-gather output (produced at kick time) | moved inside the side-stream context (stream-ordered) |
| E3 | Kick **writes**: fresh side-pool allocations reusing blocks freed from the graph consumed two backwards ago (e.g. kick(X) vs chunk X+2's freed graph), whose queued bwd kernels may still be reading them | (b) last_bwd_end |
| E4 | Kicked graph's **outputs** read by bwd(X) on the compute stream | done_event wait + `record_stream(compute)` on each output (v2, kept) |
| E5 | Kicked graph's **internal saves** (side pool) read by bwd(X) on the compute stream, then freed host-side | the blocks free host-side only after bwd(X) is fully pushed, so no kick inside backward(X) can reuse them; the earliest possible reuser is the kick inside backward(X−1) (kick(X−2)), whose (b)-wait is on bwd_end(X) — exactly the reader set. No per-tensor record_stream needed (that alternative costs ~1 extra transient graph of pool growth + ~31k event records/step — rejected) |
| E6 | Cross-microbatch reuse (side pool persists; mb i+1's first kick reusing mb i's freed graphs) | (a): the first kick's input_event was recorded in mb i+1's forward, which follows mb i's whole backward on the FIFO compute stream |
| E7 | NCCL: the kick's MoE A2As on the EP PG comm stream | torch's ProcessGroupNCCL records input/output streams internally (unchanged from v2); comm-stream sharing with bwd(X+1)'s A2As is by design (stream 83 stays ~100 % busy either way) |
| E8 | Host-side state (RNG fork/set/restore, fp8 snapshot, FIX-C replay cache, DSA carrier) | single autograd thread (v2, kept); V4 audit stands (re-assignment only, replay reads only) |

The empty_cache trim knob (§5) releases free segments from all pools; it runs
only at a microbatch boundary sweep point (no side-stream work in flight) and
cudaFree's implicit sync makes it safe by construction.

## 4. In-flight depth bound (helmholtz consideration 1)

The dependency structure — not the v2 serialization — caps live kicked graphs
at **2**: kick(X) is issued only after X+1's stash is popped (inside
`backward(X+1)`), and X's stash is popped before kick(X−1) is issued. v3
changes *when* within the window a kick runs, not *how many* are live. Peak
side-pool demand therefore stays ~2 chunk graphs (≈23 GiB at 131k), matching
the v2 measurement's structure. **But the +25.8 GiB figure was measured under
the serialized regime and does NOT carry over as a number** — the canary must
re-measure peak reserved (poller) and re-capture a 16-rank allocator snapshot
(boltzmann's round-3 method) under v3. Bar in §7.

## 5. Memory model + the retention knob (requirement c) and 16k (requirement d)

Settled decomposition at 131k-d4 (v2 arm, snapshot-verified): **+25.8 GiB =
~11.5 GiB intrinsic** (two ~11.4 GiB chunk graphs live at the mid-backward
peak; MoE-pipeline saves dominate) **+ ~14 GiB allocator retention** on the
side-stream pool (26–30 GiB free-cached, uniform across ranks). The intrinsic
part scales with shape; the retention part is a knob.

v3 specs the knob: **`BT_MOE_LOOKAHEAD_TRIM_EVERY=N`** (default 0 = OFF) —
`torch.cuda.empty_cache()` at every Nth microbatch boundary (the registry
sweep point; N=4 = once per step at d4). This is a *global* trim (the caching
allocator has no per-pool release): it also drops the compute pool's ~29.5
GiB free-cached, so the cost is a device sync + cudaMalloc churn that the
canary must measure before any reliance (window line carries a `trims`
count). At 131k the knob stays OFF — +25.8 GiB vs 44.9 GiB headroom already
clears the ≥10 GiB bar. The knob exists for the 16k-enablement workstream;
the cleaner follow-up (a `torch.cuda.MemPool` scoped to the side stream so
the trim is pool-local) is noted but not built — global trim is the honest
primitive available in the pinned torch.

**16k×d32 stays HARD OFF.** Corrected model there: MoE/MLP terms ×4 (32,768
tok/rank/mb) ≈ 13–14 GiB/chunk intrinsic + retention + fragmentation —
genuinely borderline against the 16k headroom. Any 16k experiment needs its
own justification + measurement (trim knob likely required), and is not part
of this canary.

## 6. Win model with the HBM-bandwidth term (requirement b)

Per layer-bwd window at 131k-d4 (300 windows/step), C′-on profile:
recompute ≈ 41 ms (23 comm + 18 compute), bwd ≈ 65 ms (23 comm + 30–35
compute + tails), serial ≈ 105–110 ms. v3 hides the kick under the following
bwd window:

```
wall_window(c, d) ≈ 65 + (1 − c)·41 + d·(bwd compute)
```

- c = capture (fraction of the kick's ~41 ms that overlaps the window's
  non-critical resources). The SM-slack measurement says co-residency is
  feasible over most of the window (83.5 % of kernel time <10 % occ; the
  ~27.5 s/rank/step of SendRecv windows at 0 % occ are the richest territory
  — the kick's compute lands there; its A2As share stream 83 by design).
- d = dilation on bwd compute from co-running — **the HBM-BW contention
  term**. curie's caveat stands: the occupancy figures are launch-config
  co-residency estimates, NOT bandwidth measurements. Total HBM bytes/step
  are conserved (same kernels); only time-compression raises instantaneous
  demand. Expect dilation concentrated in the full-width GEMM/attention
  blocks (35 % of compute time at 90–100 % occ — no-go zones where the kick
  timeshares) and little elsewhere (the <10 %-occ tail is regs/smem-limited,
  grids ≥148 blocks — latency-bound, not BW-saturated).

At c = 60–80 % and d small: −25…−37 ms/window × 300 ≈ **−7.5…−11 s/step**
(the memo's −8…−12 at 60–80 % capture stands; floor ≈ max(comm 46, compute
~50) + tails ≈ 55–65 ms/window). Break-even on d: the win vanishes when
d·(bwd compute ≈ 10 s/step) ≈ c·(recompute ≈ 10.8 s/step), i.e. d ≈ c — so
any capture at all nets positive unless co-running nearly doubles bwd compute
time; the realistic risk is a *smaller* win, not a negative one. The canary
measures c and d directly (§7) rather than trusting this bound.

## 7. Telemetry + pre-registered canary bars (for curie's frame)

**Log bars (unchanged from v2 — counter semantics identical):**
kicks == stash_hits == 78/mb (312/step at d4); stash_misses == 1/mb (4/step;
the last chunk recomputes inline); sweeps == 0 (a non-empty sweep is a
structural bug, halt); fallbacks == 0. The window now rolls on **300 chunk
backwards** (~1 step) and is labeled truthfully (the v2 events-based window
caused the false 150/mb reading — fixed in v3, regression-netted by the
estate rule). Gate-on config assert (dropout == 0) unchanged.

> **CHUNK-COUNT CORRECTION (2026-08-10, curie, from the v3 canary arm):** the
> measured count on the C′-ON stack is **75 chunks/mb** (74 kicks + 1
> structural inline), not the frozen 78/mb — the 78 was model-derived (78
> hidden layers + MTP) and likely conflated the DSA-layer count with the
> checkpoint-chunk count. The bar's STRUCTURE held exactly (kicks == hits ==
> chunks−1, misses == 1); only the constant was off. Treat the per-mb
> constant as stack-derived-at-runtime, not model-derived, in any future
> frame.

**New in-log signal:** `kick_ms_avg` / `kick_ms_max` per window (CUDA-event
kick duration on the side stream). Reference: ~41 ms inline. Dilation bar:
avg ≤ ~60 ms (≈1.5×); sustained max ≫ that = contention biting.

**Trace bars (boltzmann's analyzers, arm vs C′-on baseline):**
1. Side-stream concurrency (`boltzmann_w3_side_stream.py`): ≥ 50 % of
   side-stream kernel time concurrent with other GPU work (v2: 0.1 %).
2. Consume-stall: gap at the head of each `LookaheadCheckpointFunctionBackward`
   (done-event wait exposure) — avg ≤ ~10 ms/window; larger = kicks not
   fitting the window.
3. bwd-phase compute-kernel dilation, duration-weighted, per occupancy bucket
   (`sm_slack` analyzer re-run on the arm trace): the HBM term d — report it,
   no pre-set pass number (first measurement of its kind here); wall verdict
   uses it.
4. SendRecv-in-bwd-window accounting by CAUSE (drains vs eventSyncs vs
   exposed), never totals (standing rule).

**Memory bars (re-measured, NOT inherited from v2):** peak reserved delta vs
the C′-on baseline ≤ +28 GiB (headroom ≥ ~17 GiB at 131k; ship bar ≥10 GiB);
16-rank allocator snapshot piggyback on the arm boot; `trims` == 0 (knob OFF
at 131k).

**Numerics (requirement e):** unchanged class — pure scheduling (same
kernels, same inputs, same RNG, same reduction orders; allocation addresses
differ, which no kernel's result depends on). In-process gates stay
hard-bitwise (CPU suite 40/40 green incl. the dropout RNG-isolation proof);
ship verdicts through the house band (≤2e-3 pass / >5e-3 stop, matched
`--warmup-datums` MANDATORY — mismatch = INVALID, re-run) with the 20-step
drift cover. Two-tier verdicts: mechanism (this section) first, wall second
vs the box-1 C′-on anchor (666 s class); never quote uncalibrated wall.

**16k:** not armed, not measured, HARD OFF (§5).

## 8. Risks / open items

| # | Risk | Handling |
|---|------|----------|
| R1 | HBM-BW contention shrinks capture (the open second-order term) | measured per §7.3 + kick_ms dilation; win model §6 bounds it |
| R2 | Consume-stall if a dilated kick exceeds its window | §7.2 measures; fallback lever if seen: side-stream priority (not in v3 — controlled variables) |
| R3 | Allocator-edge mis-analysis (E3/E5/E6) | the audit above + canary is the test; the v2 wait_stream form is one revert away (patch lineage preserved) |
| R4 | Peak mem differs under concurrency | re-measured (§7 mem bars); trim knob specced for 16k |
| R5 | kick_ms event overhead | 2 event records/kick ≈ µs-scale × 312/step — negligible; events are timing-enabled, waits unaffected |
| R6 | CDMC: boxes run UNSET (8 connections, /proc-verified, bohr) — v3 assumes that; a CDMC=1 boot would re-serialize (single hardware channel) | launch-env truth stays `/proc/<pid>/environ`; a =1 boot is out of scope (bohr: low priority) |

Open follow-ups (not v3 scope): memo §7b memory-estimate correction (the
2–4 GiB undercount; the settled model is in ESTATE_NOTES); W3 telemetry
relabel — DONE in v3 (window rolls on chunk backwards); option-6 stays
un-parked behind v3 per the W2V2 stub's disposition; W2-v2 seq-bump re-weigh
after the v3 readout (non-additivity note in the stub).

## 9. Patch identity + verification

- `runs/overnight_20260810_round3/overlap/patches/w3-lookahead-recompute-v3.patch` — md5 **05dda37f68f3113747a812e357d9b552**,
  638 lines: `megatron/core/lookahead_checkpoint.py` (NEW, 588 lines) +
  `megatron/core/recompute.py` (hunk byte-identical to v2 — the call site is
  unchanged; index-line hashes corrected after the v2 post-image staleness
  was found at regeneration — see W3_PATCH_NOTES.md).
- Base: vendored mcore @ 57efae08b + FIX A/B/F + FIX C (recompute.py base
  blob d43d8621b), same as v2. Verified: `git apply --check` clean on the
  reconstructed base; applied tree reproduces the Mac tree byte-for-byte
  (lookahead_checkpoint.py md5 7304202b…, recompute.py 0ec487cd…).
- **Application path (corrected 2026-08-10 — the box trip):** this is a FULL
  patch against a NO-W3 base, NOT a v2→v3 delta. Against a tree already
  carrying v2 it fails by design (new-file collision on
  lookahead_checkpoint.py; the recompute.py hunk already applied). On the
  v2-in-tree box clone: **revert v2 first** (`git apply -R` of v2 patch
  6f08c5dc…; check = lookahead_checkpoint.py absent + recompute.py back at
  d43d8621b), then apply v3 (post-state: 7304202b… / 0ec487cd…).
  grothendieck's on-box resolution followed exactly this (md5-proofed).
- Tests: `tests/test_w3_lookahead_checkpoint.py` 40/40 green on Mac CPU —
  all v2 sections (bitwise parity under dropout=0.5, counters, eviction,
  fallback, RNG round-trip, integration-arity + AST call-site guard) plus
  **sec8 v3 ordering guards**: no `wait_stream` anywhere in the module; the
  kick waits exactly the two events; the distribute gather sits inside the
  stream context; forward records `input_event`; backward stores the
  `last_bwd_end` event. (Events are CUDA-only; CPU runs the inline path —
  semantics preserved, overlap absent, as in v2.)
