# HANDOFF — LPS-1062 orchestrator (cauchy → lebesgue)

Written 2026-08-13 ~3:3x PM CDT by cauchy (succeeded borel ~01:40 AM).
Jack ordered this succession verbally before going out ("pass off to
lebesgue when you are ready, your context is getting full"). You are
lebesgue, the fourth orchestrator of this push (maxwell → borel → cauchy
→ you). Succession rule (memory-indexed): take over at the current
atomic boundary; in-flight work does NOT stop.

## Read first
1. NOTEBOOK.md from "SUCCESSION COMPLETE: cauchy holds the reins" down —
   the full day arc. It is COMPLETE and current; trust it.
2. REPORT_MORNING_0813.md — the deliverable. Status: FINAL, zero TBDs,
   header says so. It reopens ONLY for the new program below (§5/§6
   rewrite + E1 result). Keep its chronology-honest style: overturned
   framings stay in place, marked, never silently rewritten.
3. Memory index: plain-language updates; measure-first (pre-register
   every probe reading BEFORE results); fleet-succession-promptly;
   lightweight-review-policy (docs = no reviews; box-bound code = ONE
   disposable-subagent review; eyeball trivial diffs yourself).

## The day in one paragraph
Overnight mission achieved (918/1052 tok/s/GPU) but the parity protocol
caught corruption. Morning fix campaign root-caused EVERYTHING to one
stale kernel wheel (cudnn-frontend 1.26.0+dsatopk1, a known TMEM race,
shipped by a vendored-wheels build bug — papercut pc_c89b5acdeed2, top
queue follow-up). Fix = bump to 1.27.0. Fully validated: every topology
× padding cell agrees to ~5e-5; parity Gates 1+2 PASS; re-anchored perf
d16 ≈ 984 (947-1023 window) / d4 ≈ 880; peak mem −32 GiB at every rung.
The 0.19 "topology gap", the slot corruption, the nondeterminism, and
the tail-pad corruption were all that one wheel.

## JACK'S CURRENT ORDERS (verbal, before going out — he is OUT now)
Work autonomously. Help the report. Specifically:
1. **"See why DeepEP is not making it faster — realistically it should
   be."** L5 closed at −12.6% on the stale wheel; Jack wants the WHY,
   not just the verdict.
2. **"The overlap stuff."** The 25-28s/step overlap prize (report §5).
3. The convergence thesis I launched the program on (validate, don't
   assume): DeepEP standalone loses because 131k a2a is bandwidth-bound
   at line rate — but DeepEP's async hooks are the designed transport
   for comm/compute OVERLAP, and the fixed wheel's −32 GiB may have
   removed the overlap flag's memory wall. If the pieces land, L5
   reframes from "not the lever" to "not the lever ALONE" and §5/§6 get
   rewritten around: contract shim (serre scoped, 0.5-2d) + E1-fits +
   flex-as-overlap-transport = executable path to the prize.

## IN FLIGHT RIGHT NOW (launched ~3:2x PM, results land on your watch)
- **doppler, PROBE 1**: E1 re-measure on the fixed wheel (selective-
  recompute memory config, d2-scale, >265 GiB abort guard). Question:
  does the overlap memory config now fit under 275 (old: 258.8; if the
  −32 GiB workspace applies → ~227 = fits)? PROBE 2 (tuned-flex d4 A/B
  vs the ~880 fixed-wheel anchor) is QUEUED pending serre's audit.
- **gauss**: trace decomposition answering "why isn't flex faster" —
  where the 5.1s/step delta actually goes (DeepEP kernel delta is only
  ~0.4-1.4s; suspects: SM contention, serialized windows, probs
  latency) + the roofline argument checked against trace numbers.
  Deliverable: memo + report-ready paragraph.
- **serre**: DeepEP config audit (was L5 even TUNED? num_sms, buffers,
  mode, chunking) → tuned-flex probe spec or a plain "no knobs";
  PLUS the overlap-connection scoping addendum (does the overlap
  executor use flex async hooks).
- **hausdorff**: final evidence sweep mid-flight — pings you when
  BOX_A_ARTIFACT_MANIFEST.md is final → update report §9 sweep note to
  "verified" + close task #6.

## Fleet state
- doppler (box): armed, excellent all day. Box w56lorq HELD idle-armed:
  fixed wheel venv, exact tree 73c24b00+TF32, headline config READY.
- gauss, serre: reactivated on the new program (above).
- weierstrass: standing by; owns MORNING_MERGE_QUEUE.md (9 slots + a
  future section; wheel-staleness build fix on top).
- hausdorff: mid-sweep.
- **poincare: DARK since ~13:0x** (unresponsive through the gate runs;
  doppler covered mechanics per spec). If they resurface they own
  legs/analysis again; their session may need Jack's attention.
- Send-message gotcha: single quotes (backticks in double quotes
  expand). Crossed-message risk: sequence sends against latest state.

## Jack decisions PARKED (his, do not act)
Merge-queue GO (9 items, PR 987 first); F1 async-save escalation
(package parked, never file without his word); upstream cudnn note
(parked, we are fixed); box final stop (after the probes; manifest
makes stop safe once hausdorff finishes); CPFS dirty-mcore disposition
(gauss doc §7; the gate-flip probe patch is also still on the box tree,
inert with env unset — revert it before any parity-class boot).

## Operational gotchas (day's additions; maxwell/borel lists stand)
- TASK BOARD IS PER-SESSION: rebuild yours from this doc (mine dies
  with me). Open items: #6 final sweep (in flight), everything else
  completed or superseded-pending-Jack (#2 cache A/B, #3 evidence tail
  — resurrect only on his word, and they need re-baselining on the
  fixed wheel).
- The trainer's ARMED-IDLE posture pins GPUs at 100% (posted NCCL recv)
  — looks exactly like a wedge on nvidia-smi. py-spy before concluding.
- attention_backend=unfused at 131k stalls collectives past the NCCL
  heartbeat watchdog (papercut filed) — raise it for any reference-path
  boot.
- Snapshot failure logs BEFORE relaunch (start_trainer.sh clobbers).
- Probe evidence discipline: pre-register readings, distinct run labels
  (-fe127 convention), never overwrite prior evidence sets.
- One box-seeker at a time; 232gxvq + team_qzr5p83 off-limits.

## Your first moves
1. Announce yourself to doppler/gauss/serre/hausdorff/weierstrass (+
   poincare in case they wake). They know succession is coming.
2. Rebuild the task board.
3. Take the E1 verdict + gauss/serre deliverables as they land; update
   report §5/§6 per the convergence thesis IF the evidence supports it
   — pre-register the E1 reading before doppler reports if you can
   still beat the result.
4. When the program lands and Jack is still out: the natural next
   probes are (a) tuned-flex A/B if serre finds knobs, (b) the
   golden-padded belt-and-braces leg (one PP1/CP16 boot) if box time is
   free. Keep every run pre-registered and evidence-first.
