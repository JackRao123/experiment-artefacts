## 2026-08-10 — W3 VERDICT + succession (minkowski → fermi)

**W3 canary verdict (boltzmann, via kepler): mechanism-PROVEN, win REFUTED
as implemented.** +25.8 GiB for 0 wall; kicks 99.9 % serialized. Two
causes: (a) the whole-backlog `wait_stream` ordering serializes the kicks;
(b) no B300 SM slack — the bwd-phase timeline is ~97 % kernel-covered (the
C′ drain finding), so there is no idle compute capacity for the kicked
recompute to run in. (My watch-list amendment-2 warning — drains are
kernel-covered, not reclaimable idle — was the right caution.) W3 is
DISARMED from all subsequent arms; 16k stays HARD OFF; scorecard line:
"mechanism proven, win refuted on B300 at 131k, +25.8 GiB, occupancy
question open".

**W3-v3 gate (binding):** any v3 design work is GATED behind boltzmann's
SM-slack measurement from the existing traces. If there is no bwd-phase
occupancy headroom, no ordering scheme wins and W3-on-B300 closes. Do not
start v3 before that lands.

**Option-6 / W2-v2 disposition (my call per the W2V2_DECISION stub —
UN-PARK):** the stub's decision rule fires its else-branch: W3's measured
capture (~0 %, 99.9 % serialized) is decisively < 50 % (rule 3). BUT the
un-park comes with a resource distinction the stub's authors didn't have:
W3 failed because it needed SM SLACK to run extra compute concurrently;
option-6 (bwd-chain reorder for the probs reverse, ~1–1.5 s, the §6.1
engine-issue-order residual) and the v2 seq-bump are COMM-RESEQUENCING
plays — they add no compute and hide existing comm behind existing bwd
windows. Saturated SMs are what comm-hiding needs, so the W3 refutation
evidence is neutral-to-supportive for them; the SM-slack gate does NOT
apply. Sequence: option 6 first (small, targets the understood W1
residual), v2 seq-bump behind it (stub scope: two seq-bumps + wait-aware
wrappers + event plumbing, ~1–2 days; v1 bitwise argument carries over).
One evidence caveat: re-confirm the bwd-phase comm-window cover from the
canary trace under C′-on (the drains analysis has most of it) before
locking v2's win model.

**SUCCESSION (Jack's order, trigger reached 2026-08-10):** my successor is
**fermi** (Kimi K3); the new orchestrator is **helmholtz** (Claude Fable).
Handoff = `HANDOFF_DESIGN_minkowski.md` + these estate notes.

**POST-SUCCESSION ADDENDUM (2026-08-10, boltzmann's SM-slack measurement —
recorded by minkowski as the requested estate entry; the v3 DECISION is
fermi's):** the W3-v3 gate result is IN and it **PASSES** — and it refutes
the verdict's cause (b). Measured from the existing traces: 83.5 % of
bwd-phase kernel time runs at **<10 % occupancy** (~35.9 s of SM
slack/step); the 25 s of SendRecv comm time sits at **~0 % occupancy** —
SMs entirely free during the backward A2A, exactly W3's intended overlap
target. So the v2 serialization was **ORDERING** (`wait_stream` behind the
whole compute backlog), NOT SM saturation — boltzmann's full-width
hypothesis refuted by his own occupancy data. The reconciliation of the
apparent contradiction with the C′ finding ("drains 97.4 %
kernel-covered"): the timeline IS kernel-covered, but at <10 % occupancy —
**timeline coverage ≠ throughput saturation**; the earlier verdict
conflated them. A v3 that orders kicks against input dependencies only has
~35.9 s of slack to hide ~10.8 s of recompute (~3.3× cover). Second-order
term to model BEFORE promising magnitude: **HBM bandwidth contention**
between kicked compute and in-flight A2A. boltzmann's analyzers are
durable at `dispatcher_opt/analyzers/` (`w3_overlap` / `side_stream` /
`sm_slack` / `drain_attrib`). The W3 telemetry events-vs-chunks mislabel
(300-EVENT windows printed as "chunk backwards") is confirmed as the cause
of the false 150/mb reading — a one-line relabel kills the class (queued
for the next patch revision). Scorecard line should read: "mechanism
proven; v2 win refuted by ORDERING serialization (not SM saturation —
35.9 s occupancy slack measured); v3 = input-dependency-only ordering,
HBM-contention term open".

**Method note (boltzmann's sign-off precision, for the handoff):**
COVERAGE, CONCURRENCY, and OCCUPANCY are three different measurements and
each answers exactly one question — mixing them is how a true row
("97.4 % kernel-covered") becomes a false conclusion ("no slack"). The C′
drain-coverage measurement answered a busy-CPU-vs-blocked-on-GPU question
and its wait-conservation arithmetic stands independent of occupancy; the
W3 verdict's error was reading that coverage row as an occupancy answer.
The amendment-2 warning ("drains are kernel-covered, not reclaimable
idle") was right in its lane — the timeline isn't idle; the occupancy cut
completed the picture in its lane — the throughput is mostly free.

# ESTATE NOTES — minkowski (design owner, LPS-1062, from 2026-08-09)

Running log of corrections, rulings, and verification artifacts for the
design estate. Inherited from helmholtz (HANDOFF_DESIGN_helmholtz.md);
orchestrator kepler; box runner fourier; second reviewer hilbert.

---

## 2026-08-09 — CORRECTION: inherited box-state claims unverified

- **Falsified:** helmholtz's "all patches verified apply-clean +
  byte-for-byte" (HANDOFF §2 artifact table) held for Mac↔patch only, NOT
  for the box. fourier's guard-(b) found the box tree carrying **W2 v1**
  (box patch md5 `2b06d11a…`, 865 lines) where v3 (`c94f72e6…`, 967 lines)
  was believed deployed; zero hits for v3 marker comments in the box's
  vendored `token_dispatcher.py`.
- **Standing rule (ratified by kepler):** any inherited box-state claim is
  UNVERIFIED until fourier's full patch-estate md5 sweep (box-vs-Mac, incl.
  staged W3 + A-v3) reports. The artifact table's "verified" column is
  Mac-side provenance only.
- **Consequence:** T2 gate held until W2-v3 surgery (reverse-v1 → verify
  pre-W2 → apply-v3 → verify manifest below) passes by inspection.

## 2026-08-09 — W2 v3 post-surgery verification manifest (issued to fourier)

Patch identity: `w2-moe-a2a-pipeline.patch` md5
`c94f72e62e0431eec915ddcf792f4fa0`, 967 lines, 47334 bytes. Purely additive:
**+810 / −0** across 3 files — `token_dispatcher.py` +713, `moe_layer.py`
+59, `experts.py` +38 (all under `megatron/core/transformer/moe/`).
[CORRECTED per fourier's numstat footnote: my first manifest read
+754 (662/56/36) from an awk `^\+[^+]` count, which excludes added lines
whose content begins with `+` (patch `++` lines) — 56 such lines.
`git apply --numstat` is authoritative: 713/59/38, 0 deletions. Marker
tables below are unaffected — they were counted with `grep -c -F` on tree
files and inclusive `^+` on the patch.]

Every marker below is patch-introduced (patch-adds == Mac-tree count ⇒ zero
pre-existing in base), so each is a clean discriminator. Markers keyed † are
v3-ONLY (hilbert round-1/2 + T2-defect fixes) — they distinguish v3 from v1,
not just W2 from base.

**Phase 1 — after reverse-v1 (pre-W2 state):** all markers ZERO hits in all
three files: `_W2ChunkPlan`, `BT_MOE_A2A_PIPELINE`, `forward_expert_group`,
`w2_local_counts_host`, `w2_global_counts_host`,
`_A2A_PIPELINE_DISARM_LOGGED`, `had_buf`.

**Phase 2 — after apply-v3, expected `grep -c -F` per file:**

| Marker | token_dispatcher.py | moe_layer.py | experts.py |
|---|---:|---:|---:|
| `BT_MOE_A2A_PIPELINE` | 20 | 5 | 1 |
| `_W2ChunkPlan` | 3 | 0 | 0 |
| `w2_local_counts_host` † | 9 | 0 | 0 |
| `w2_global_counts_host` † | 10 | 0 | 0 |
| `_A2A_PIPELINE_DISARM_LOGGED` † | 1 | 3 | 0 |
| `had_buf` † | 2 | 0 | 0 |
| `subset of chunks, not a permutation` † | 1 | 0 | 0 |
| `reason = "overlap_moe_expert_parallel_comm"` † | 1 | 0 | 0 |
| `forward_expert_group` | 0 | 1 | 1 |

**Phase 3 — byte-for-byte tier** (Mac ship-tree file md5s; valid iff the
rest of the estate — base `57efae08b` + FIX A/B/F + FIX C + W1-v1 — matches
box, which the estate sweep covers):

- `token_dispatcher.py` = `3857aeddcd37263252baf630b17ac416`
- `moe_layer.py` = `71301302050b90ffb2ad8ea130ec3f80`
- `experts.py` = `aec9b3b8aafcff7ee7c9c314ce58eba1`

Mismatch with exact markers/diffstat ⇒ drift elsewhere in the file; diff
against the Mac copy to localize. Apply cleanliness: dry-run clean, no
`.rej`/`.orig`, diffstat exactly +810/−0 over the 3 files.

**Surgery outcome (fourier, 2026-08-09): PASS.** Markers 13/13 verbatim,
moe_layer + experts md5 exact, patch identity verbatim. Phase-1 reversal
required `patch -R --fuzz=3` (v1 hunk-1 context spanned bohr's
instrumentation slot lines; `git apply -R` inapplicable); pre-W2 state
proven by W1 reverse-check clean + zero W2 markers + exact diffstat.
Provenance preserved at `w2_surgery_provenance/` on the box (pre-v3 .orig
kept). Gate GO per kepler; fourier executing.

**token_dispatcher.py file-md5 tier — box-expected-different, standing:**
the box legitimately carries `fixc_verify_instrumentation.patch` (4
verify-mode logging blocks) which the Mac ship tree deliberately lacks, so
box token_dispatcher.py md5 ≠ Mac `3857aedd…` by exactly those blocks —
fourier localized the drift by diff, zero W2-content delta. For any future
estate sweep: the md5 mismatch is the EXPECTED state; the diff-localization
is the proof, not the md5. HAZARD (from HANDOFF_BOX_bohr §6): the
instrumentation patch's hunk 3 fuzz-MISPLACES over the W1/W2 tree
(`entry.routing_map_dev = …` before `entry = _ReplayEntry()` →
UnboundLocalError); bohr hand-fixed the placement on-box and the on-box
file is the working authority, but the archive patch remains
misplaced-as-authored pending regeneration — **live owner: boltzmann**
(queued behind his C′ adjudication; hilbert is released — corrected per
kepler, 2026-08-09). Any from-patches tree rebuild must re-apply that
hand-fix or it reintroduces the crash. **RETIRED same-day: boltzmann
regenerated `fixc_verify_instrumentation.patch` from the box tree with a
byte-for-byte round-trip proof (md5 31f8e0bc box↔Mac); stale 191c41ae
quarantined as `.STALE-DO-NOT-APPLY`. Rebuilds need no hand-fix.**

## 2026-08-09 — C′ verdict + W3 watch-list amendment (from boltzmann's adjudication via kepler)

C′ PASS on mechanism (HOLD lifted; A-v3 unlocked in ARM-5). Finding with
direct W3 consequence: under C′-on, the 312 DSA-bwd drains absorb ~24 s of
redistributed CPU wait (was ~19 s with the replay throttle present), 97.4 %
GPU-kernel-covered, wall-flat — **wait conservation, not new work**.

W3 canary watch-list amendments (binding at readout):
1. **Baseline discipline:** the W3 canary's deltas are vs the C′-ON profile
   (ARM-5 baseline), never the pre-C′ anchors. The bwd-phase timeline W3's
   lookahead targets now carries its wait in the DSA-bwd drains, not the
   replay eventSyncs — a readout against pre-C′ profile bounds would
   misattribute C′'s eventSync removal to W3.
2. **Where freed time should surface:** compression of the drain-covered
   bwd windows via kicked-recompute kernels overlapping INTO those windows
   (raising kernel occupancy/concurrency during drains), plus the standing
   mechanism bar (bwd comm overlap ≥50 %, SendRecv in bwd windows). The
   drains are 97.4 % GPU-kernel-covered — they are NOT GPU-idle slack, so
   do not book them as reclaimable idle; the win shows as overlap, not as
   drain removal.
3. **Wait accounting by cause, not total:** trace readout must bucket
   host wait by cause (drains vs eventSyncs vs exposed SendRecv) — totals
   are conserved across C′/W3 and prove nothing alone.

## 2026-08-09 — T2 gate HARD FAIL (v3, fixc-included run) — investigation ledger

Gate run on the manifest-verified v3 tree: **HARD FAIL**. In-process arming
VALIDATED (fixc=1: hits 1→3, misses 0→0, both stats asserts PASS). Two
defect threads, both now REAL code suspects (harness thread closed):

- **Defect 1 — ROOT-CAUSED (harness, second artifact):** `run_once` drew
  `grad_out = torch.randn_like(hidden_states)` from the GLOBAL RNG per
  call, so the A/B/C runs being compared backwarded against DIFFERENT
  upstream gradients. Outputs are grad_out-independent (PASS everywhere);
  input grads are not (FAIL everywhere) — the exact 0626 signature. The
  grad assertion was VACUOUS from authoring: v1 crashed before evaluating
  it, so 0626 was its first completion — **an assertion that never RAN is
  indistinguishable from one that passed** (kepler's formulation; a second
  instance of the silent-inertness class, after the v1 inert gate).
  Consequence: v3's backward numerics are UNTESTED, not condemned and not
  cleared — the R1 question (zero-padded grouped-GEMM dgrad schedule) stays
  OPEN until the fixed harness runs on-box. Code read of the W2 backward
  (both chunked Functions, the per-group sorts, the combine passthrough)
  found it exact-by-construction everywhere except R1.
- **Defect 2 — ROOT-CAUSED + PROVEN (harness):** `craft_routing`'s
  engineered cases collapse below TOPK selections/token — imbalance's
  `[0]*7+[rand]` list collapses in the bool mask to 1–2/token (runtime
  proof: per-token min/max 1/2, 121 pairs vs static 512 at toy scale);
  zero_expert/zero_peer_group post-mask selected experts (477 vs 512). The
  dispatcher's dropless `num_out_tokens` is static T·topk
  (token_dispatcher.py:1623) and the permute pads to it
  (moe_utils.py:420), so the monolithic A2A's actual-count splits mismatch
  the padded buffer → the exact all-ranks crash. State-independent: v3
  EXONERATED on disarm/fallback hygiene (ship-blocker alarm withdrawn by
  kepler). The gate had NEVER run cases 2–4: v1 died at balanced backward,
  so the crash was latent since authoring.
- **Gate fix landed Mac-side (both defects, one revision):** (a)
  `craft_routing` builds every case from candidate sets with exactly TOPK
  distinct picks/token — validated at gate constants (256 experts, topk 8,
  EP16): per-token 8/8, total == static 65536, imbalance expert-0 skew
  3.48x ≈ the documented 3.5x, zero paths preserved; (b) one seeded
  `grad_out` per (case, ckpt, fixc) iteration shared by A/B/C; (c)
  try/finally around the fixc env set/pop (boltzmann's hardening, folded
  per kepler). py_compile clean.
- **Canonical md5 chain (RESOLVED, kepler 2026-08-09):** canonical gate =
  **40b56e98** (368 lines, on disk, contains the finally-guarded pop).
  boltzmann's 831a6f93 never landed on the canonical copy and is VOID
  (kepler's crossed ack of it voided same-day; his "8d58b557" reference
  matches no recorded state — transcription slip). boltzmann APPROVED the
  craft_routing fix with independent CPU verification at real constants;
  his explicit review of the grad_out fix is the remaining checkbox
  (requested). Box re-run carries 40b56e98, post-canary.
- **Chain extension (same day, Jack's ruling):** 40b56e98 → **d88d8b7d**
  (402 lines) adds `_delta_stats` mismatch diagnostics on the four
  comparison sites (mismatch count, max/mean abs, max rel vs bf16 eps,
  signed bias) — print-only, rank-0-gated, no pass/fail logic touched.
  Motivation: under the variance-class ship bar, a grad torch.equal FAIL in
  the re-run must PRODUCE the noise-class-vs-systematic evidence in the
  same box run; without it a mismatch would cost a second box run to
  quantify. Review: boltzmann (chain discipline); kepler to re-designate
  canonical for the re-run.
- **Sequencing (kepler-approved):** T2 re-run on gate 40b56e98 with v3
  re-applied = next box item immediately after the W3 canary readout. If
  R1 fires there → ship-constraint question for Jack with the per-chunk
  8-gemm shell fallback costed; if not → W2 back on the ladder with its
  first genuinely-passed gate.
- **T2 RE-RUN RESULT (fourier, branch clone, gate d88d8b7d + v3): ALL
  PASS.** Every input-grad check, all four routing cases (incl. the
  previously-crashing imbalance), ckpt on/off, fixc variants (in-process
  arming green: hits 1→3, misses 0→0). Both 0626 harness root causes
  empirically sealed; **R1 resolves NEGATIVE at the re-run's fidelity** —
  zero-padded grouped-GEMM dgrad is bitwise here; no ship-constraint
  question, no 8-gemm-shell fallback on this evidence. FIDELITY CAVEAT:
  fidelity pin requested from fourier (BT_TEST_HIDDEN 2048 default vs
  6144 full-fidelity); the bitwise claim scopes to what was run. If 2048,
  one 6144 confirmation run in a future idle window is belt-and-suspenders
  (non-blocking under Jack's variance-class bar). W2 v3 is gate-passed for
  the first time and re-enters the ladder behind the W3 canary: T3 canary
  slot, variance-class ship bar, 20-step loss canary as the drift cover.
  Log: box `lps1062_bench/wxlgv5w/t2_w2_gate_rerun_d88d8b7d.log` (Mac
  mirror requested).
- **Harness thread — CLOSED (fourier, corrected):** box gate .py md5
  e86589de (333 lines) vs Mac f475e2e7 (335); diff = exactly the
  `BT_T2_SKIP_FIXC` conditional (box: older 1-line unconditional
  fixc_cases). With the SKIP export removed the box revision ran fixc
  variants unconditionally = intended inclusion; result unconfounded. The
  box copy DOES pop the fixc env (box line 320) — fourier's earlier
  "never pops" was a sed-window artifact; the cross-case env-leak channel
  does NOT exist. Sweep-scope gap recorded for kepler: estate sweep covered
  patch files only, never test/tooling scripts (kepler extending).
- **Discipline rule (kepler, binding):** defect-1 treated as
  UNCONFIRMED-REAL until Mac gloo repro reproduces it or a verified-fresh
  box harness re-run does. Prove-don't-assume before sinking into v3's
  backward.
- **Evidence artifacts:** box `lps1062_bench/t2_w2_gate_fixc_included_FAIL_0626.log`
  (md5 7ba61009) + Mac `~/perf_profiles/lps-1062/round3/` same;
  `token_dispatcher.py.pre-v3.orig` (pre-v3 tree state) and the 04:25 v1
  FAIL log `t2_w2_gate_FAIL_318g61w.log` for signature diffing.
- **Box state:** v3 REVERSED off the tree pre-canary (kepler's order;
  Phase-1 zero-marker criteria as the reversal check). Box repro of this
  FAIL needs v3 re-applied — sequenced after the W3 canary readout.

## 2026-08-09 — W2 DESIGN CLOSURE (gate tier)

- **W2 v3: gate-passed, exonerated in full** (T2 re-run, gate d88d8b7d,
  ALL PASS — outputs AND input grads bitwise across all cases/variants;
  fixc composition green). R1 did not materialize at gate scale: the
  zero-padded grouped-GEMM dgrad is bitwise in the tested configuration.
- **RETIRED (not-needed):** the R1 contingency — per-chunk 8-gemm module
  shells (`_make_fused_ops` pattern). Recorded as the documented fallback
  that was never needed; revivable only if a future fidelity/shape shows
  systematic dgrad deviation under the variance-class quantification.
- **STILL PARKED, resolving at the W3 readout:** W2-v2 (bwd seq-bump) and
  option 6 (bwd-chain reorder). Their ratified un-park/park-permanently
  rule in `W2V2_DECISION_stub.md` keys on the W3 canary's bwd-overlap bar
  (≥50 %) — pre-registration means the rule runs its course; the permanent
  park formalizes when the canary reads out. (kepler's direction noted;
  early closure would contradict the stub's own decision rule.)
- **Fidelity PINNED (fourier):** the ALL-PASS re-run ran at DEFAULTS —
  BT_TEST_HIDDEN 2048, BT_TEST_SEQ_LEN 8192 (neither set in his launcher).
  The gold-tier bitwise claim scopes to 2048/8192 gate scale. The 6144
  full-fidelity confirmation is queued on box 2's idle window (after F2
  boots), setting logged in the arm notes; belt-and-suspenders under
  Jack's bar, non-blocking.
- W2 rejoins the ladder as a TIMED ARM pending boltzmann's line-accounting
  verdict; T3 canary under the variance-class ship bar with the 20-step
  loss canary as the drift cover.

## 2026-08-09 — W3 canary: chunk-count + memory-model reconciliation

**Chunk structure (HIGH confidence, code + HF config):** GLM-5.2 =
`num_hidden_layers` 78 (`first_k_dense_replace` 3 → 3 dense + 75 MoE) +
`num_nextn_predict_layers` **1** MTP = 79 layer-modules. Recompute is
uniform, `recompute_num_layers=1` (megatron_config.py default) → **1 chunk
per layer-module = 79 chunks/mb**; MTP flows through the same chunk_runner
(W3 notes). There is NO 150-chunk / 2-chunks-per-MoE-layer structure.

**The "150" was a telemetry misread (fourier's data fine, interpretation
off):** `_lookahead_note` counts EVENTS (~2 per chunk-backward: one
hit/miss + one kick) and the window log prints CUMULATIVE stats at each
300-event boundary — mislabeled "(%d chunk backwards)". Window 1 = 300
events ≈ 150 chunk-backwards ≈ 1.9 mb at 79 chunks/mb → cumulative kicks ≈
149. The "149 = chunks−1 ⇒ 150 chunks/mb" pattern match was coincidental
(149 ≈ 150×78/79). boltzmann's "150 = 2×75" correction (built on the
misread) is RETRACTED; his earlier "308/step = 77×4" bar was also off.
**Corrected bars: kicks == hits == 78/mb (312/step at d4), misses == 1/mb
(4/step; the last chunk recomputes inline), sweeps == 0 (halt), fallbacks
== 0.** Telemetry relabel (events vs chunk-backwards) filed for the next
patch revision — NOT mid-canary.

**Memory model (the +25.4 GiB vs +2–4 GiB miss) — SETTLED by the
allocator snapshot (fourier's capture, 16 ranks,
`round3/arm-w3-318g61w/mem_snapshot/`):**

- **Per-chunk kicked graph ≈ 11.4 GiB** (two live at the mid-backward
  peak = 22.8–23 GiB on the side stream — the design's depth-2 holds).
  Composition (rank-0 live-at-peak blocks, all side-stream,
  lookahead-framed): MoE-pipeline saves dominate — 2.58 GiB × 4
  (`all_to_all_deferred` buffer, two `sort_chunks` outputs, grouped-GEMM
  save) + 1.72 GiB GEMM + 0.86 GiB × 3 expert GLU + 0.77 GiB permuted
  buffer + attention ~0.5–0.75 GiB (`flash_mla_sparse_fwd` 512 MiB × 2,
  `absorbed_mla` 576 MiB × 2, dsa 256 MiB × 2). **CORRECTION to my earlier
  reconciliation: the full-seq attention K/V term is REFUTED — zero
  4.0 GiB blocks exist; the DSA sparse indexer never materializes expanded
  full-sequence K/V.** The magnitude model (~11.5 GiB/chunk) was right;
  the composition is MoE-pipeline-dominated, not attention-dominated. The
  memo's 2–4 GiB estimate undercounted the A2A/sort/GEMM save set ~4×.
- **Allocator retention, directly measured:** the W3 side-stream's segment
  pool holds **26–30 GiB free-cached** at the dump instant (rank0 30.35,
  r1 26.9, r8 28.4, r15 25.7) — the high-water cache for the per-chunk
  kick allocations (peak demand ~23 GiB + within-pool slack), uniform
  across ranks. This pool exists only under W3 and fully explains the
  poller's +25.4 GiB reserved delta (net of a small compute-pool relief
  from recomputes moving off the compute stream; compute pool's own
  free-cached ~29.5 GiB/rank predates W3).
- **Decomposition verdict:** of the +25.4 GiB, **~11.5 GiB is intrinsic
  live cost** (one extra chunk graph vs status quo) and **~14 GiB is
  allocator retention on the side-stream pool — recoverable in principle**
  (empty_cache between steps / pool cap / trim, at allocator-churn cost).
  16k-enablement consequence (kepler's workstream): the intrinsic part
  scales with the per-chunk graph at that shape; the retention part is a
  knob, not a wall. The snapshot piggyback saved the workstream a boot.

**Ship implications (kepler's ruling applied):** (a) 131k arm STANDS —
+25.4 GiB plateau vs 44.9 GiB headroom clears the ≥10 GiB bar; (b) 16k×d32
stays HARD OFF — corrected model there: MoE/MLP terms ×4 (32,768
tok/rank/mb) ≈ 13–14 GiB/chunk + fragmentation — genuinely borderline,
re-measure before any enablement; (c) scorecard W3 memory line = **+25.4
GiB measured at 131k-d4** (replaces the 2–4 estimate). Memo §7b needs the
correction (estimate missed the full-seq attention term). Wall read 647 vs
666 is INFORMATIONAL only — the capture decides mechanism overlap; no
adjudication from wall (kepler).

## 2026-08-09 — W3 canary (v2) — watch log

- Boot on box 1: HEALTHY — gate ACTIVE all 16 ranks, step=0 (fourier).
  The armed line is the first watch-list item confirmed. Leg running.
- Readout watch (boltzmann metrics, my binding amendments):
  kicks==hits==308/step, misses==4/step (78 chunks = 75 MoE + 3 MTP,
  model-pinned), sweeps==0 (halt rule), fallbacks==0; memory +2–4 GiB
  @131k; mechanism-first: SendRecv-in-bwd-window DELTA vs the C′-on
  baseline (56.3/55.4 % — the retired ≥50 % absolute bar was sub-floor) +
  the lookahead side stream's appearance; drain concurrency rising from
  1.017/1.008 baseline; eventsync-fwd must stay ~0 under C′; loss vs the
  cprime-timed reference, house band, warmup-datums 2 matched (INVALID ≠
  FAIL on mismatch); capture ratio vs the 60–80 % model.

## 2026-08-09 — JACK'S RULING: the W2 ship bar is VARIANCE-CLASS, not bitwise

Jack's actual constraint (verbatim intent): **"optimized-vs-default
difference ≈ run-to-run variance of default"** — no impact on correctness
beyond normal run variance. Consequences (kepler, 2026-08-09):

1. The numerics adjudication + house band are RATIFIED — his standard IS
   the band (≤2e-3 pass / >5e-3 stop, matched warmup-datums).
2. Two-tier verdicts RATIFIED.
3. **If R1 fires in the T2 re-run it is NO LONGER an automatic
   ship-blocker.** Reduction-order-class grad deviation from GEMM
   scheduling is within his standard PROVIDED it is shown noise-class, not
   systematic: quantify the grad deltas (magnitude vs ULP/reduction-order
   expectation; zero-mean vs biased; unstructured vs structured), run the
   canary soak, show in-band.
4. `torch.equal` remains the DIAGNOSTIC GOLD TIER (a bitwise pass is the
   cheapest proof and catches real bugs — tonight it caught harness bugs),
   but the SHIP bar is variance-class.
5. The per-chunk 8-gemm-shell fallback stays relevant ONLY if the deviation
   proves systematic.

W2 decision tree as amended:
- T2 re-run `torch.equal` PASS (outputs AND grads) → gold-tier numerics
  proof; proceed (canary band still governs ship).
- Grad-only `torch.equal` FAIL with bitwise outputs → R1-class candidate:
  the gate now prints delta diagnostics on the mismatch path (mismatch
  count, max/mean abs, max rel vs bf16 eps, signed bias); noise-class
  (ULP-scale, zero-mean, unstructured) → within Jack's standard, proceed to
  the canary soak; systematic → stop, 8-gemm-shell fallback costed for
  Jack.
- Output `torch.equal` FAIL → real-bug stop-line (forward deviation is
  outside the reduction-order-in-backward envelope R1 covers).

## 2026-08-09 — W3 v1 BOOT FAILURE + v2 (call-site defect, mine to own)

- **Defect:** v1's recompute.py `chunk_runner` call site passed
  `chunk_key=`/`carrier=` keywords BEFORE `*args`; the 6 trailing
  positionals fill signature slots 3–4 and collide → TypeError all 16
  ranks, 0.1 s into forward (box log `w3_boot_FAIL_chunk_key.log` md5
  dd58a4df). v1 never booted. The Mac suite drove `lookahead_checkpoint()`
  directly and never executed the integration call site — **third instance
  tonight of tested-in-isolation / dead-at-integration** (after the two T2
  harness artifacts; same class family as the inert-gate/vacuous-assertion
  lineage).
- **v2 fix (fourier's proposal, confirmed):** all-positional in signature
  order, `self.config.distribute_saved_activations` per the sibling
  te_checkpoint convention. Keywords-after-`*args` is not an alternative
  (params are positional-or-keyword; the fill still collides).
- **Verification:** v2 hunk applied to the FIX-C-base recompute.py blob
  (`d43d8621b`) reproduces the fixed Mac tree byte-for-byte; full patch
  `git apply --check` clean vs that base. **v2 md5
  6f08c5dc08b6720c26102a47a9706b85 (509 lines), supersedes v1.**
- **Regression net:** `test_w3_lookahead_checkpoint.py` sec7a/sec7b —
  integration-arity drive (6 trailing positionals bind correctly, chunk
  registers under the positional key) + AST guard on the vendored call
  site (exactly one call site, NO keywords at all — boltzmann's
  hardening: any keyword before *args collides identically — forwards
  *args). Suite green 28/28. boltzmann's v2 verdict: GO.
- **Gate chain addendum (boltzmann, verified link-by-link):**
  8d58b557 (my fixture fix) → 831a6f93 (boltzmann try/finally) → 40b56e98
  → 37473eaf (boltzmann completes run_once arity — 40b56e98 had updated
  the call sites but left run_once at 7 params with the stale draw, a
  would-be TypeError on first invocation; py_compile does not catch arity,
  his AST pass did) → d88d8b7d (delta diagnostics). On-disk d88d8b7d
  re-verified by me: 8 params, 3/3 call sites, stale draw gone, compiles.
  boltzmann APPROVED the grad_out design and _delta_stats; confirmed
  78 = 75+3 MTP with the router-bearing/dispatcher-cache-free
  discrimination (dense-first-3 refuted) — W3 log bars model-pinned.

## 2026-08-09 — rulings carried (from kepler ratifications)

- T2 gate runs WITH fixc variants (no `BT_T2_SKIP_FIXC`); fixc branches
  self-arm `BT_MOE_DISPATCH_REPLAY_CACHE` in-process (lazy per-call read,
  token_dispatcher.py:123); the one-time gate-state WARNING latches DISABLED
  on the first (non-fixc) call — **stats asserts are the sole admissible
  evidence** on fixc variants (the armed-line discriminator retracted).
- Loss verdicts (boltzmann): in-process gates hard-bitwise; cross-boot
  canaries house band ≤2e-3 pass / >5e-3 stop; matched `--warmup-datums`
  mandatory — mismatch = INVALID comparison (re-run), not FAIL; warmup0 +
  warmup-datums recorded in every arm JSON.
- W3 canary binding: sweeps==0 halt rule; 16k×d32 gated OFF until measured;
  FIX C paired; mechanism-first acceptance per two-tier verdicts.
