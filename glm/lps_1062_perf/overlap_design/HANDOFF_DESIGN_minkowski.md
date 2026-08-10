# HANDOFF — MoE A2A⇄compute overlap design owner (minkowski → fermi)

**Date:** 2026-08-10 · **Ticket:** LPS-1062 (GLM-5.2-FP8 perf, 2×8 B300,
golden TP1/PP1/EP16/CP16, full recompute, attention-only LoRA r32)
**Successor briefing order:** read this file, then
`ESTATE_NOTES_minkowski.md` (the running log — rulings, md5 chains, root
causes, corrections), then `DESIGN_helmholtz.md` §0–§2 + §6.1 for the
baseline anatomy and the field-calibration record. The standing rule from
my tenure: **inherited state claims are unverified until re-proven** —
tonight produced three instances of tested-in-isolation/dead-at-integration
plus a vacuous assertion. Prove, don't assume.

---

## 1. Lever states (exact)

- **W1** (probs A2A on a second NCCL comm) — **SHIPPED as v1.** Mechanism
  confirmed on-box (probs off-stream 900/900, gap −2.07 ms/pass); wall
  −0.46 s (residual = the bwd-window engine-issue-order exposure — the
  thing option 6 targets). W1-v2 paired-dispatch stays ARCHIVED-DEAD on
  hilbert's negative gate.
- **W2** (intra-layer chunked pipeline, K=2) — **v3 GATE-PASSED, first
  genuine pass** (T2 re-run on fixed harness: outputs AND input grads
  bitwise, all cases/variants, fixc composition green; R1 resolved
  negative at 2048/8192 gate scale). The 0626 "failures" were both harness
  artifacts (see §3). Ahead: 6144 full-fidelity confirmation (box-2 idle
  window, queued, belt-and-suspenders under Jack's bar), then the T3
  canary slot under the variance-class ship bar with the 20-step drift
  cover. 8-gemm-shell fallback retired (not needed).
- **W3** (lookahead recompute) — **MECHANISM-PROVEN, WIN REFUTED as
  implemented on B300 @131k.** +25.8 GiB for 0 wall; kicks 99.9 %
  serialized. Causes: whole-backlog `wait_stream` ordering + **no SM
  slack** (bwd phase ~97 % kernel-covered). DISARMED from all arms; 16k
  HARD OFF. **Any v3 is GATED on boltzmann's SM-slack measurement from the
  existing traces — if no bwd-phase occupancy headroom exists, W3-on-B300
  closes; do not start v3 before that lands.**

## 2. The active disposition: option 6 / W2-v2 UN-PARKED (my call, stub rule 3)

The `W2V2_DECISION_stub.md` decision rule fired its else-branch (W3 capture
≈ 0 % < 50 % bar). **Un-parked, with a resource distinction the stub
predates:** W3 failed for want of SM slack (it adds concurrent compute);
option 6 (bwd-chain reorder for the probs reverse, ~1–1.5 s — the §6.1
engine-issue-order residual, i.e. W1's documented residual) and the v2
seq-bump are **comm-resequencing** plays — no new compute, they hide
existing comm behind existing (saturated) bwd windows. The SM-slack gate
does NOT apply to them. Sequence: **option 6 first** (small, mechanism
understood), **v2 behind it** (stub scope: two seq-bumps + wait-aware
wrappers + event plumbing, ~1–2 days; the v1 bitwise argument carries
over). Caveat: re-confirm the bwd-phase comm-window cover from the W3
canary trace (C′-on) before locking v2's win model.

## 3. Process lessons — the regression nets are built, use them

Three instances of **tested-in-isolation / dead-at-integration** in one
night, plus a **vacuous assertion** (an assertion that never RAN is
indistinguishable from one that passed — kepler's formulation):

1. T2 gate `craft_routing` collapsed engineered cases below topk
   selections/token (bool-mask duplicates) → permute pads to static
   T·topk → A2A split mismatch crash. Fixed: candidate-set construction,
   exactly TOPK distinct/token, validated at gate constants.
2. T2 gate `run_once` drew `grad_out` from the global RNG per call → A/B/C
   backwarded against different upstream grads; the grad assertion was
   vacuous from authoring. Fixed: one seeded `grad_out` per iteration,
   shared; `_delta_stats` diagnostics ride every mismatch (Jack's
   variance-class bar needs the quantification in-run).
3. W3 v1 call site passed keywords before `*args` → TypeError at boot;
   the Mac suite never executed the integration call site. Fixed
   (all-positional, v2 patch md5 `6f08c5dc…`) + `test_w3…` sec7a/sec7b
   regression net (integration-arity drive + AST guard forbidding ALL
   keywords at the call site).

Also filed: the W3 telemetry window counts EVENTS (~2/chunk-backward) but
is labeled "chunk backwards" and logs CUMULATIVE stats — it produced a
false "150 chunks/mb" read (true structure: 78 transformer layers (3 dense
+ 75 MoE) + 1 MTP = 79 chunks/mb). **Corrected canary bars: kicks==hits==
78/mb (312/step at d4), misses==1/mb, sweeps==0 (halt), fallbacks==0.**
Relabel queued for the next patch revision.

## 4. Memory model — SETTLED (snapshot-verified)

W3 stash at 131k-d4 = **+25.4 GiB** (poller), decomposed by the 16-rank
allocator snapshot (`round3/arm-w3-318g61w/mem_snapshot/`): **~11.5 GiB
intrinsic** (two ~11.4 GiB kicked chunk graphs live at the mid-backward
peak; MoE-pipeline-dominated — 2.58 GiB×4 A2A/sort/GEMM saves; the
full-seq attention-K/V hypothesis is REFUTED, no 4 GiB blocks exist) +
**~14 GiB allocator retention** on the side-stream pool (26–30 GiB
free-cached, uniform across ranks) — the retention is a recoverable knob
(empty_cache/pool cap), the intrinsic scales with shape. This is the
16k-enablement workstream's opening evidence. Memo §7b's 2–4 GiB estimate
undercounted the MoE save set ~4× — correction still QUEUED (follow-up).

## 5. Standing rulings (carry them)

- **Jack's ship bar (2026-08-09):** "optimized-vs-default difference ≈
  run-to-run variance of default" — variance-class, NOT bitwise.
  `torch.equal` stays the diagnostic gold tier; ship verdicts go through
  the house band (≤2e-3 pass / >5e-3 stop, matched `--warmup-datums`
  MANDATORY — mismatch = INVALID comparison, re-run, not FAIL; warmup0 +
  warmup-datums in every arm JSON). In-process gates stay hard-bitwise.
- **Two-tier verdicts:** mechanism-first (trace/telemetry/canary), wall
  second — wall conversions on this box have surprised twice; never quote
  uncalibrated wall-seconds.
- **Baseline discipline:** all deltas vs the C′-on profile (ARM-5), never
  pre-C′ anchors; wait accounting by CAUSE (drains vs eventSyncs vs
  exposed SendRecv), never totals.
- **md5 chain discipline** for every artifact that crosses Mac↔box, incl.
  test/tooling scripts (the estate sweep now covers them). Canonical gate:
  `d88d8b7d…`. Canonical W3 patch: `6f08c5dc…` (v2). W2 v3 patch:
  `c94f72e6…`.
- The 9 design invariants from `HANDOFF_DESIGN_helmholtz.md` §4 stand
  verbatim (combine offsets SEND-side; TE fused sort PERMUTATION-ONLY;
  state on carriers never pass-through tensors; push-order rule; gates
  prove they fired; bitwise parity standard; recompute determinism; one
  host sync per layer-pass; FIX-C pass-marker keying).

## 6. Open items for fermi (in order)

1. **boltzmann's SM-slack measurement** (existing traces) — gates W3-v3
   vs W3-on-B300 closure. Watch for it; do not pre-start v3.
2. **Option 6** — un-parked per §2; scope and ladder slot with helmholtz
   (new orchestrator).
3. **W2 T3 canary** — v3 gate-passed; canary slot under the variance-class
   bar, 20-step drift cover, house band.
4. **6144 full-fidelity gate confirmation** (box-2 idle window, queued) —
   scopes the R1-negative claim to full fidelity.
5. **Follow-ups:** memo §7b memory correction; W3 telemetry relabel;
   W2-v2 build only after option 6 reads.

## 7. Contact points

- Orchestrator: **helmholtz** (Claude Fable; succession kepler → helmholtz).
- Box runner: fourier. Verdicts/measurement: boltzmann. Second reviewer:
  (per helmholtz's re-org). My predecessor's handoff
  (`HANDOFF_DESIGN_helmholtz.md`) and the memo remain the design record;
  `ESTATE_NOTES_minkowski.md` is the rulings/root-cause/correction log.
