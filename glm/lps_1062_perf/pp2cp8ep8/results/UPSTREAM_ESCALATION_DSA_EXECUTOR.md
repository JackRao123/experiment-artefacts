# UPSTREAM ESCALATION — combined-1F1B executor is nondeterministic on GLM-5.2 DSA/THD

> **Status 2026-08-14 (addendum §7):** the §4 exposure hypothesis
> ("conventional schedule hides the race, the executor exposes it") is
> **falsified** by a same-night control, and §3's control wording is
> corrected. The nondeterminism is schedule-independent. Read §7 before
> forwarding.

Drafted by ramanujan, 2026-08-14 (LPS-1062 overlap campaign). For Jack to
forward upstream or hand to the team. Measured numbers throughout; the
reproducer is exact and on a pushed branch. Companion docs:
`EXECUTOR_CONTRACT_SCOPING.md` (the trainer-side contract that makes the
executor reachable — our work, landed and correct), `UPSTREAM_PROPOSAL.md`
(the per-layer recompute dial, whose executor-dependent rungs are HELD behind
this verdict), and `KERNEL_DEFECT_PRIOR_ART.md` (the prior DSA race saga).

## 1. The ask

A **DSA × combined-1F1B numeric-equivalence investigation** upstream. The
`overlap_moe_expert_parallel_comm` combined-1F1B executor, driven through the
schedule-plan protocol, trains to aggregate precision but is **nondeterministic
at the per-token level on GLM-5.2 (DSA sparse attention + THD packing)** —
the same input, same weights, same boot, back-to-back, reproduces nothing at
the per-token level. The overlap flag's premise is "hide the all-to-all, same
math"; on this stack the math is not reproducible. We need upstream to either
confirm a known DSA-kernel race under interleaved scheduling or scope the
executor's DSA support.

## 2. The evidence (determinism probe, pre-registered)

Setup: flag-ON, one fresh boot, the parity driver run 3× with an identical
6-datum input set (109,827 tokens), `forward_backward`, no optimizer step —
so the model, the data, and the weights are fixed and only the executor's
run-to-run behavior varies.

| compare | logprobs max_abs_diff | bar | aggregate loss |
|---|---:|---:|---|
| within-boot run1 vs run2 | **5.098** | 1e-3 | rel 2.3e-5 |
| within-boot run2 vs run3 | **6.343** | 1e-3 | rel 3.8e-5 |
| boot-to-boot (two fresh boots) | **3.424** | 1e-3 | — |
| (reference) flag-ON vs flag-OFF | 3.758 | 1e-3 | rel 1.26e-4 |

Within a single boot, the executor reproduces nothing per-token (3–6
**thousand**× the kernel-noise bar), while the aggregate loss stays tight
(rel ~3e-5). This is run-to-run nondeterminism, not a systematic
decomposition difference (which a boot-pair alone could not have excluded).

**Characterization** (per-token divergence structure, Mac-side analysis,
`tools/parity_divergence_structure.py`): a **uniform ~0.08/token run-to-run
noise floor** (80× the kernel-noise bar — the executor's timing-dependent
reduction order is measurably noisier than the conventional schedule even in
the bulk) **plus scattered large spikes (up to 5.1) at recurring positions** —
the worst recurs at datum 3 position 25055 (92% through the 27,111-token
datum) on *both* within-boot compares, deep in the DSA top-k boundary region.
Read: the executor's interleaved comp/comm scheduling introduces
timing-dependent numeric noise, and the DSA indexer's discontinuous
top-2048 KV selection amplifies borderline flips into large per-token
divergences at knife-edge positions.

## 3. Reproducer recipe (exact)

- **Model:** `zai-org/GLM-5.2-FP8` (DSA, 78 layers, 256 experts, top-8),
  LoRA rank 32 (frozen base; the frozen-embedding grad-root fix is in the
  tree — see §5).
- **Branch:** `jackrao/lps-1062-overlap-contract-shim` @ `1cd31535`
  (trainers); submodule chain megatron-bridge @ `146f2636` → Megatron-LM
  @ `b37c01f2e`. The trainer-side schedule-plan protocol (the shim) is this
  branch's work and is **exonerated** by the structure analysis (the
  divergence is in the executor's forward decomposition, not the contract).
- **Config:** `trainer_pp2cp8ep8_32k_selective_vpp2_overlap.json` —
  PP2 / VPP2 / CP8 / EP8 / TP1 / ETP1, `max_seq_len=32768`, selective
  recompute, `comm_overlap.overlap_moe_expert_parallel_comm=true`,
  alltoall dispatcher.
- **Env:** fixed wheel (`nvidia-cudnn-frontend 1.27.0` — see §4),
  `BT_SKIP_WARMUP=1`, `BT_SAVE_STATE_SYNC=1`.
- **Probe:** `tools/parity_driver.py` 3× (default `/forward_backward`, NOT
  `--forward-only` — the executor is gated `and not forward_only`), no
  optim_step; compare run-pairs.
- **Control that passes:** the same boot with the flag OFF (conventional
  schedule) reproduces the reference per-token logprobs within the 1e-3 bar,
  and the flag-ON arm's *aggregate training loss* matches the control
  (12.297 vs 12.299) and trains correctly — the executor's forward is right
  to aggregate precision; it is the per-token reproducibility that fails.

## 4. The DSA-kernel-race prior (why this is the first suspect)

This campaign already traced one nondeterministic corruption to a
**scheduling-dependent DSA kernel race** (`KERNEL_DEFECT_PRIOR_ART.md`): the
stale cudnn-frontend 1.26.0+dsatopk1 wheel produced a nondeterministic
uninit-read corruption under THD varlen + CP attention; the pinned 1.27.0
wheel carries the LPS-1003 race-fix series — the DSA wrapper stream-race root
fix (#354) and the SM100 DSA backward race/sync fixes (#395/#396/#426/#429/#439,
including a TMEM WAR race at *exactly GLM-5.2's head_dim 576/512* → silent
dkv corruption). The combined-1F1B executor's interleaved per-layer scheduling
is precisely the scheduling class that saga lived in — and it is a scheduling
pattern the fixed wheel's race series was not exercised against (the
conventional schedule does not interleave layer sub-modules across streams the
way the executor does). Kernel-source snapshots from that investigation are on
disk (`pp2cp8ep8/kernel_src_snapshot/`, `cutlass_dsl_src_snapshot/`,
`kernel_src_snapshot.tar.gz`, `cutlass_dsl_src_snapshot.tar.gz`).

**Hypothesis for upstream:** a DSA kernel (indexer top-k, or the cuDNN sparse
attention forward/backward) has a residual race that the conventional
schedule's serialization hides but the executor's interleaved stream
scheduling exposes — same class as the fixed LPS-1003 series, a manifestation
it doesn't cover.

## 5. What is NOT the cause (ruled out, so upstream doesn't re-litigate)

- **The trainer-side shim / schedule-plan contract** — the executor is reached
  correctly and trains to aggregate precision; the batch threading
  (packed_seq_params, masks, fp32 boundary) is verified value-exact. The
  divergence is in the executor's forward decomposition of the DSA attention.
- **The PreProcessNode grad-root landmine** (frozen-embedding
  `requires_grad=False` → RuntimeError at the first stage-0 backward) — found
  and fixed on this branch (mcore `b37c01f2e`), hardware-validated; orthogonal
  to the nondeterminism (it gates boot, not per-token values).
- **A reporting/ordering offset** in the logprob stitch — ruled out by the
  sorted-multiset test (the value *distributions* differ; not a permutation).

## 6. Impact / why it matters

The overlap flag is the designed enabler of the 25–28 s/step schedule-hideable
all-to-all prize at long context (measured exposed-a2a mass on this stack).
Without per-token numeric reproducibility, the executor cannot be A/B'd or
shipped for DSA models — and any consumer running DSA + THD + the overlap flag
is silently nondeterministic today. A clean executor would unblock both the
overlap A/B and the per-layer recompute dial (whose rungs are held behind this
verdict).

## 7. ADDENDUM (2026-08-14, hertz — LPS-1062 P4 S2 noise matrix): attribution corrected — the nondeterminism is schedule-independent

Appended after new evidence; §1–§6 are preserved as drafted (honest
chronology). Evidence: the S2 noise matrix, pre-registered decision rule in
`runs/overnight_20260813_overlap_campaign/S2_NOISE_MATRIX_PREREG.md`, full
record in `pp2cp8ep8/NOTEBOOK.md` (2026-08-14 ~17:1x entry). All matrix
artifacts are job-id+timestamp-stamped and sha256-verified. Measurement
conditions: ship stack (trainers `73c24b00` + TF32-head patch, mcore
gate-stack tree), fixed wheel `nvidia-cudnn-frontend 1.27.0`, mission shape
PP2/CP8/EP8 @ 131k, parity driver, 9 datums / 262,032 tokens, no optimizer
step.

### 7.1 The first direct conventional-path control

Before tonight, no within-boot per-token determinism measurement of the
**conventional** PP2/CP8/EP8 path existed anywhere in the campaign (all W2
detprobe arms ran the executor; fixed-wheel conventional figures were
loss-level only). Tonight's matrix measured it, B/F host caches absent,
conventional schedule:

| cell | design | per-token max-abs | % tokens >1e-3 | loss rel |
|---|---|---:|---:|---:|
| N1a | OFF within-boot, run1 vs run2 (same boot) | 3.716 | 95.7% | 7.11e-05 |
| N1b | OFF within-boot, run2 vs run3 (same boot) | 5.455 | 95.7% | 3.18e-05 |
| N3 | OFF cross-boot, two fresh boots | 4.270 | 95.7% | 1.58e-04 |
| N4 | OFF vs ON cross-boot (the B/F parity compare) | 3.495 | 95.7% | 9.11e-05 |

### 7.2 The falsified contrast

§4's first-suspect hypothesis — that "the conventional schedule's
serialization hides" a residual DSA race that "the executor's interleaved
stream scheduling exposes" — is **falsified at the per-token level**. The
conventional path, with no executor and no schedule interleaving, reproduces
run-to-run per-token divergence at max-abs **3.7–5.5 within a single boot**
and **4.27 boot-to-boot**, ~95.7% pervasive — the same magnitude class as the
executor's within-boot 5.098/6.343 (§2). Same boot, same gates, same tree,
conventional schedule: the divergence reproduces anyway.

The reframed mechanism (adopted campaign-side): the executor "race" and the
conventional nondeterminism are **one phenomenon** — the DSA learned
indexer's discontinuous top-2048 KV selection flipping borderline per-token
choices run-to-run, **regardless of schedule**. kolmogorov's nuance, verbatim:
"the executor IS louder in the bulk ~0.08/token uniform floor; the large-spike
pervasive class is shared." The bulk-floor excess remains
executor-attributable; the pervasive large-spike class does not.

Corollary correction to §3: the W2-era "control that passes" was not a
within-boot conventional per-token determinism measurement (none existed at
writing); the first such measurement (N1, above) **fails** the 1e-3 bar at
the mission shape.

### 7.3 The re-sited blocker — the ask, sharpened

What survives for upstream: the executor's within-boot nondeterminism is still
real, still measured, and still blocks the per-token parity gate for the
overlap program — **the ask stands; its attribution sharpens.** The blocker
re-sites from "executor correctness under interleaved scheduling" to the
**DSA kernel-floor class** (the LPS-1003 / wheel-saga lineage, §4): the pinned
1.27.0 wheel carrying the race-fix series does **not** deliver per-token
run-to-run determinism on GLM-5.2 DSA/THD at these shapes, and the residual
is schedule-independent. Same upstream owner; the sharper ask is a
determinism investigation of the DSA indexer top-k / sparse-attention path
(borderline tie-breaking in the learned top-2048 selection under THD varlen +
CP), with the executor's interleaving no longer the suspected exposure
mechanism. The ruled-out list (§5) and the shim exoneration stand unchanged.

### 7.4 Shape caveat (stated honestly)

The executor evidence (§2) is 32k / 6-datum / selective-recompute config;
tonight's conventional floor is 131k / 9-datum / mission config. The
one-phenomenon claim rests on magnitude-class matching across shapes, not a
same-shape control. A same-shape 32k conventional within-boot control would
close the gap — estimated cost one boot plus ~30 min driver-only — and was
**not run** (box hours governed the call). It remains available as a
confirming control.

### 7.5 Campaign rule going forward: noise-relative parity gates

No tight parity gate (1e-6 loss-rel / 1e-3 per-token) is measurable on this
stack at these shapes until the floor is fixed — the floor sits 3–6
**thousand**× above the per-token bar. Every parity gate on this stack must
be **noise-relative**: bars derived from a measured floor (within-boot and
boot-to-boot, at loss AND per-token granularity), not aspirational constants.
This rule already governed tonight's ship-stack adjudication: the B/F
host-cache correctness claim stands in noise-relative wording (off-vs-on
indistinguishable from the path's intrinsic nondeterminism at both
granularities — N4 within the N1/N3 floors on all three pre-registered
statistics; caches bitwise-exact by construction; canary agreement ≤7e-4),
with no 1e-6/1e-3 claim imported anywhere.
