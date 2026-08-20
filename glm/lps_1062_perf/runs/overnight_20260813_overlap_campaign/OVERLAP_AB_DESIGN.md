# OVERLAP A/B DESIGN — W2 shim canary + the 131k overlap A/B (lebesgue, 2026-08-13)

Pre-registration for the overlap program's two remaining box legs. Consumes:
`EXECUTOR_CONTRACT_SCOPING.md` (the shim, landed as
`jackrao/lps-1062-overlap-contract-shim` @ e13de4d7, submodule chain carrying
the mcore grad-root fix b37c01f2e — see §2),
`A2A_EXPOSURE_DECOMPOSITION.md` (the prize), BOX_SCHEDULE.md (W2 slot, W5+
follow-on), and jacobi's MEMORY_LEG_DECISION.md (pending — the 131k leg is
CONDITIONAL on it; see §6). doppler drives all trainer lifecycle; this doc
hands doppler exact configs and bars. (Post-rename roster, bayes 2026-08-14:
W2 = ramanujan specs + adjudicates, **lovelace** (ex-doppler) drives;
grothendieck is the W1b window driver + standby operator; the memory leg is
pauli (ex-jacobi); jacobi is now ex-kepler.)

## 1. What's being tested, one sentence each

- **W2 (shim canary):** the contract shim makes the combined-1F1B executor
  reachable from our trainer *with unchanged numerics* — flag-ON trains the
  same loss curve as flag-OFF at memory-safe scale.
- **W5+ (131k overlap A/B):** with the memory path armed (jacobi's leg), the
  flag captures a measurable fraction of the 25–28 s/step schedule-hideable
  a2a ceiling at the mission topology.

## 2. Preconditions (all standing)

- Fixed wheel (cudnn-frontend 1.27.0 venv bump) — W0 gate, per BOX_SCHEDULE.
- **Tree (branch spec, fermi-ordered 2026-08-13):** the shim branch
  `jackrao/lps-1062-overlap-contract-shim` (trainers @ **a3da1223** — the W2
  boot tip), whose submodule chain carries the overlap+LoRA landmine fix as
  its own mcore commit: bridge @ 146f2636 → mcore @ b37c01f2e
  (`jackrao/lps-1062-overlap-contract-shim` on both submodule forks). The
  mcore commit is ONLY the PreProcessNode grad-root hunk from jacobi's
  b907b6153 — the rest of that commit is dial-specific and does not apply at
  the campaign pin 57efae08b (the dial field enters with e5f8643a0, which W2
  deliberately does not carry; W2 stays single-purpose). a3da1223 adds the
  DSA-legal `(12,2,2)` layout entry (estate for the held 1-node smoke — not
  exercised by the W2 mission-config boots). Parity-class discipline: no
  probe hunks; the tree is the branch, exact.
- `BT_SKIP_WARMUP=1` exported on every flag-ON boot (belt-and-braces: VPP>1
  auto-skips the M=1 warmup since 6fa3bfa7; the env var is the manual hatch
  and costs nothing). **Operational note (lovelace, W2 leg 1):** with
  `BT_SKIP_WARMUP=1` the trainer goes armed-idle WITHOUT the READY banner
  (GPUs pin 100% on posted NCCL recv) — the bench status probe is the
  readiness signal, not the banner. Applies to both arms (the control skips
  too, keeping warmup treatment symmetric for the ON-vs-OFF read).
- `BT_SAVE_STATE_SYNC=1` everywhere; `wait_trainer_health.sh` only.

## 3. R3 disposition (fermi's gate — answered, recorded)

**The mission bench config does not run R3.** (a) No campaign config carries
`router_replay_mode` → `NONE` default (control.py:491). (b) The workload is
SFT `cross_entropy`; router replay is armed only in the RL loss family. (c)
The runner hard-fails any datum carrying `routed_experts` under mode≠R3
(training_runner.py:366-371); every campaign boot ran clean → no routes in
the data. **Consequence: no R3-off parity caveat attaches to any arm below.**
The shim's R3×flag rejection (config-parse + step-time) bites only
hypothetical future RL-on-overlap runs.

## 4. VPP-arm choice at PP2 (scoping §5 risk-5 discharge)

The combined executor exists for no-pipelining (PP1) and interleaved (VPP)
schedules only; the shim's config guard makes flag+PP2 ⇒ VPP mandatory.

**Chosen: VPP2** — the only pre-coded GLM-5.2 PP2 VPP layout
(`_GLM52_DSA_PIPELINE_LAYOUTS[(78, 2, 2)]` = [18+emb, 20, 20, 20+loss];
chunks pp0:v0=1-18, pp1:v0=19-38, pp0:v1=39-58, pp1:v1=59-78; chunk starts
19/39/59 all satisfy the DSA `(start-3) % 4 == 0` alignment). VPP4 was
rejected: 78 layers don't quarter evenly (78 = 4×19.5), a new layout would
need DSA-aligned uneven chunks, and deeper VPP deepens the pipeline bubble
at the campaign's small M. Legality: `microbatch_group_size_per_vp_stage`
defaults to PP=2 (model_parallel_config.py:473-474); the interleaved
schedule needs it ∈ [PP, M] — at the canary floor d2, M=2 → 2 ∈ [2,2] ✓.
(M=1 is illegal — the warmup auto-skip/hatch.)

## 5. W2 — shim canary (BOX_SCHEDULE W2, ~2h, gate: canary + parity PASS)

**Memory framing (load-bearing):** VPP2 holds 2 chunks in flight per rank;
at 131k that wall OOMs (L0b-mem aborted >255 GiB at d2; E1 measured 258.8
GiB at ONE in-flight mb under selective). W2 therefore runs at 32k, where
VPP2 fits (the 32k boots ran 183–203 GiB). 131k flag boots are W5+ only,
behind the memory gate (§6).

Flag-ON config (written, validates against the shim guards):
`pp2cp8ep8/configs/trainer_pp2cp8ep8_32k_selective_vpp2_overlap.json` — the
L0 131k config
(`trainer_pp2cp8ep8_131k_L0_selective_vpp2_overlap.json`) at
`max_seq_len=32768`. Control arm exists:
`trainer_pp2cp8ep8_32k_selective_novpp_plain.json` (flag-OFF, plain PP2).

Note on the control's topology: flag-OFF *VPP2* is unrunnable tonight — the
conventional VPP2 path hits the filed loss-path bug (chunked_lm_head.py:196,
poincare-domain). The control is therefore plain PP2 (mission topology, no
VPP). The delta flag-ON(VPP2) vs flag-OFF(PP2) packages executor + VPP
chunking together; both are schedule-only — per-token math is unchanged, and
the parity runbook already established PP-topology invariance at the ≤1e-6
loss bar. For GLM-5.2 the shim's fp32 boundary upcast is numerically exact
(the chunked head casts `hidden.float()` regardless).

Additional packaging note (W2 leg-1, the exp03 validator wall): the flag-ON
arm also carries `moe_shared_expert_overlap=False` — mcore's bridge validator
(comm_overlap.py:500) forbids it under the overlap flag, and the GLM provider
defaults it on (glm5_bridge.py:111), so the shim branch (d34f76d9) clears it
whenever the flag is on, independent of dispatcher. The control (flag-OFF)
runs with the GLM default (shared-expert overlap ON). Scheduling-only, no
per-token math — parity numerics are clean — but the ON-vs-OFF *perf* delta
packages (executor + VPP chunking + shared_expert_overlap-off); carry that
into the W5+ A/B accounting (§6), where shared_expert_overlap-off is part of
the ON arm's honest cost.

### Ladder (in order; each gates the next)

1. **d2 canary, flag-ON, 32k.** Bar: loss in the same-scale flag-OFF band
   (32k d2 measured 12.295–12.304 on the selective twin; re-anchor a fresh
   flag-OFF d2 on-box first — the canary is ON-vs-OFF at one scale, not a
   borrowed band), gn comparable to the control's, loss trains across steps.
   STOP on: any contract/executor error, loss out of band, gn drift.
   **Boot-checklist flag (pre-registered):** this first flag-ON boot is ALSO
   the mcore landmine fix's first hardware validation (b37c01f2e — the
   PreProcessNode grad-root fix; under LoRA's frozen embedding the executor's
   closing pre_process.backward would otherwise raise RuntimeError at the
   first stage-0 backward). If the boot dies with that RuntimeError, read it
   as FIX-INSUFFICIENT evidence, not shim failure — STOP → bayes + pauli
   (ex-jacobi, the fix's author).
2. **Small-seqlen parity leg, flag-ON PP2/VPP2 vs flag-OFF PP2.** The parity
   driver (`tools/parity_driver.py`, 9 mixed datums 8k–64k, 262,032 tokens,
   no optim_step — weights never move). Bars (runbook): loss rel diff
   ≤ 1e-6; per-token logprobs max abs diff ≤ 1e-3; datum count/lengths
   exact. Watch (pre-registered): if loss exceeds 1e-6 but logprobs hold
   ≤ 1e-3, suspect the fp32 boundary — Option A (CE inside the plan) is the
   escape, per the scoping doc. Memory pre-check: the 64k tail under VPP2 —
   if the driver OOMs at 64k, cap the datum set at 32k and record the
   deviation.
3. **Memory ramp, flag-ON, VPP2:** d2 at 32k → 64k → 96k; record driver +
   nvidia-smi peak per scale. Evidence leg, not a gate: characterizes the
   2-chunk wall, validates jacobi's S_eager model against the flag's real
   footprint, and pins the exact 131k gap the dial must close.
   **Measurement discipline (pauli's W1b reconciliation, bayes 2026-08-14):**
   torch-reserved creeps ~+20 GiB over a run's early steps before plateauing
   (allocator accumulation) — every rung's fit read is taken at MATCHED STEP
   POSITIONS across rungs, or at plateau; never at window-1. A window-1 read
   under-measures by up to ~20 GiB and would mis-size the 131k gap.

**W2 PASS = legs 1+2 green.** Leg 3 is evidence for pauli (the memory leg)
either way.

### Pre-registered diagnostic branch (added 2026-08-14, bayes-approved) — RESOLVED ROUTE (b)

Leg 2's first run FAILED both bars (loss rel 1.26e-4, logprobs max 3.758) with
a decisive structural read: the ON arm's per-token logprob *values* differ from
OFF's (sorted-multiset divergence up to 1.35 rules out a reporting/ordering
permutation) — a forward-value divergence in the executor's decomposition of
GLM-5.2 DSA/THD. The shim's threading is exonerated (fp32_output is a no-op
under the executor; masks/packed_seq_params thread identically; the fp32
boundary is value-exact).

**Determinism probe RESULT — ROUTE (b): the executor is NONDETERMINISTIC on
the GLM-5.2 DSA stack. BLOCKER.** Within-boot run-pairs (same boot, weights,
data, back-to-back) diverge at logprobs max 5.098 / 6.343; boot-to-boot 3.424
— all 3-6 thousand× the ~1e-3 kernel-noise bar, while aggregate losses stay
tight (rel ~3e-5). Characterization: a uniform ~0.08/token run-to-run noise
floor (80× the kernel-noise bar — the executor's timing-dependent reduction
order is measurably noisier than the conventional path even in the bulk) plus
scattered large spikes (up to 5.1) at RECURRING top-k-boundary positions
(datum 3 pos 25055, 92% through, on both within-boot compares). Read: the
executor's interleaved scheduling introduces timing-dependent numeric noise
and the DSA indexer's discontinuous top-2048 selection amplifies it into large
per-token flips. First suspect: the DSA kernel race class (the wheel-saga
prior). **Banked and unaffected: the shim contract (executor reachable +
trains to aggregate precision) and the landmine fix (hardware-validated). The
blocker is the executor's DSA nondeterminism, not the shim.** Executor rungs
(incl. the dial program's executor-dependent rungs) stay HELD; escalate
mcore-side DSA×executor.

## 6. W5+ — the 131k overlap A/B (CONDITIONAL on jacobi's MEMORY_LEG_DECISION.md)

Memory-leg direction (jacobi's interim, 2026-08-13): the leg is the **dial
(P1, upstream per-layer recompute)**, with stock **block+K full recompute**
(`recompute.granularity=full, method=block, num_layers=K` — already exposed
in trainer config) as its zero-code validation proxy and an immediate
no-flag perf lever (~5–6% recompute-tax refund at d16). The valve (P2
offload) is demoted to a composable follow-up; the expert_fc1 "coverage gap"
is root-caused as not-a-bug (LoRA excludes routed experts → fc1 input never
retained → 0.00 GB offloaded is correct).

**Pre-registered memory gate (jacobi's wording):** the 131k flag-ON arm runs
the dial at K set from the block+K sweep; fit bar = d2-ramp peak ≤ 255 GiB
reserved with 2 chunks in flight. Gate fails → no A/B; back to the memory
leg. (Note: block+K is `full` granularity and is itself incompatible with
the flag — mcore asserts `recompute_granularity != 'full'`
(transformer_config.py:2633) — so block+K is the no-flag proxy; the flag arm
uses the dial.)

### Arms (once the gate passes)

- **ON:** PP2/VPP2/CP8/EP8 @131k + `overlap_moe_expert_parallel_comm=true` +
  dial at K. Tree: shim branch + the dial patch.
- **OFF:** the mission config (plain PP2 selective, no flag) — the standing
  fixed-wheel anchors: d4 878–886 tok/s/GPU, d16 984 ±3%.

### Pre-registered bars

1. **Correctness (PRIMARY, gates perf):** d2 canary in the 131k band
   (12.2–12.4), gn comparable to the fixed-wheel 0.36–0.49, loss trains.
   The deferred-schedule-must-train rule: gn drift = STOP + verbatim report.
2. **Perf (SECONDARY):** ≥2 A/B pairs per arm at d4, mean delta vs the arm
   spread (the L1 discipline: |delta| < spread ⇒ perf-indistinguishable).
   d16 escalation only on clean pairs + consistent sign. Report the captured
   fraction of the exposed-a2a mass (gauss: 31.4 s/step exposed at d16,
   ~70% imbalance-wait ≈ 22 s + ≈ 9.4 s wire floor) — the schedule-hideable
   ceiling is 25–28 s/step; the honest read is fraction-captured, not the
   ceiling itself.
3. **Memory (reported, not a bar):** peak vs the OFF arm; the dial's cost
   in tok/s/GPU is part of the ON arm's honest net.

## 7. Risk register / watch items

- **Overlap+LoRA landmine (FIXED + HARDWARE-VALIDATED at W2 leg 1):** under
  LoRA's frozen embedding, the executor's closing `pre_process.backward`
  roots a backward at `decoder_input` with `requires_grad=False` →
  RuntimeError at the first stage-0 backward. Never seen before because
  every prior overlap attempt died in the first forward (memory). Fixed on
  the W2 tree (mcore b37c01f2e; jacobi's hunk from b907b6153). **VALIDATED at
  hardware on 1cd31535 (lovelace, W2 leg 1): the flag-ON warmup0 ran the full
  fwd+bwd with zero RuntimeError.** (Its first contact was delayed by the
  fp32×LoRA-delta loss-forward bug, which died upstream of any backward.)
- **Reporting CP all-reduce stream placement** (scoping §4 risk 2): the
  loss node's `_loss_report` all-reduce runs on the compute stream; rank-
  symmetric schedule ⇒ no desync expected. Watch the first W2 trace for a
  serialization blip.
- **fp32 boundary transient** (scoping §5 risk 4): +192 MB/mb at 131k/CP8 —
  inside the ramp's per-scale peak read; Option A is the escape. **FOUND +
  FIXED at W2 leg 1:** the boundary's fp32 upcast broke the chunked head's
  LoRA-*delta* path — `_project_logits` passed the fp32 hidden to the bf16
  adapter weights (matmul dtype mismatch at the loss node, forward phase).
  The base projection was already fp32-safe; the delta was a latent
  bf16-assumption the boundary exposed. Fixed at the branch (1cd31535): the
  delta's hidden is cast to the base layer's bf16 working dtype — no-op
  conventionally, value-exact under the executor. Regression test added. (The
  landmine fix's first *backward* hardware contact remains pending — this
  failure was in the loss-node forward, upstream of any backward.)
- **VPP2 loss-path bug** (chunked_lm_head.py:196, poincare-domain): blocks
  only the flag-OFF VPP2 control — which this design never runs. Restated
  here so no one "fixes" the control arm by adding VPP2 to it.
- **Warmup:** VPP2 auto-skip is load-bearing on every flag-ON boot; the
  banner must appear in each boot log (doppler records it).
- **First-boot unknowns:** the shim is CPU-tested only until W2 leg 1; the
  contract's real gates are exactly the canary + parity legs above.

## 8. Dependencies and coordination

- jacobi: MEMORY_LEG_DECISION.md (~30–45 min from 2026-08-13 19:5x CDT) —
  §6's gate finalizes when it lands; the block+K sweep shape comes with it.
- poincare: VPP2 loss-path fix is NOT on this design's critical path
  (§5 control-arm note); no ask.
- fermi owns BOX_SCHEDULE.md; W2 fires when the box (**wov4kzq** — q8eg0gq
  was reaped and recreated, 2026-08-13 late) clears W1a/W1b.
