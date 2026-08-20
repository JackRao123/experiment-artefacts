# LANE HANDOFF — pauli (ex-jacobi): memory-leg scout + dial builder

Date: 2026-08-14. To: turing's named successor for the memory/dial lane.
State: **fully delivered, parked in good standing; the dial's
executor-dependent rungs are HELD pending the upstream escalation.** Nothing
is mid-flight; no box is held by this lane.

## What this lane did (the 30-second version)

Scouted the memory leg for making the EP a2a-overlap flag reachable at 131k
(the E1 wall: selective recompute OOMs at ~259 GiB torch-allocated with one
microbatch in flight). Found and had ratified: (1) the offload valve's
plumbing is complete-but-never-booted and its headline "expert_fc1 coverage
gap" is correct-by-construction (frozen LoRA experts save no fc1 input — TE
v2.16 source-verified); (2) a zero-code third leg (stock block+K partial
recompute) that reordered the box schedule; (3) then BUILT the per-layer
recompute dial (the structural answer) — built, reviewed, pushed, and
rung-1-validated. The four-round W1b memory-model reconciliation produced the
campaign's memory model and two standing measurement rules.

## Doc pointers (all on disk)

Campaign folder `experiment_artefacts/glm/lps_1062_perf/runs/overnight_20260813_overlap_campaign/`:

- `MEMORY_LEG_FINAL_STATUS.md` — **the report section** (bayes: cited
  verbatim). Dial/valve/blockK final states + the memory model of record +
  the two measurement rules.
- `MEMORY_LEG_DECISION.md` — the original scout, with four in-place
  correction notes (honest chronology: S_eager 2.25→2.94, a model rejection
  walked back, the metric-artifact resolution).
- `BLOCKK_WARMPOOL_PREREG.md` — the conditional blockK revival probe
  (pre-registered, fail-closed, one boot; fires ONLY on leftover box hours
  after W3/W4 + P4 soak).
- `DIAL_GRAD_EQUIV_SPEC.md` + `DIAL_VALIDATION_LADDER.md` — kolmogorov's
  validation specs, code-verified against the branch.
- `rung1_dial_grad_equiv_7of7_20260814.log` — rung-1 evidence: **7/7 PASS**
  (T1 LoRA-trap grad equivalence, T2 negative control, T3 RNG fork/restore,
  T5 K=L vs stock full recompute, T4/T6/T7 structure).
- `w1b_evidence/` — the W1b CSVs/runlogs/trainer peak lines + W1a anchors.

Design + spine: `pp2cp8ep8/results/OVERLAP_PER_LAYER_RECOMPUTE_DESIGN.md` and
`UPSTREAM_PROPOSAL.md` (serre; ramanujan's upstream escalation cites these);
`pp2cp8ep8/NOTEBOOK.md` (the spine — W1b entries carry the reconciliation).

## The branch stack (all pushed)

`jackrao/lps-1062-recompute-dial` at three levels: trainers `6b4dabc6`,
megatron-bridge `acbbf06a`, mcore `06393114b` (mcore off the campaign pin
`57efae08b`). The PreProcessNode grad-root fix also rides lebesgue's shim
branch standalone (mcore `b37c01f2e`) — **merge note: both branches carry the
same hunk; trivial either-way resolve when they meet.** Trainer config
surface: `comm_overlap.moe_ep_overlap_checkpoint_num_layers` (requires the
big overlap flag; parse-time enforced). Memory probe: `BT_DIAL_MEM_PROBE=1` →
`[dial_mem] rank=.. layer=.. kind=ckpt alloc_before_mib=.. alloc_after_mib=..
delta_mib=..` per opaque layer per microbatch.

## Open items (in priority order)

1. **Dial executor rungs (HELD pending upstream).** kolmogorov owns the
   validation ladder: rung-2+ = d1 executor-slope memory ramp → K recomputed
   from the executor-measured slope → d2 canary → d4 A/B vs block+K. The
   memory gate is TWO-STAGE by ratified pre-registration: the executor
   re-measure happens BEFORE any canary K commitment (the plain-path
   constants don't transfer verbatim through the executor's per-node
   detach/free machinery).
2. **The warm-pool blockK probe** — conditional, self-driving from
   `BLOCKK_WARMPOOL_PREREG.md`; the test is the plateau read vs the
   pre-registered ~245–250 smi.
3. **The valve engagement probe** (one boot; revived as coverage-multiplier
   inside the dial program, not a standalone leg): config + bars sketched in
   MEMORY_LEG_DECISION.md §"Valve probe"; the CPU-pool-for-MoE patch
   (`fine_grained_activation_offload.py:365` hardcode, ~3 lines, needs the
   freeze-exception process) is the steady-state-cost fix.

## Traps (hard-won; read before touching any of this)

- **Triple-metric footgun:** torch-allocated < torch-reserved (`/status`,
  what bench JSONs carry) < nvidia-smi (poller CSVs) — 8–12 GiB gaps at 131k.
  NEVER mix metrics in one fit (papercut `pc_248d82831027`). The 255 abort
  line is POLLER-reserved; the cap is 267.69 GiB smi; torch-available ≈ 259.5.
- **Matched-step rule (now a campaign rule):** torch-reserved creeps +20.8
  GiB over a run's early steps then plateaus — memory A/Bs at matched step
  positions or at plateau; a window-1 read is never a steady read.
- **Cold-pool vs trainer-warmup:** SKIP_WARMUP trades the trainer warmup's
  ~+60 GiB transient base (weight-load unsettled, M=1) for a cold-pool
  first-window burst (~+69 torch at M=1). Never cold-pool into a
  near-ceiling config; settle-then-warm or graduated driver warmup.
- **The LoRA trap (dial-critical):** `CheckpointFunction` never fires
  backward without a grad-requiring input — the patch forces `requires_grad_`
  on the node's *existing* detached input leaf (re-detaching would crash
  `backward_impl`; the review verified). T2 is the negative control.
- **NoopScheduleNode needs `backward_dw`** — the schedule calls
  `mlp.backward_dw()` unconditionally (added on the branch).
- **Validation ordering:** the dial field is written onto the provider after
  the first `finalize()`, but `runtime_config_update` → `cfg.validate()`
  re-runs `model.finalize()` AFTER `comm_overlap.setup()` lands the big flag
  — mcore's validators DO re-fire there; the ordering is load-bearing
  (comments on-branch are corrected to say so).

## Known-unresolved (bounded, non-blocking)

- The trainer-warmup +60 GiB transient base is characterized as a class
  (weight-load-unsettled), not root-caused to a line.
- The stage-0 model fit carries a 2–3 GiB residual (dense-layer pricing
  nuance); stage-1 binds all decisions and fits exactly.
- Whether the executor shifts S_eager (the transfer caveat) — that's what
  rung-2's d1 ramp measures.

— pauli, 2026-08-14. Responsive until released.
