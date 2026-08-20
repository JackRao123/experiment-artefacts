# DIAL VALIDATION LADDER — pre-registered runbook for the per-layer recompute dial (P1)

Owner: bohr (validation + box execution when the window lands). Build: jacobi
(`jackrao/lps-1062-recompute-dial` off c30afc3e). Design: serre's
`pp2cp8ep8/results/OVERLAP_PER_LAYER_RECOMPUTE_DESIGN.md`; memory framing:
jacobi's `MEMORY_LEG_DECISION.md`; A/B framing: lebesgue's `OVERLAP_AB_DESIGN.md`
§6. Written 2026-08-13 ~21:0x PDT against the announced interface
(`moe_ep_overlap_checkpoint_num_layers`, first-K-per-stage MoE-only, builder
contract per serre §3.2); sync pass when jacobi's frozen interface lands.

**Gating chain (nothing here runs out of order):** jacobi's build → rung 0/1
(shim-free) → block+K probe (jacobi's zero-code leg — measures S_eager, sets
the dial's K) → lebesgue's W2 shim canary PASS (the 131k flag boot is W5+) →
rungs 2–5.

## Rung 0 — unit, no box (shim-free)

Ownership per the pushed branch's docstring (jacobi's
`tests/unit_tests/a2a_overlap/test_recompute_dial.py`, landed on
`jackrao/lps-1062-recompute-dial`): **config-surface tests ride jacobi's
branch; plan-structure and grad-equivalence tests live with this validation
harness.** So mine to write/execute: grad-equivalence T1–T6 +
plan-structure T7 per `DIAL_GRAD_EQUIV_SPEC.md` (T1–T3 drive the callable
standalone via the §4 mock-node recipe — no executor needed; T4–T7 construct
the real plan at 1 GPU). All written against the frozen, pushed interface
(mcore `06393114b`; contract statements in the spec are code-verified).

## Rung 1 — grad-equivalence execution — **COMPLETE 2026-08-14 (7/7 PASS)**

Executed on wlxj8vw (B200, EP2 via torchrun — the overlap flag is rejected at
EP1, transformer_config.py:2624; noted in the suite docstring). Suite:
`test_recompute_dial_grad_equiv.py` (this folder; staged on-box at the mcore
tests dir). **7/7 PASS on both ranks** (38.9s/36.3s): T1 LoRA-trap
grad-equivalence (adapter grads nonzero + matching, input grad correct), T2
negative control BOTH arms (flip stubbed ⇒ trap fires; b907b6153 source
guard), T3 RNG fork/restore (checkpoint recompute == eager ground truth
bitwise with active dropout), T4 K=0 no-op, T5 K=L vs stock full recompute,
T6 node contract, T7 dense-exclusion/K-semantics. Evidence:
`rung1_dial_grad_equiv_7of7_20260814.log` (this folder). Three harness bugs
found+fixed during execution (EP1 rejection → EP2; MLA q_head_dim 192 →
flash-attn sm100 2CTA assert → 128; T3/T5 test-design bugs — process-global
RNG tracker sharing, and the raw-mcore reference arm lacking the bridge
patch's grad-root force — both fixed and documented in the suite). **The
dial's correctness claims are now hardware-evidenced; the trap is closed by
construction AND the test provably catches it.**

## Rung 2 — CONSUME W1b's measured per-layer table (SUPERSEDED 2026-08-14, bayes)

**Scheduling change (bayes, ~00:1x CDT):** the mission box wprm693 (2×8 B300
ali) landed, so **W1b (block+K21 @131k on the REAL config, BT_DIAL_MEM_PROBE
on) measures the real per-layer table hours from now — this supersedes the
cut-down-snapshot rung 2 entirely.** The B200-side snapshot plan and the B200
contract smoke are DEAD unless the mission box dies again. My rung 2 becomes:
consume W1b's measured `[dial_mem]`/block+K table from the mission box,
compute the dial's K from the measured per-layer deltas (the formula below),
and fold the result into the dial's box window prep. The B200 box (wlxj8vw)
is rung-1-complete and parked.

- K set from W1b's **measured** S_eager/S_ckpt (never the 2.25 GiB bound —
  risk §6.3 retired by measurement). **W1b closed DOES-NOT-FIT (2026-08-14):
  blockK25 = 265 GiB reserved vs the 255 bar — 10 GiB over ⇒ fitting needs
  MORE recompute, K≈28–29 territory by pauli's slope (overlap coverage shrinks
  toward ~11/40 layers). The K number of record is pauli's in-flight model
  revision — cite that, not any spec-side estimate.** (Correction on record:
  my earlier "K~21" aside ran the wrong direction.)
- **Mem probe (format PINNED, code-verified on mcore `06393114b`):**
  `BT_DIAL_MEM_PROBE=1`; one `logger.warning` per opaque layer per
  microbatch-forward per rank:
  `[dial_mem] rank=<r> layer=<global 1-78> kind=ckpt alloc_before_mib=<f>
  alloc_after_mib=<f> delta_mib=<f>`. Parser keys on
  `^\[dial_mem\] rank=(\S+) layer=(\S+) kind=(\S+) alloc_before_mib=(\S+)
  alloc_after_mib=(\S+) delta_mib=(\S+)$`; global layer number ⇒ stage
  attribution free. Eager layers unsampled by design (S_eager from the block+K
  sweep, zero-touch).
- K formula (from jacobi §3/§5): pick largest K with
  `I × (K·S_ckpt + (L−K)·S_eager) ≤ headroom` (I=2 in-flight, L=40 stage-1,
  headroom = 275 GiB budget − measured base) — inputs from W1b/pauli's
  measured table only.
- (Retained for the mission box's dial boot, when scheduled) d1-ramp bar:
  peak reserved ≤ 255 GiB with 2 in-flight; miss → recompute K from the
  measured delta, one retry; second miss ⇒ memory model wrong ⇒ memory leg.

## Rung 3 — d2 canary (correctness gate; every subsequent rung inherits it)

- Loss 12.2–12.4 band; gn comparable to the fixed-wheel 0.36–0.49; loss
  trains across steps. gn drift = STOP + verbatim report (the
  deferred-schedule-must-train rule).
- **Dial-specific addition:** adapter grad norms logged and NONZERO per step.
  The loss canary alone is too slow to catch silent-zero-adapter-grads at this
  LR (serre §6.1) — the trap's at-scale detector is the grad-norm line, not
  the loss curve.

## Rung 4 — d4 A/B vs block+K (NOT vs 918/984)

jacobi's pairing: the OFF arm is **block+K at the same K** (granularity=full,
method=block, num_layers=K — stock, flag-incompatible). Both arms pay
recompute on the same K layers and run eager on the same L−K; ON−OFF isolates
the **pure overlap effect** (the recompute refund is shared and cancels).

- ON: PP2/VPP2/CP8/EP8 @131k + `overlap_moe_expert_parallel_comm` + dial at K.
  OFF: `trainer_pp2cp8ep8_131k_blockK{K}.json` (jacobi's starter artifact
  pattern, K substituted).
- Bars: ≥2 pairs at d4; win = ON > OFF + 2% with the ±2% within-boot control
  gate (controls tripping ⇒ discard and repeat, per jacobi §5); mean-vs-spread
  rule (the L1 discipline). Reference shape: block+K d4 predicted ≈ 915–955
  (+4–8% over the 878–886 anchor) — the dial arm must beat THAT.
- d16 escalation only on clean pairs + consistent sign. Report the
  fraction-captured of the 25–28 s/step schedule-hideable ceiling (gauss:
  31.4 s exposed at d16 ≈ 22 s imbalance-wait + 9.4 s wire floor) — the honest
  read is fraction-captured, not the ceiling.
- Memory reported, not a bar: ON-arm peak vs OFF-arm peak is the dial's honest
  cost line.

## Rung 5 — trace check (rides the first clean d4 ON-arm)

`BT_PROFILE_RANKS=0,8`, traces pulled off the volatile dir IMMEDIATELY
(standing rule). Expected: on eager layers, dispatch/combine kernels on the
comm stream overlap the paired microbatch's compute; on checkpointed layers,
the comm stream is idle (expected, not a bug — serre §8). Rank-0 vs rank-8
stage asymmetry reported. An eager layer showing NO overlap = the flag isn't
engaging on the eager side → investigate before trusting the A/B.

## Standing constraints (all rungs)

Fixed wheel only (cudnn-frontend 1.27.0); `BT_SAVE_STATE_SYNC=1` everywhere;
`BT_SKIP_WARMUP=1` on flag-ON boots + the VPP2 auto-skip banner (6fa3bfa7)
recorded per boot; `wait_trainer_health.sh` only; one-variable discipline;
memory abort >255 GiB reserved; canary band 12.2–12.4.

**First-boot checklist item (jacobi, code-verified):** every flag-ON boot must
run the dial branch tip — or verify the `b907b6153` hunk present — because of
the **pre-existing PreProcessNode grad-root landmine**: under LoRA the frozen
embedding makes `decoder_input` arrive `requires_grad=False`, and the
executor's closing `pre_process.backward` raises RuntimeError at the first
stage-0 backward. Blocks ANY overlap+LoRA boot, dial or not; the fix is in the
branch. Verify = grep for the hunk (`PreProcessNode`, "Force the grad root")
or the boot dies loudly at the first stage-0 backward.

## Dependencies (tracked)

1. ~~jacobi's frozen interface~~ — **RESOLVED** (trainers `6b4dabc6` / bridge
   `acbbf06a` / mcore `06393114b`, all pushed; sync pass DONE,
   code-verified). [Roster note 2026-08-14: jacobi's build lane is now pauli
   (ex-jacobi); box mechanics = grothendieck; kepler = jacobi; lebesgue =
   ramanujan.]
2. ~~per-layer stored-memory log format~~ — **RESOLVED** (pinned in rung 2).
3. ~~block+K probe / K calibration~~ — **SUPERSEDED into W1b** (bayes,
   2026-08-14): the mission box runs block+K21 @131k with BT_DIAL_MEM_PROBE;
   my rung 2 consumes that table.
4. lebesgue's→ramanujan's W2 shim canary PASS (131k flag boots are W5+ per
   OVERLAP_AB_DESIGN §6).
5. Box window assignment from bayes (W1c B/F window on the B300, after
   W0/W1a/W1b).

## Decision rules (summary)

| rung | fail action |
|---|---|
| 1 | back to jacobi with spec §3 triage class; no box rerun until fixed |
| 2 | one K retry from measurement; second miss ⇒ memory model wrong ⇒ memory leg |
| 3 | STOP + verbatim report; no forward debugging |
| 4 | > +2% beyond spread ⇒ d16 escalation; \|Δ\| < spread ⇒ perf-indistinguishable, dial stays default-off; negative beyond spread ⇒ report honestly (dial costs more than it hides at this K) |
