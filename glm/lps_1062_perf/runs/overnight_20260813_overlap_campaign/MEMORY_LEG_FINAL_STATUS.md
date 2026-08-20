# MEMORY LEG — FINAL STATUS (for the campaign report)

Author: pauli (ex-jacobi), 2026-08-14. Scope: the three memory candidates for
making the EP a2a-overlap flag reachable at 131k (the E1 wall: selective
recompute OOMs at ~259 GiB torch-allocated with one microbatch in flight vs
the ~259.5 torch-available ceiling). Sources: MEMORY_LEG_DECISION.md (four
correction notes, honest chronology), NOTEBOOK.md (W1b entries), the W1b
evidence set (w1b_evidence/), and the branch stack below.

## One-paragraph state

The memory leg produced a **built and unit-validated dial** whose purpose is
now blocked upstream, a **complete-but-never-booted offload valve** whose
coverage gap turned out to be correct-by-construction, and a **dead
config-only lever** (block+K) whose failure produced the campaign's memory
model and two standing measurement rules. The dial remains the right
structural answer; its prize (~12–13.5% step time at d16) is documented as
CONDITIONAL on the upstream executor-contract escalation and the empirical
K-ramp that the memory model now anchors.

## 1. The per-layer recompute dial (P1) — BUILT, rung-1-proven, purpose-blocked upstream

**What it is:** `moe_ep_overlap_checkpoint_num_layers` — the first K MoE
layers of each pipeline stage run as opaque whole-layer checkpoints (no
overlap, minimal saved memory), the rest run the stock five-node overlapped
decomposition. Design: `pp2cp8ep8/results/OVERLAP_PER_LAYER_RECOMPUTE_DESIGN.md`
(serre); upstream ask: `UPSTREAM_PROPOSAL.md`.

**State:** built, fresh-subagent-reviewed (APPROVE), and pushed at all three
levels of the vendored stack, branch `jackrao/lps-1062-recompute-dial`:
trainers `6b4dabc6`, megatron-bridge `acbbf06a`, mcore `06393114b` (mcore off
the campaign pin `57efae08b`). Validation: 38/38 trainer tests + 8/8 mcore
config/structure tests under the Mac stubbed-import harness; the 6 wider-dir
failures are the documented pre-existing spawn-stub artifacts (identical on
the base commit). The review's major catch — a pre-existing overlap+LoRA
landmine (frozen embedding → `pre_process.backward` roots a backward at a
non-grad tensor → RuntimeError at the first stage-0 backward) — is fixed on
the branch and also rides lebesgue's shim branch standalone (mcore
`b37c01f2e`).

**Not yet hardware-validated:** the LoRA-trap grad-equivalence test and the
executor-slope memory ramp (kolmogorov's ladder, specs code-verified against
the branch) await box windows; the executor-dependent rungs are HELD pending
the upstream escalation (ramanujan writing it; the design docs are cited).

**Prize (conditional, soft — no pre-registration until ramp data):** coverage
37.5% of layers at the measured K=25 floor → ~12% step time at d16 (replay-a2a
deletion + a2a hiding + recompute-compute refund), ~13.5% if the warm-pool
boot unlocks K=23.

## 2. The offload valve (P2) — plumbing complete, never booted; coverage gap closed as not-a-bug

**State:** three commits on the campaign branch tip region — `f2407a10`
(config plumbing), `6d8b22da` (review fixes), `4e7d5d3e` (the
`max_inflight_offloads` backpressure valve + engagement probe). **Never
booted**: both prior offload boots (the 131k abort, the 32k discriminator)
ran the pre-valve tree `6d8b22da`. The valve targets the first-step-burst
class that killed the 131k arm.

**expert_fc1 = 0.00 GB is correct, not a bug — CLOSED, no fix.** TE v2.16
`_GroupedLinear.forward` (source-verified upstream) sets `inputmats =
[None]*N` when `weight_requires_grad=False`; our LoRA deliberately excludes
routed experts (`lora_targets.py:143-174`), so the fc1 input is never saved
for backward and there is nothing to offload. `moe_act` fires because
activation backward always needs its input. Dropped from future module lists.

**Role going forward:** coverage-multiplier stacked on the dial (offload the
eager layers' stored set → push K below the floor), load-bearing only past
K~23. Moot while the dial is held. Known sub-items if revived: MoE groups are
hardcoded off the CPU pool (`fine_grained_activation_offload.py:365` — the
pinned-alloc tax was the bulk of the −30% stash cost at 32k), and the
dispatcher-combine saves are the first coverage-extension site.

## 3. block+K partial recompute (W1b) — DOES-NOT-FIT at 131k (closed), warm-pool open question

**Verdict (closed under tonight's protocol; bayes guardrail: no
retro-appeal):** K=21 OOM'd twice (trainer-warmup and bench-warmup0; stage-1
pinned at the ~259.5 torch-available ceiling); K=25 survived warmup0 (canary
PASS 12.3242/0.4185) but peaked 265.0 smi / 256.8 torch-reserved — over the
255 poller line — and was terminated before mains. **No steady-state
measurement exists.** The open question (does K=25 fit at steady state/plateau
once the cold-pool first-window excess is absorbed?) is a NEW experiment with
its own pre-registration: `BLOCKK_WARMPOOL_PREREG.md` (conditional — fires
only if box hours remain after W3/W4 + P4 soak).

## 4. The memory model of record (the reconciliation's durable output)

Constants (prior for the dial's empirical K-ramp, not a fit commitment):
base ≈ 139 GiB torch-reserved (mission box, stage-1, d2-class, step-1);
S_eager ≈ 2.9–3.1 GiB/layer/mb-set (fully-eager, DSA attention internals
included); S_ckpt = 0.19 GiB (16384 tok × 6144 × bf16); I = min(M, PP) = 2;
smi ≈ torch + 8–12 GiB (non-torch overhead at 131k). First-window excess at
M=1 ≈ +60–70 torch (cold-pool/compile/weight-load-transient class — the
trainer warmup and the driver's warmup0 both carry it; SKIP_WARMUP trades it
for a cold-pool burst at the first bench window). Postdictions: K=21
steady-at-ceiling OOM ✓, K=25 warmup0 = steady + excess to the decimal ✓.

> **2026-08-14 — CORRECTION (warm-pool probe; evidence
> `blockk_warmpool_probe/PROBE_RESULT.md`, hertz's verdict appended to
> `BLOCKK_WARMPOOL_PREREG.md`): the model is VINDICATED at M=1 and CORRECTED
> at M≥2.** The probe's warmup0 reproduction was exact to 0.15 GiB (256.68
> reserved / 253.25 allocated vs the 256.8/253.2 postdiction; canary
> in-band) — the M=1 constants above stand as predictors of the warmup0
> point. But the "first-window cold-pool excess" framing is REFUTED as an
> absorption story: the first d2 main (M=2) hit an actual CUDA OOM — stage-0
> (leader) ranks spiked 217.3 → 267.4 GiB in ~20 s against the 267.69 cap,
> still climbing, while stage-1 (worker) sat flat at ~264.4. The binding
> term at M≥2 is the **M-scaling of stage-0 activation holdings** (eager
> layers × microbatches in flight at K=25: +50.1 GiB stage-0 demand
> M=1→M=2), which this model — fit on stage-1 — did not carry. There is no
> plateau; the 265-class warmup0 peak is the M=1 STRUCTURAL demand, not a
> transient. Consequences (ruled by hertz): blockK stays dead at 131k; the
> dial K floor stands at 25+; the valve stack is the coverage path; no K<25
> follow-ups. — poincare

**Two standing measurement rules (now campaign rules):**
1. Never mix torch-reserved (`/status`, what the bench JSONs carry) and
   nvidia-smi (poller CSVs) in one fit — an 8–12 GiB gap at 131k (papercut
   pc_248d82831027).
2. Memory A/Bs compare at matched step positions or at plateau —
   torch-reserved creeps +20.8 GiB across a run's early steps then plateaus
   (W1a d16: 143.0→163.8 over steps 6→9, flat through step 16). A window-1
   read is never a steady read.
