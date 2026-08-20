# Merge queue — August campaign ship work

Plain-language version, updated 2026-08-20 after Jack's review. The original
dense version (full history) is archived at
`MORNING_MERGE_QUEUE_ORIG_20260813.md`.

**How to read this:** one PR is in flight now (Active). The Queue is what
merges next, in order. The Backlog is everything deferred — revisit at the
end.

---

# ACTIVE — in flight now

## ★ THE CHECKPOINT — PR #1070: enable the 2-node setup + make the pipeline actually pipeline

**Jack, 2026-08-20: this PR is the baseline. All future work stacks on top
of it (branch `jackrao/lps-1062-pp2-mn-combined` until it merges, then
main). Nothing rebuilds below this point.**

**Status: DRAFT PR OPEN —
https://github.com/basetenlabs/trainers/pull/1070 (helmholtz, 2026-08-20).
5 commits, ported onto restructured main, local tests + lint/typecheck
green, CI running. Jack flips it from draft when satisfied. One review
flag, documented in the PR body: the lifted safety block exempts DSA
models broadly rather than only the exact tested layer layouts — a
tighter guard can ride the "fail loudly" backlog bundle later.**

Two pieces:

**The enablement:** we split the model across two machines (half the layers
each) while also splitting each sequence across the 8 GPUs within a
machine. The code had a safety block refusing that combination as
"untested"; it's now tested to death. This defines the exact layer split we
use and lifts the block for our model type.

**The speed fix:** the most expensive bug of the campaign. Work is supposed
to flow through both machines at once, assembly-line style, but the code
handed the scheduler one chunk at a time — so the machines mostly took
turns idling. The fix hands the scheduler all 16 chunks at once. Worth
**+56%**.

Porting note: main was restructured since these commits (backend files
moved to `server-megatron-bridge/src/trainers_server_megatron_bridge/`), so
this is a port across renames, not a plain cherry-pick. The PR body must
also disclose that the headline numbers were measured with the TF32 change
(now in the backlog) turned on.

> Handles: cherry-pick `21d0c578` (enablement), then `36c3c8f4`,
> `b34b7aed`, `d1c939c3`, `8b0ef108` (speed fix). `b8d868ff` is excluded —
> never ran on hardware. Drafts: PR_DRAFT_INFRA.md (INFRA-A) +
> PR_DRAFT_MN_PACKING.md. Fresh branch off current main, exactly these
> commits.
>
> Added 2026-08-20 per Jack (6th commit, `1e62a604`): the golden trainer
> config row for GLM-5.2-FP8 on B300 @131k (2 nodes × 8 GPUs,
> TP1/PP2/EP8/CP8). Finding surfaced by its validation: router replay is
> incompatible with pipeline splitting today (trainer router names are
> per-stage, sampler route stamps are whole-model), so the PP2 row ships
> replay-off and automatic replay stays PP1-only. Making replay work
> under PP needs a stage-to-global mapping — tracked in the backlog.

---

# QUEUE — merge next, in this order

## Q1. Fix the "save safely" switch that does nothing

First in the queue because it's a live risk, not a speed win. Saving a
checkpoint mid-training can hang the whole trainer (known bug when
sequence-splitting is on — it fired at the end of our 60-step stress test:
the background write stalled and all 16 GPUs froze). The env switch
everyone sets to avoid it (`BT_SAVE_STATE_SYNC=1`) is read **nowhere** in
the current code — it silently does nothing, and the risky save mode is
hardcoded on. The fix is one small already-written change; the safe path is
fully tested (real 16-GPU save in ~71 s).

> Handle: cherry-pick the `megatron_config.py` change from `db5d1826`.
> Evidence: EXPORT_TEST.md section F1.

## Q2. +5% speed: cache the repeated CPU bookkeeping

Every step the CPU recomputed identical layout bookkeeping ~54,000 times,
blocking the GPUs. Two caches remember it instead. Measured **+5.0%** (in
the predicted range), clean 60-step stress test, off-by-default switches
(off = byte-identical). Fully validated, sitting ready. Merge after the
active PR so its evidence matches main, then update the pointers so the
trainer picks it up.

Wording rule if described anywhere: our model naturally varies run-to-run
(an attention tie-break quirk), so say "differences are within the model's
natural run-to-run noise" — never promise exact-match parity.

> Handles: Megatron-LM fork PR #26; then branches
> `jackrao/lps-1062-bf-rebuild` on megatron-bridge (`0e356eb2`) and
> trainers (`e864115a`).

---

# BACKLOG — deferred, revisit at the end

## Groundwork for interleaved pipelining (VPP) — demoted from queue 2026-08-20

VPP = interleaved 1F1B: each machine holds several small slices of the
model instead of one block, shrinking the waiting bubble. Demoted on the
numbers: the bubble is already near its floor (7.2% measured vs 5.9%
ideal), so VPP2's gross win is ~2.8% of the step — before paying for
double the cross-machine messages and more in-flight activation memory on
a memory-bound stack. It has never completed a training step on hardware
(five failed boots; known unfixed crash in the loss path,
`chunked_lm_head.py:196`). Merge someday as inert substrate (zero behavior
change at VPP=1) — do not invest in hardware bring-up unless a future
overlap design needs it. Touches the same file region as the active PR;
whichever lands second reconciles a small conflict. Two required verbatim
PR-description sentences are in the archived original.

> Handle: cherry-pick `61c41d9e`, `8c13ed31`, `6fa3bfa7`.

## Faster final-layer math (TF32)

Runs the model's last step (output → word probabilities) in TF32, a faster
GPU math mode. Measured **+6.7% by itself** (550 → 589 tok/s/GPU;
environment-only change ruled out at +0.4%). Behind an on/off env switch.
Deferred by Jack: modest win, needs its own verification pass.

> Handle: trainers PR 995 (open, `BT_TF32_LM_HEAD=1`).

## Two debugging switches

Pick which GPUs get profiled + a peak-memory report. Off by default, unit
tested, zero behavior change. Merge whenever convenient.

> Handle: cherry-pick `ad39a97d`, `760021be`.

## Build system can install a stale local package over the pinned one

This is how we lost a night to a "kernel corruption" hunt that was actually
an old buggy wheel beating the lockfile. Fix: build checks freshness
against the lockfile. (Old speed anchors were measured on the buggy wheel —
numbers may move when re-measured.)

## Sparse-attention training loss counts padding as real data

Harmless for our current runs (that loss's weight is zero for us), but will
silently corrupt full fine-tuning. One-field fix.
TAIL_PAD_DSA_SOURCE_MEMO.md section 3.4.

## "Fail loudly" bundle

Three places where the code silently falls back to a default instead of
erroring when it can't do what was asked (two in the VPP layout lookup, one
in the sparse-attention config path, `dsa.py:1599`). One small PR closes
all three.

## Router replay under pipeline splitting

Found while adding the PP2 golden config row (PR #1070): router replay —
feeding the sampler's recorded expert-routing choices back to the trainer —
only works without pipeline splitting, because the trainer names router
layers per-stage while the sampler's route stamps span the whole model. The
PP2 config ships replay-off. Fix needs a stage-to-global layer mapping;
naively removing the guard would silently misalign routes.

---

# Waiting on Jack's go

- **File the save-hang bug upstream?** Write-up ready (EXPORT_TEST.md
  section F1). Goes nowhere until you say so.

# Closed

- **Installer race fix** — PR 987, merged 2026-08-15.
- **Network settings package** — rejected; PR 1000 closed 2026-08-20
  (zero effect on the 2-node setup; cross-node configs not planned).
