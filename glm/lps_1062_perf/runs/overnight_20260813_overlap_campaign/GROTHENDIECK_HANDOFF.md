# LANE HANDOFF — grothendieck (W1b driver → support), for turing's named successor

2026-08-14 ~05:0x CDT. Lane state: CLOSED, at an atomic boundary, zero
in-flight work, zero box processes owned by me. Rehydrate with this file +
GROTHENDIECK_W1B_RUNBOOK.md (same folder) + NOTEBOOK.md tail.

## Verdict of record (W1b, block+K partial recompute @131k)

**blockK does NOT fit at 131k on this stack.** K=21: OOM ×2 (trainer
warmup, then bench warmup0 post-settle — 267.4 GiB poller-max stage-1).
K=25 (single authorized retry): warmup0 completed with in-band canary
(loss 12.3242, gn 0.4185) but peaked 265.0 GiB poller-reserved vs the 255
bar (trainer-reported 256.8 reserved / 253.2 allocated); d16 projection
~282 = unreachable. No third boot per pre-registration. The probe's
designed product shipped: **S_eager = 2.94 GiB/layer/mb** (not 2.25 — that
was the E1-selective constant with core_attn checkpointed; fully-eager
layers retain DSA attention internals). Dial K math inherits this (pauli
has the evidence set + dual-metric flag).

## Open items owned by this lane

NONE. All delivered: evidence set (sha256-verified, `w1b_evidence/`) →
pauli; verdict → bayes/lovelace; NOTEBOOK entries written; papercut filed
(pc_8f3109ed42d0, wait_trainer_health exit codes).

## Box-side state I hold (wprm693)

- NOTHING running. Wheel belongs to **lovelace** (they hold pre-boot-1
  pending kolmogorov's ruling — coordinate before ANY box contact).
- Files I staged on the shared FS (harmless, superseded):
  `lps1062_pp2/trainer_pp2cp8ep8_131k_blockK21.json` and
  `~/.cache/user_artifacts/trainer_config_blockK21.json`. Live retry
  config is blockK25 (Mac-side canonical sha256 ddc75eed241b…; lovelace
  staged box-side themselves).

## Traps (each cost someone time tonight)

1. **Trainer warmup pads to full shape.** The seq-64 warmup datum pads to
   the 131k packed buffer → trainer warmup IS a full-shape M=1 fwd+bwd on
   unsettled load transients. Memory-tight configs: BT_SKIP_WARMUP=1 and
   let the bench driver's warmup window absorb compile (it did: K=25
   warmup0 193.4s compile-heavy, then sane).
2. **start_trainer.sh TRUNCATES trainer_srun.log.** If a boot's log
   matters, copy it before the next start. (Both K=21 OOM logs were lost
   this way; verbatim numbers preserved in w1b_evidence/W1B_OOM_EVIDENCE_NOTE.md.)
3. **Two memory metrics, don't mix.** Poller nvidia-smi max ≈ trainer
   reserved + ~8 GiB non-PyTorch overhead. The 255 fit bar is
   POLLER-reserved (pauli's instruction). E1's 258.8 was torch-allocated.
4. **wait_trainer_health.sh exits 1 for both death and 3-min timeout.**
   Discriminate on TEXT ("TRAINER PROCESS DIED" vs "TIMEOUT").
5. **No blocking waits — zero tolerance (Jack).** Background everything;
   stay message-responsive. A blocking session hands its lane off. My 1h
   stall cost me this lane; the rule is real.
6. **Bench JSON only at full completion.** A terminated run leaves runlog
   + mem CSVs, no JSON. fold_mem.py then crashes on the missing JSON
   (harmless, expected).
7. **CPFS close-to-open quirk:** sha-verify staged files from BOTH nodes
   (sibling can read NULs).
8. **Bench evidence is volatile box-side:** pull Mac-side immediately;
   boxes die.

## Who owns what (fleet map at handoff)

turing = orchestrator (reports route to them). lovelace = wprm693 wheel +
lifecycle. pauli = dial constants/memory model. jacobi = trace reads
(:9006/:9007). kolmogorov = dial ramp (executor-side S_eager re-measure).
New fleet: hertz/curie/kirchhoff/poincare/dedekind/nash.
