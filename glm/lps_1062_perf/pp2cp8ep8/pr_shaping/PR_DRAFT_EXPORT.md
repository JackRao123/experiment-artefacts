# PR DRAFT — EXPORT package (dedekind lineage @ db5d1826)

Drafted by weierstrass, 2026-08-13 night, for Jack's morning review.
Draft only — no PR created, nothing pushed. Source branch:
`jackrao/lps-1062-pp2-export` (pushed, in sync with origin), worktree
`~/Documents/wt-pp2-export`. Full evidence write-up:
`export_test/EXPORT_TEST.md`; durable evidence (survives box reclaim):
`export_test/evidence/`.

Terms used below: **PP** = pipeline parallelism; **CP** = context
parallelism; **LoRA adapter export** = the `save_weights_for_sampler` path
that gathers the trained low-rank adapter matrices from all parallel ranks
and writes `adapter_model.safetensors` + `adapter_config.json` for the
sampler; **DCP** = PyTorch distributed checkpoint, the format
`/save_state` writes; **F1** = tonight's finding that DCP `/save_state`
hangs under CP>1 (EXPORT_TEST.md §F1); **G1/G2** = the two boot gates in
`megatron_config.py` that block PP>1+CP>1 today (EXPORT_TEST.md §3).

---

## What this branch is

Four commits, one mission (Jack's constraint 3): prove — or find and fix
breakage in — the LoRA adapter export path when both PP>1 and CP>1, using
a minimal model (Qwen3-0.6B), not the full GLM.

| commit | content |
|---|---|
| 8a4ae08b | the export tests (L1 legs) + the G1/G2 test-only gate relaxations + an `attention_backend` kwarg on the `mb_cluster` test harness |
| ef2e8036 | the L2a faithfulness leg (DCP round-trip PP2/CP2 → PP1/CP1, bit-identical export comparison) |
| 6d962102 | L2a marked skip with a pointer to F1 (its vehicle — DCP save under CP>1 — is what hangs); on-box validation results documented in the module docstring |
| db5d1826 | `BT_SAVE_STATE_SYNC=1` toggle + the PP1/CP2 sync-save probe test |

## Verdict being shipped (evidence, all on-box 2026-08-12, w56lorq 2×8 B300)

**The PP>1+CP>1 LoRA adapter export path WORKS** within the stated scope
(EXPORT_TEST.md §7):

- **L1 PP2/CP2 (4 GPUs) and L1 PP2/CP4 (8 GPUs): green.** Export completes
  in seconds under a 60s deadlock cap; the exported key/shape set is
  byte-identical to the PP1/CP1 reference (392 keys = 28 layers × 14
  tensors — catches any dropped-layer-subset gather bug); values
  non-degenerate (all finite; every `lora_A` nonzero; every post-step
  `lora_B` nonzero — an all-zero B after a real optimizer step would mean
  dropped gradients).
- **L2b statistical faithfulness: green.** Same datum, same learning rate,
  fresh PP1/CP1 reference run: 392/392 keys match, worst per-tensor
  standard-deviation relative deviation **1.3%** against a loose 50% gate.
  By design this is report-not-gate (EXPORT_TEST.md §6 — cross-topology
  LoRA-init RNG streams differ, so exact-value comparison is invalid by
  construction). The committed artifact is the evidence, not a test:
  `export_test/evidence/ws_ref_pp1cp1/`, `ws_test_pp2cp2/`,
  `workers_pp1cp1_l2b.log`, `workers_pp2cp2_l2b.log`.
- **L2a (the strong, RNG-agnostic leg): NOT RUN — blocked by F1**, the
  DCP save hang below. The test is committed, marked `skip` with a pointer,
  and re-enables the moment the CP>1 DCP save path is fixed.

Explicitly not validated (the test does not claim): GLM-5.2 DSA itself
(MoE + EP>1 + grouped-expert adapter branches are not exercised by a dense
model), TP>1 combined with PP>1+CP>1, training numerics of any CP>1 stack
(export mechanics only), multi-node anything, long-run stability.

## F1 and the toggle — the ship vehicle

**F1 (EXPORT_TEST.md §F1): DCP `/save_state` hangs under CP>1.** Every
trainer hardcodes `async_save=True` (no config knob), and under CP>1 the
async writer never completes: the `nvidia_resiliency_ext` async-writer
daemon crashes on its first queue item (`RuntimeError: received 0 items of
ancdata` in torch's CUDA-storage fd-passing,
`multiprocessing/reduction.py:164`), worker ranks park in
`schedule_async_call → queue.join`, leader ranks park at the NCCL barrier
in `save_checkpoint`, the trainer stays HTTP-200 alive but wedged.
Attribution is CP-triggered and PP-independent (probe matrix: PP2/CP1
PASS, PP1/CP2 HANG, PP2/CP2 HANG). **Confirmed at production scale**
tonight: the real GLM-5.2 PP2/CP8/EP8 @131k 16-rank trainer, clean-init,
never completed a bounded 10-minute `/save_state`
(`export_test/evidence/big_trainer_save_probe/`, 42 files including
16-rank py-spy dumps and the full trainer log with the ancdata traceback).
Blast radius: CP>1 is a shipped golden topology (GLM-5.2-FP8 CP32 and
CP16), so every CP>1 production run that calls `save_state` wedges today.
One honest caveat carried from EXPORT_TEST.md: the reproductions are on
the devbox venv (cu13 + dsatopk1 cudnn shim); a prod-image repro is still
needed — but the production-scale repro tonight was the real trainer on
the real topology, so "small-model artifact" is ruled out.

**The toggle is the confirmed workaround.** db5d1826 adds
`BT_SAVE_STATE_SYNC=1`, which flips `async_save` off (default unchanged:
async stays on). The sync path was verified tonight at **both** scales:

- small model: PP1/CP2 sync save completed (113s wall including boot,
  60MB checkpoint) — `test_qwen3_06b_pp1cp2_sync_save_state_probe` green;
- **production scale: PP2/CP8/EP8 @131k sync `/save_state` completed in
  70.8s, 537MB checkpoint written**
  (`/tmp/checkpoints/pp2cp8-save-probe-sync/iter_0000000`).

F1 is closed in both directions: async wedges, sync completes. The
escalation package (paste-able summary in EXPORT_TEST.md §F1) is parked
awaiting Jack's word — per standing rule, nothing outward-facing has been
filed autonomously.

### Recommendation on the toggle

Cherry-pick the `megatron_config.py` hunk of db5d1826 (the 5-line
`sync_save` lever, not the test) as a tiny standalone mainline PR. It is
orthogonal to everything else tonight, default-preserving, and it is the
only way a CP>1 production run can checkpoint at all until the async path
is root-caused. The commit comment currently says "Not a production
setting" — Jack's call whether to keep that framing or re-label it as the
supported workaround; either way the F1 escalation goes out with it
attached.

## Gate relationship — G1/G2 stay TEST-BRANCH-ONLY (deliberate)

8a4ae08b relaxes the two boot gates so the minimal-model tests can run:

- **G1** `_validate_thd_context_parallelism`: PP>1+CP>1 downgraded from
  hard error to loud warning.
- **G2** `_apply_thd_cp_provider_overrides`: generic GPT providers (Qwen3)
  allowed through CP with the default P2P transport (was: DSA and
  hybrid-Mamba only).

**These two hunks must not merge to mainline.** The mainline version of
the G1 question is the infra PR's DSA-only exemption (21d0c578), which
keeps the hard error for every non-DSA stack and ships with tonight's
validation evidence. The two branches rewrite the same function and the
same unit tests in opposite directions — merging both is a textual
conflict and a semantic clash (warn-for-everyone vs
reject-everyone-except-DSA). Reviewers: if you see both, merge the infra
one only.

Consequence, stated plainly so nobody is surprised later: because Qwen3 is
not DSA, the export integration tests **cannot boot on mainline gates** —
they need G2. That is why this branch is the standing validation harness
rather than a mainline test PR. If Jack wants the tests mainlined anyway,
the path is: rebase onto the infra exemption, drop the G1 hunk, keep G2 as
a clearly-marked test-only escape hatch (or gate the Qwen3 legs behind an
env flag) — a deliberate follow-up, not part of this package.

## What merges, summarized for review

1. **Mainline candidate (recommended):** the `BT_SAVE_STATE_SYNC` lever
   (db5d1826's `megatron_config.py` hunk) — the F1 workaround vehicle.
2. **Stays on the branch:** the export test suite (L1 PP2/CP2 + PP2/CP4,
   L2a skipped-with-pointer, L2b evidence), the sync-save probe test, the
   G1/G2 relaxations, the `mb_cluster` kwarg.
3. **Not in this package:** the F1 root-cause fix (upstream-looking;
   hypothesis and evidence in EXPORT_TEST.md §F1 — fd-passing failure
   class, unverified by instrumentation).

Self-review findings for this lineage: SELF_REVIEW.md (one deliberate
divergence, one evidence-not-test note, no blockers).
