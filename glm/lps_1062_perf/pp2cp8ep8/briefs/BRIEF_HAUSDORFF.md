# BRIEF — hausdorff: PP>1+CP>1 LoRA export verdict + CP>1 DCP save hang (handoff from dedekind, 2026-08-12)

You are **hausdorff**, succeeding **dedekind** (kimi k3, export-test stream).
Orchestrator: **borel** (succeeded maxwell/pauli). Reply/report via:
`~/.agents/scripts/send-message.sh borel "hausdorff: <message>"`.
Report on milestones/blockers; don't go silent >30 min while active.

## TL;DR of dedekind's completed work (no redo needed)

**Mission (Jack constraint 3) is ANSWERED: PP>1+CP>1 LoRA adapter EXPORT
works.** Validated 2026-08-12 on w56lorq (2×8 B300), Qwen3-0.6B minimal model:

- L1 PP2/CP2 (4 GPU): GREEN, 178s — export completes, no hang.
- L1 PP2/CP4 (8 GPU): GREEN, 134s.
- Exported key/shape set byte-identical to PP1/CP1 reference (392 keys = 28
  layers × 14 LoRA tensors); all values finite, post-step lora_B nonzero;
  adapter_config.json correct (r=8, alpha=32, 7 targets, auto_mapping stamped).
- L2b statistical faithfulness vs fresh PP1/CP1 reference (same datum/LR):
  GREEN — worst per-tensor std deviation 1.3%.
- L2a (DCP round-trip bit-exactness): NOT RUN — blocked by Finding F1 below.

Full verdict with explicit validated/NOT-validated scope (for gibbs's
mainline gate exemption 21d0c578 to cite):
`experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/export_test/EXPORT_TEST.md` §7.
Scope headliner: validated for dense GPT + P2P CP + TE fused attention,
single-node, DP=1. NOT validated: GLM-5.2 DSA (MoE/EP/grouped-expert emitters/
allgather transport), TP>1 combos, CP numerics, multi-node, long-run, and
**DCP checkpoint save/load under CP>1 (separately tested, BROKEN — see F1)**.

## Finding F1 (the thing that matters most): DCP /save_state hangs under CP>1

- **What breaks**: `/save_state` (DCP checkpoint, the Tinker/loops save API)
  hangs forever under CP>1. Every rank parks in mcore dist_checkpointing
  `maybe_finalize_async_calls` → `is_current_async_call_done` (async_utils.py)
  polling loop; the forked async-writer child processes spin (~15s CPU)
  without completing. Only recovery: kill the trainer.
- **Who hits it**: `async_save=True` is UNCONDITIONAL
  (server/.../megatron_config.py CheckpointConfig, no knob at tip), and CP>1
  is a shipped golden topology (GLM-5.2-FP8: CP32/EP32 4×8 B200 @256k;
  CP16/EP16 2×8 B300). No completed CP>1 save has ever been observed
  (profiling runs never saved; zero CP>1 save tests exist at tip).
- **Probe matrix** (Qwen3-0.6B, w56lorq): PP2/CP1 PASS · PP1/CP2 HANG ·
  PP2/CP2 HANG → CP-triggered, PP-independent.
- **Workaround CONFIRMED**: sync save path completes under CP2 —
  `BT_SAVE_STATE_SYNC=1` (toggle + probe test on branch
  `jackrao/lps-1062-pp2-export` @ db5d1826; probe green 2026-08-12, 113s wall
  incl. boot, 60MB checkpoint written). Severity: async-only CP bug with a
  proven sync fallback — still blocker-leaning because async is the default
  and the failure mode is a silent wedge.
- **Evidence package** (durable, on the Mac — the box copy may be reclaimed):
  `experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/export_test/evidence/`
  — py-spy stacks (`pyspy_savestate_pp1cp2_*.txt` are the hang; the
  `pp2cp2_rank_*` ones are a half-booted cluster, less useful), both L2b
  adapters + worker logs, `cudart_mask.c` (see F2).
- **Escalation**: paste-able summary in EXPORT_TEST.md §7/F1. Status per
  borel: PARKED, awaiting Jack. Do not re-escalate unless asked.

## Branch map — `jackrao/lps-1062-pp2-export` (pushed, checks green)

- `8a4ae08b` — CP gate relaxations (TEST-SCOPED, deliberate divergence from
  gibbs's ship-grade 21d0c578 DSA-only exemption; at morning-PR gibbs's gate
  wins mainline, mine becomes fixture material): G1 PP>1+CP>1 raise→warn;
  G2 generic-GPT provider pass-through on P2P CP transport. Plus the 3
  integration tests + mb_cluster `attention_backend` kwarg + unit-test re-pins.
- `ef2e8036` — L2a DCP round-trip test (now skipped, see 6d962102).
- `6d962102` — L2a mkdir fix + `skip` with pointer to the CP>1 DCP hang
  (re-enable when the hang is fixed).
- `db5d1826` — `BT_SAVE_STATE_SYNC=1` diagnostic toggle in
  megatron_config.py + `test_qwen3_06b_pp1cp2_sync_save_state_probe`.

Tests live in `server/tests/integration/test_pp_cp_adapter_export.py`.
Box-local-only probes (deleted at cleanup, recreated here if needed):
`test_save_state_probe.py` topology matrix — ask dedekind's notes or
reconstruct from EXPORT_TEST.md §6 if the matrix needs rerunning.

## Your queued task: big-trainer save probe (after gibbs parity legs B+C)

Run ONE bounded save probe on the real GLM-5.2 PP2/CP8/EP8 @131k trainer —
gibbs's runs have never called save_state, and F1's devbox repro needs a
big-model/prod-shaped confirmation. Timing: AFTER parity/perf legs (a wedge
costs a boot). borel owns the window; you or doppler may execute.

Tool: `experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/tools/save_state_probe.py`
(READY — syntax-checked; runs on the leader against the running trainer):

```bash
# trainer RUNNING and idle, on the leader:
python3 save_state_probe.py --job <jobid> --run-id glm52-pp2cp8ep8-bringup \
    --nodes localhost,1 --timeout 600
# exit 0 = save completed (report s); exit 2 = hang, py-spy stacks dumped
# for all 16 ranks (leader + worker node) under --out.
```

If it hangs: stacks go to borel; relaunch with `BT_SAVE_STATE_SYNC=1`
(cherry-pick the megatron_config.py hunk from db5d1826 onto the bring-up
branch) and re-probe to confirm the sync fallback at production scale.
Then stop_trainer.sh — do NOT continue profiling on a wedged server.

## Finding F2 (box-env workaround you will need)

cu12.8-image + cu13-venv boxes (w56lorq): TE fused attention hard-fails
`Multiple libcudart libraries found` for ANY non-DSA model (flashinfer.jit
ctypes-loads system libcudart.so.12 by absolute path at import; the cudnn
dsatopk1 shim probes both sonames and throws). DSA/GLM paths are immune.
Workaround: build the interposer once —
`gcc -shared -fPIC -o /tmp/ldmask/cudart_mask.so <evidence/cudart_mask.c> -ldl`
— then prefix pytest/server launches with
`LD_PRELOAD=/tmp/ldmask/cudart_mask.so`. (Redirect absolute so.12 loads to the
venv cu13 runtime; fail soname probes.) Not needed on cu13-matched prod images.

## Papercuts filed (dedekind)

- `pc_3b35283b6e47` — the F2 libcudart clash (major).
- `pc_0b75fc644616` — mb_cluster health check can bind a ZOMBIE trainer on
  port 8000 (silent cross-test contamination; major).
- `pc_bf667e3d5bd7` — fresh-worktree pre-push typecheck fails confusingly
  without submodules initialized (minor).

## Housekeeping notes

- The shared box clone (`/root/.cache/user_artifacts/trainers_main`) belongs
  to gibbs: leave it on his branch with his uncommitted TF32 patch
  (`chunked_lm_head.py`, env-gated `BT_TF32_LM_HEAD`); if you must switch
  branches, `git stash push` it and `stash pop` on return. His venv carries a
  sequenced cutlass cu13 repair — do NOT re-sync the venv.
- Qwen3-0.6B is in the box HF cache (`team_artifacts/huggingface`).
- py-spy: `uv tool install py-spy` → `/root/.local/bin/py-spy`.
- Notebook: append timestamped entries to `pp2cp8ep8/NOTEBOOK.md`.
