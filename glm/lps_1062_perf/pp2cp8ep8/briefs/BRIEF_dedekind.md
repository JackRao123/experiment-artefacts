# BRIEF — dedekind: LoRA adapter export under PP>1 + CP>1 (2026-08-10 night)

You are **dedekind**, working for **pauli** (manager, this Mac). Reply/report
via: `~/.agents/scripts/send-message.sh pauli "dedekind: <message>"`.
Report on milestones/blockers; don't go silent >30 min while active.

## Mission

Jack's constraint 3: "Test if PP>1 CP>1 LoRA adapter export works. Do this
with a minimal model — don't spawn the entire GLM." We're bringing up GLM-5.2
PP2/CP8/EP8 @131k tonight (gibbs owns that); your job is to prove — or find
and fix the breakage in — the adapter EXPORT path when both PP>1 and CP>1.
Program goal: `experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/GOAL.md`.

## Your workspace

- Mac worktree: `~/Documents/wt-pp2-export`, branch `jackrao/lps-1062-pp2-export`
  (off origin/main df831501). Jack's `~/Documents/trainers` is read-only
  reference.
- Box: **1×8 B200 on vul**, provisioning now — pauli sends the job id when
  ready (~30-60 min). 8 GPUs = enough for PP2×CP2 (4) or PP2×CP4 (8).
- Artefacts: `experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/export_test/`
  (create it; report `EXPORT_TEST.md` there).

## Mac-side first (while the box provisions)

1. Find the adapter export path in `server/src/trainers_server/` (grep for
   export/adapter/safetensors/peft/save — likely an HTTP endpoint on the
   server or a weight_sync-adjacent path). Understand how LoRA weights are
   GATHERED across parallel ranks for export: PP means different layers'
   adapters live on different ranks; CP shouldn't shard weights (it shards
   sequence) but the export path may still assume rank-0-has-everything or
   world-size/group assumptions that break. Read it and write down your
   failure hypotheses BEFORE running.
2. Pick the minimal model. Note `glm52_dsa.py` has an `(8, 2)` debug layout
   (8-layer GLM, PP2) — find what config/checkpoint exercises it (grep tests,
   debug configs, `num_layers` overrides). If there's a tiny debug GLM
   checkpoint, use it. Otherwise ANY small model the megatron_bridge backend
   supports with PP2+CP2+LoRA works (a small dense llama-class model is fine —
   the export path, not the architecture, is under test). Prefer whatever has
   weights already in the box's HF cache (check when box is up) or is a fast
   download.
3. Write the test plan into `EXPORT_TEST.md`: exact configs (reference
   PP1/CP1 vs test PP2/CP2 — same model, same seed, same 2-3 datums,
   fixed LR), steps (boot → N fwd-bwd+optim → export → compare), and the
   comparison criterion:
   - Level 1 (must pass): export completes, no hang/crash; adapter file
     loads; every expected LoRA tensor present (all layers! PP gather bugs
     drop layer subsets) with correct shapes; values non-degenerate (not
     zeros/NaN).
   - Level 2 (strong): PP2/CP2 adapter ≈ PP1/CP1 adapter after identical
     seeded steps on identical data (small numeric tolerance; if CP changes
     datum sharding/loss reduction the grads may differ slightly — if so,
     fall back to: export(PP2/CP2 model state) == in-memory adapter state
     gathered manually, i.e., export is faithful to what's trained).
4. Also eyeball CP1 special-casing: CP>1 with PP1 export presumably worked
   (CP32 runs exported? unclear — check git history/tests for evidence).
   Note anything CP-conditional in the export path.

## On-box (when pauli sends the job id)

devbox-up leaves lifecycle scripts in `/root/.cache/user_artifacts/.devbox_up/`
(start_trainer.sh / wait_trainer_health.sh / stop_trainer.sh), `env.sh`
sourced on login. Flow: `ssh tj-<job>`; checkout your branch in the shared
`trainers_main` clone; configs on `/root/.cache/user_artifacts/`; export
`BT_TRAINER_CONFIG_PATH` + `BT_TRAINER_SERVER_CONFIG_PATH`; start_trainer.sh
(it uses all provisioned nodes; single node here) + wait_trainer_health.sh in
background — inspect every ~3-min checkpoint.

**Bounded timeouts (Jack's explicit instruction): PP+CP is virgin territory —
5-10 min cap per probe; if logs stop advancing or a collective hangs, kill
(stop_trainer.sh), read all rank logs, diagnose, relaunch. A tiny model boots
in ~1-2 min, so hangs are unambiguous fast.**

Run the plan: reference run, test run, export both, compare, write verdict.
If export is BROKEN: diagnose root cause, implement the fix on your branch,
re-test, and report — that fix is on the critical path for shipping PP2 (the
big config is useless to customers if the adapter can't come out).

## Reporting

Append timestamped entries to `pp2cp8ep8/NOTEBOOK.md`. Message pauli on: plan
done, box up + first boot, verdict (works/broken + why), fix landed. File
papercuts for friction (`papercuts add ... --tag lps1062`).
