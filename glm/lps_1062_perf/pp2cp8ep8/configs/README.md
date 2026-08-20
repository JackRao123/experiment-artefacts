# GLM-5.2 PP2/CP8/EP8 trainer configs

## DO-NOT-SCHEDULE list

- **`trainer_pp2cp8ep8_131k_selective.json.OOM_DO_NOT_SCHEDULE`** — renamed
  from `trainer_pp2cp8ep8_131k_selective.json` on 2026-08-20. Selective
  recompute with NO offload does not fit at 131k: godel's projection is
  **292 GiB on rank 0 (first stage) against the 247.7 GiB effective ceiling**
  (267.7 cap minus the ~20 GiB cold-pool burst), over by ~45 GiB; rank 8
  lands at ~248 GiB, exactly on the line. Scheduling it burns a boot on a
  guaranteed OOM. Selective recompute and the MoE offload are a package at
  131k — there is no selective-without-offload configuration at that length.
  (Historical refs in ../NOTEBOOK.md and ../E2_REVISED_SPEC.md point at the
  old name; they are journal/spec docs, not runnable tooling.)

## Activation-placement ladder (plan: ../ACTIVATION_PLACEMENT_PLAN.md)

One variable per arm. All parse-checked against TrainerControllerConfig
(trainers `jackrao/lps-1062-actplace` @ 0c794d36).

| Rung | Config | Notes |
|---|---|---|
| 1 | `trainer_pp2cp8ep8_131k.json` | Full-recompute baseline; the census boot. |
| 2a | `trainer_pp2cp8ep8_32k.json` + `trainer_pp2cp8ep8_32k_selective_novpp_plain.json` | 32k full baseline (noise floor) vs 32k selective, no offload. Moved to 32k because selective-only OOMs at 131k (above). |
| 2c | `trainer_pp2cp8ep8_32k_selective_offload_moe_act_combine_attn_proj.json` | Full phase-1 config at 32k. **BRING-UP ARM ONLY — not a measurement; 32k throughput is NOT a throughput claim and must not be reported as one.** |
| 3a | `trainer_pp2cp8ep8_131k_selective_offload_moe_act.json` | First config that runs at 131k at all — bring-up milestone, expect first-boot problems. |
| 3b | `trainer_pp2cp8ep8_131k_selective_offload_moe_act_combine.json` | Adds moe_combine. Boots only once carnot's dispatcher hook + vendored vocabulary hunk land. |
| 3c | `trainer_pp2cp8ep8_131k_selective_offload_moe_act_combine_attn_proj.json` | Adds attn_proj (legal under core_attn recompute; trainers 0c794d36). Same carnot-hunk boot dependency as 3b. |

Boot env for ALL offload rungs: `NVTE_CPU_OFFLOAD_V1=1` must be in the
LAUNCHER environment. NUMA: carnot's allocator binds NUMA-local by default;
`BT_OFFLOAD_NUMA_BIND=off` is the opt-out A/B switch — leave it unset for
real arms.

**THE TIMING TRAP (read before touching that env var):** Transformer Engine
latches `NVTE_CPU_OFFLOAD_V1` at IMPORT time via a module-level
`os.environ.get` (`transformer_engine/pytorch/cpu_offload.py:28`), while the
bridge validator reads it lazily at boot config-validation. Consequences:

- var UNSET or `=0` with offload enabled → the boot FAILS LOUDLY
  (ValueError from `_validate_fine_grained_activation_offloading`). A
  forgotten export cannot silently no-op.
- var set WORKER-SIDE AFTER TE IMPORT (e.g. in trainer code, a late
  sitecustomize, or any shell that starts after the trainer's Python
  imports TE) → the validator passes (it sees the late value) but TE
  already latched `0` and silently runs the V0 path — the
  weights-offloading behavior the validator exists to prevent. The run
  looks like "offload does not help" when offload never ran correctly.

So: `NVTE_CPU_OFFLOAD_V1=1` belongs in the launcher environment of every
offload rung, never in worker-side code. carnot's boot-time engagement probe
logs both the env value and TE's latched value; `env=1 latch=0` is the
signature of this trap — treat it as a failed boot, not a slow one.
