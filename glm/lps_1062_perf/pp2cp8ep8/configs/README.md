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
| 2c | `trainer_pp2cp8ep8_32k_selective_offload_moe_act_attn_proj.json` | Full phase-1 config at 32k. **BRING-UP ARM ONLY — not a measurement; 32k throughput is NOT a throughput claim and must not be reported as one.** |
| 3a | `trainer_pp2cp8ep8_131k_selective_offload_moe_act.json` | First config that runs at 131k at all — bring-up milestone, expect first-boot problems. |
| 3b | `trainer_pp2cp8ep8_131k_selective_offload_moe_act_attn_proj.json` | Adds attn_proj, legal under core_attn recompute. |
| 5 | `trainer_pp2cp8ep8_131k_rung5_control_full_recompute.json` + `trainer_pp2cp8ep8_131k_rung5_phase1_offload.json` | The headline MATCHED PAIR at d16: identical except recompute+offload, frozen under rung-5 names so later ladder-arm tuning cannot move the headline. Treatment arm intentionally mirrors rung 3c's content. |

**Rung 5 reports a RATIO, not an absolute.** The pair runs on one tree, one
seed, one data order, so TF32, the B/F caches, the wheel version, and the
box cancel — they are on both sides or neither. The absolute
tokens-per-second figures are context only and are NOT comparable to the
historical record band (984–1103 tok/s/GPU): those numbers came from a
different tree and environment, and comparing against them validly would
require porting the backlogged TF32-head PR #995. Do not report a rung-5
absolute number as if it were a record-band number.

Observational bonus (hilbert; not a gate): rung 1's d2 baseline and rung 5's
d16 control together give a microbatch-scaling read on one tree for free —
d2 vs d16 at the same full-recompute config isolates the pipeline-bubble
share of the step (~33% at d2 vs ~6% at d16).

## Blocked-hardware workaround (2026-08-20): single-node PP1 bring-up pair

The org-wide multinode regression (worker pod's bt-interactive-session
configmap never created) has killed every 2-node box. These two configs run
the never-booted offload machinery on ONE 8xB300 node (tj-wlmlkeq):

- `trainer_cp8ep8_32k_pp1_bringup_full_recompute.json` — PP1 full-recompute
  control; the pool-fix tripwire's length-matched reference (the prior
  trial's 30% pinned-allocation cost was measured at 32k).
- `trainer_cp8ep8_32k_pp1_bringup_offload.json` — selective recompute +
  moe_act + attn_proj at TP1/PP1/CP8/EP8/ETP1, 32k.

Both are BRING-UP artifacts for a blocked-hardware workaround, NOT ladder
rungs. Every never-booted piece they exercise (pool fix, NUMA binding,
placement verification, valve telemetry, the valve, the vocabulary change +
relaxed validator, the unpermute hook, the projection hook) is per-rank
machinery that does not depend on pipeline parallelism. Conversely, PP1
means they validate NOTHING about stage asymmetry, the 2-vs-1 in-flight
split, or the tail-layer seam hazard — those need two nodes. No throughput
claim of any kind. Gate #1 (the 131k memory census) remains blocked on two
nodes.

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

## The moe_combine group was dropped (2026-08-20)

There is no combine offload arm. The group was scoped on a census row of
0.20–0.25 GiB per layer per microbatch that was **derived from tensor shapes**,
not measured against the dispatcher and Transformer Engine path this model uses.
TE's permutation source shows our combine runs the mask-map unpermute with no
merging probs, and that variant saves only `row_id_map` and `pad_offsets` for
backward — sub-megabyte index tensors, below `min_offloaded_tensor_size`. The
group would capture essentially nothing, so the arm that added it was
indistinguishable from the arm before it.

The large tensors at that seam (dispatched tokens, expert output, order 1.6 GiB
each per layer per microbatch at 131k) are **transient**: nothing saves them for
backward and they are freed during forward. There is no better hook site. The
expert-activation group is the entire resident MoE story.

Budget consequence: those bytes were attributed by subtraction, so they move
into the resident bucket. Glue goes from 0.44–0.89 to 0.64–1.14 GiB per layer
per microbatch, and the worst-case projected peak on the binding stage goes from
~213 to ~227 GiB against a ~248 GiB effective ceiling. That tightens the margin
and raises the stakes on the rung-1 census.
