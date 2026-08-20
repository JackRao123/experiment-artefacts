# LANE HANDOFF — ramanujan (formerly lebesgue): executor-contract shim + overlap A/B + W2 root-causes

Date: 2026-08-14. Status: **COMPLETE, no in-flight work.** For turing's
succession review. Everything below is on disk / pushed unless marked.

## Branch / commit pointers (all pushed to origin)

- **Shim branch: `jackrao/lps-1062-overlap-contract-shim` @ `1cd31535`**
  (trainers). The executor-contract shim + all W2 fixes. Submodule chain:
  megatron-bridge @ `146f2636` → Megatron-LM @ `b37c01f2e` (both on the
  same-named branch on their forks). Commits, newest first:
  - `1cd31535` fix: cast the LoRA-delta hidden to the base dtype under the
    fp32 plan boundary (the W2 leg-1 loss-forward bug; regression test added).
  - `a3da1223` feat: DSA-legal (12,2,2) layout for the cut-down smoke snapshot
    (estate; not exercised by mission-config boots).
  - `d34f76d9` fix: clear moe_shared_expert_overlap whenever the EP-overlap
    flag is on (the exp03 validator wall; ships the campaign's standing patch).
  - `67061721` chore: bump megatron-bridge submodule (the mcore landmine fix).
  - `e13de4d7` style: ruff-format the shim's cast() call sites.
  - `b4e6ce29` test: placate ty in the schedule-plan protocol stub harness.
  - `cd58e629` feat: the combined-1F1B schedule-plan protocol shim (Option B).
- **Campaign branch: `jackrao/lps-1062-pp2cp8ep8` @ `c30afc3e`** — the
  test_trainer_registry payload fix (b8d868ff's omitted wgrad field).

## Lane bottom line

- **The shim contract is PROVEN on hardware** (W2 leg 1): the combined-1F1B
  executor is reachable from our trainer and trains to aggregate precision
  (canary mains in the control band); the PreProcessNode grad-root landmine
  fix (b37c01f2e) is hardware-validated (full fwd+bwd, zero RuntimeError).
- **BLOCKER (route-b, escalated upstream):** the combined-1F1B executor is
  NONDETERMINISTIC on GLM-5.2 DSA/THD — within-boot run-to-run logprob
  divergence 5.098/6.343 vs the 1e-3 bar, aggregate losses tight (rel ~3e-5).
  Uniform ~0.08/token noise floor + recurring top-k-boundary spikes. Not the
  shim; the executor's forward decomposition of DSA. Executor-dependent rungs
  (incl. the dial program's) stay HELD behind the upstream verdict.
- **Merge posture (Jack's morning-queue call):** default-off + documented
  caveat + blocked-on-upstream. The branch is correct and stays.

## Docs on disk

- `runs/overnight_20260813_overlap_campaign/OVERLAP_AB_DESIGN.md` — W2 canary
  + 131k A/B pre-registration (incl. the resolved determinism-probe branch).
- `pp2cp8ep8/results/UPSTREAM_ESCALATION_DSA_EXECUTOR.md` — **THE ESCALATION
  DOC** (evidence, exact reproducer, the DSA-kernel-race prior + kernel
  snapshots, the ask, the ruled-out list).
- `pp2cp8ep8/results/EXECUTOR_CONTRACT_SCOPING.md` — serre's shim spec.
- `runs/.../LAUNCH_TAX_SCOPING.md` — kernel-launch-reduction scoping
  (roadmap, no ≤2d lever).
- `runs/.../SHIM_SMOKE_1NODE.md` + `pp2cp8ep8/tools/build_smoke_snapshot.py` —
  the held 1-node smoke estate (builder proven end-to-end; snapshot discarded,
  regenerable on-box in ~2 min).
- `pp2cp8ep8/tools/parity_divergence_structure.py` + `runs/.../
  PARITY_DIVERGENCE_STRUCTURE_W2.txt` — the divergence-structure analyzer +
  the W2 leg-2 output.
- `pp2cp8ep8/NOTEBOOK.md` — the campaign ledger, current through the
  escalation (my entries timestamped 2026-08-13 19:54 → 2026-08-14 11:4x).
- Probe JSONs: `~/perf_profiles/lps-1062/incoming/parity_overlap-on-r{1,2,3}.json`
  and `runs/.../parity_overlap-{on,off}.json`.

## Not on disk / known loose ends

- The 1-node smoke snapshot itself (discarded by design; regenerable).
- The stale `pp2cp8ep8/tools/build_shim_smoke_snapshot.py` is kernel-locked
  (macOS EPERM artifact from the Mac episode) — delete on reboot; the canonical
  builder is `build_smoke_snapshot.py`.
- W2 leg 3 (the memory ramp) never ran — moot behind the route-(b) blocker.
