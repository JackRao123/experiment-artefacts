# GLM-5.3 tip-of-main runtime + memory profile

## Contract

- Date: 2026-08-31
- Target: GLM-5.3 full model at tip of main (`4f740aa08`), one 8xB300 node
- Shape: sequence length 131072, one datum per step
- Mesh: TP1 / PP1 / CP8 / EP8 / ETP1 / DP1
- LoRA rank/alpha 32, native_fp8 expert storage, hybridep dispatcher, full uniform 1-layer recompute
- Driver: `profile_driver.py` (same protocol as glm52_postmerge_optimization_20260829)
- Capture: 3 control windows + 1 memory-profiled step + 1 runtime-profiled step
- Deliverables: control tok/s/GPU + MFU3x, one `.pt.trace.json`, `memory.rank*.pickle` all ranks, ranked bottlenecks, optimization recommendations

## Why

- GLM-5.3 has no runtime trace in the runs archive (today's glm53 runs were memory-only).
- Tip of main moved past this morning's `ceacc74fc`: adds #1243 (GLM B300 fp8 expert storage configs), #1160/#1161 (weight-sync storage iface), #1127 (router replay PP), #1244 (cuDNN/CUTLASS cuda flavor decouple).

## Log

- 17:53 devbox-up 8 b300 -t launched, job w6172jq (ali cluster)
- 18:08 venv build OK; worktree @ 4f740aa08 (tip of main)
- 18:15 trainer healthy; driver run: 3 controls + memory + runtime profile
- 18:25 SUMMARY: 1189 tok/s/GPU, MFU3x 10.8%, HFU 15.7%, control fb 13.8s; memory fb 14.9s; runtime fb 15.0s; finite losses/grad norms
- 18:28 artifacts pulled: 386 MB rank0 runtime trace, 8x memory pickles, result.json, logs
- 18:35 analysis complete — see RESULTS.md (ranked bottlenecks + levers)
- Devbox LEFT RUNNING for follow-up experiments: `ssh tj-w6172jq`; stop with `truss train stop --remote baseten --job-id w6172jq`
