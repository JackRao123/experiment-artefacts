# GLM-5.3 1d2m 131k fixes — COMPLETE

See RESULTS.md for all three accepted measurements, separate attention/MLP forward/recompute/backward timings, weighted full-model block estimates, stability checks, and artifact links.

Tracking PR: https://github.com/basetenlabs/trainers/pull/1355
Final results are also pushed in the PR documentation. The experiment trainer was stopped after retrieval; the devbox and node-local artifacts remain available.
Dependencies: Megatron-Core #76, Megatron-Bridge #84; profiling comes from trainers #1157.

Accepted data live in cp8ep8/result, cp8ep1/result, cp1ep1/result. Earlier probe/final/fixed/validated/guard-skipped attempts are diagnostics, not headline results. In particular the validated attempt was contaminated by an O(N) diagnostic counter; the accepted result attempts cache that counter once.

Remote root: /root/glm53-131k-fixes-20260909 on tj-32vj99q. Model: /root/glm53-1d2m-262k-20260909/model, staged from real GLM-5.3 layers 0,3,6. Source used for accepted runs: c9a723bf431621ca05580726f4acd01ec618326a. All use BT_FREEZE_GC_AFTER_WARMUP=1; without that flag the experimental GC policy is off.

Lifecycle scripts are run-local copies of devbox-up's generated scripts. NUM_GPUS/CUDA_VISIBLE_DEVICES select 8 GPUs for CP8 and only GPU 0 for CP1. LAYER_TIMING_DIR selects per-attempt timer output.

Reusable tools:
- profile_driver.py: three complete warmups, five controls, memory and all-rank runtime captures. Original tools/profile_driver.py and tools/mfu.py were not edited.
- instrumentation/: CUDA-event attention/MLP timers, backward boundary hooks, constant-time GC observation. Counts are cached only for diagnostics; automatic GC remains enabled.
- indexer_probe.py: numerical experiment comparing the new causal-offset scorer against the reference, not a test-suite file.
- collect.py: copy exact metadata-listed artifacts, verify sizes/SHA256, align raw backend steps to driver counters, assert GC policy active for accepted controls.
- analyze.py: reusable all-rank Perfetto CPU/GPU/GC/allocation API rollups.
- report.py and validate.py: render results and verify protocols, checksums, runtime rank counts, and memory histories.

Extra 20-control runs are stability.json under each topology. No unit/integration test files were added or modified. No indexer fallback signature remains in the accepted CP1 trace. Full-model EP1 memory feasibility is not implied by the block-time extrapolation.
