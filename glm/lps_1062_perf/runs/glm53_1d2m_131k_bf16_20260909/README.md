# GLM-5.3 1d2m: BF16 expert-storage comparison

COMPLETE. See RESULTS.md for the three measurements and quantitative explanation of the topology ordering. Same source c9a723bf431621ca05580726f4acd01ec618326a and 131072-token protocol as ../glm53_1d2m_131k_fixes_20260909; expert_weight_storage changes from native_fp8 to bf16. Same FP8 source checkpoint is dequantized at load, so underlying checkpoint values are comparable. Runtime inventory verifies expert weights are resident BF16 Parameters, not quantized wrappers.

Three complete warmups, five controls, memory and all-rank runtime captures. LoRA rank/alpha32, full one-layer recompute, CP8EP8 / CP8EP1 / CP1EP1, TP1 PP1 ETP1. BT_FREEZE_GC_AFTER_WARMUP=1. GC stays enabled for new objects. Original profile driver and mfu.py remain untouched; no tests changed.

Remote root /root/glm53-131k-bf16-20260909, source /root/glm53-131k-fixes-20260909/trainers, model /root/glm53-1d2m-262k-20260909/model, host tj-32vj99q. Run-local lifecycle copies come from devbox-up. No trainer source changes for this storage experiment.

trace_breakdown.py uses Perfetto GPU-annotation envelopes to assign kernels to attention/MLP forward/recompute scopes across all GPU streams, including unowned TE launches. Kernel sums are not additive wall time; boundaries are inferred. Gaps are gaps in scoped kernels, not hardware SM-idleness counters. Control CUDA-event timings are the authoritative module elapsed measurements; separate profiled steps provide attribution.

comm_breakdown.py identifies collectives using recorded process-group metadata. chunk_sort_probe.py independently checks launch configurations on identical input without modifying the installed TE code. sources/te-common-permutation.py snapshots the missing-token-count autotune key. The new tuner issue is diagnosed but not fixed in this storage comparison. Findings also recorded in trainers PR #1355. All experiment trainers were stopped; the devbox remains available.
