# PR #1355 clean-branch reproduction

This run uses a fresh recursive checkout of the PR and explicit environment
settings. No run-local sitecustomize, model monkeypatches, or runtime source patches.
Configs, profiling drivers, generated devbox lifecycle scripts, logs and results
belong to this independent experiment repository, not the trainers PR.

Source pins at initial staging: trainers `134851f87`, Bridge `60b1570fb`,
Core `cf81782b2`. All implementation is in those repositories.

Enabled settings: BT_EXPERIMENTAL_FSDP=1, BF16 expert storage, LoRA32,
full one-layer recomputation, persistent FSDP buffers, fast sharded HF import,
CUDA_DEVICE_MAX_CONNECTIONS=32, BT_MULTI_ADAPTER_ENABLED=0,
BT_FREEZE_GC_AFTER_WARMUP=1.

CP8 uses parameter prefetch and 4096-token head chunks. CP4 disables prefetch,
enables BT_MEMORY_EFFICIENT_LM_HEAD=1, and uses BT_LM_HEAD_SEQ_CHUNK=2048.
The optional grouped-MM debug variant sets BT_FSDP_GROUPED_MM=1.

Use operate.py to start, wait, benchmark, and stop. It calls devbox-up-generated
lifecycle scripts; it does not patch model code.

## Validation status

The clean-branch debug CP8 run completed three 131072-token warmups, five
unprofiled controls, and memory/all-rank runtime captures. Forward/backward
mean 0.6765 s, 24217.91 TPS/GPU, peak allocated 62.465 GiB, with finite loss
and gradients. This uses persistent FSDP buffers; it is not the earlier
non-FSDP three-topology benchmark.

The grouped-MM CP8 variant also completed all five controls and both profiles:
0.6495 s, 25225.42 TPS/GPU, 62.340 GiB allocated. Its rank0 trace records eight
_FrozenGroupedMM forward calls and four backward calls, confirming the new
Core implementation is used.

Debug CP4/DP2, no prefetch, fused head and 2048-token chunks also passed:
1.2164 s for two datums, 26937.50 TPS/GPU, 50.554 GiB allocated. Full-model
CP8 also completed all three 131072-token warmups and five controls, at about
1399 TPS/GPU. See full-cp8/result/summary.json for exact measurements. This
full-model integration check did not record another full-model profile;
the earlier captures remain in their original run directories.
These are integration checks, not a controlled
replacement for the earlier topology study.
