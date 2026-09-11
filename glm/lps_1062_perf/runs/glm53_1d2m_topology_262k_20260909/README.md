# GLM-5.3 1d2m topology benchmark, 262144 tokens

Status: COMPLETE. All three topologies have five controls, memory/runtime captures copied locally, validated timing records, and Perfetto analysis. See RESULTS.md. This supersedes the interrupted 1d3m and 2d2m setups; their files are retained separately.

## Experiment
- Devbox tj-32vj99q, HGX B300.
- Same pinned main as the preceding full-model run: 33d19a3542c3d67553e9dc9d30381d2bdf6db31b.
- Remote source: /root/glm53-main-262k-20260909/trainers (no source edits).
- Remote run: /root/glm53-1d2m-262k-20260909.
- CP8EP8 and CP8EP1: eight GPUs; CP1EP1: one GPU, no hidden DP replicas.
- TP1, PP1, ETP1. One synthetic 262144-token datum, seed 0xB300.
- LoRA rank/alpha 32; full uniform one-layer activation checkpointing.
- Native FP8 expert weight storage, BF16 attention/activations. HybridEP for EP8; existing alltoall dispatcher (local EP1 path) for EP1.
- One warmup, five controls, one memory step, one runtime step, each fb + optimizer.
- Original tools/profile_driver.py and tools/mfu.py are unmodified.

## Three categories
0. Dense, full indexer: original layer 0.
1. MoE, shared top-k indices: original layer 3.
2. MoE, full indexer: original layer 6.

Config: configs/glm53-debug-1d2m-config.json relative to the lps_1062_perf directory.
Both the HF indexer_types list and effective Bridge frequency/offset (2/1) implement full/shared/full.
Weights are real source tensors, remapped without dequantizing FP8 weights.
This is a performance proxy, not a quality-equivalent truncated model.

Full GLM-5.3 category counts from checkpoint config:
- Dense/full 3.
- Dense/shared 0.
- MoE/full 18.
- MoE/shared 57.

Extrapolated block time = 3*layer0 + 57*layer1 + 18*layer2, separately for each timing category.
This excludes embeddings, output head/loss, optimizer, and between-block orchestration.
It predicts work timing, not whether full-model weights/activations fit a topology.
Full/shared cache lifetimes, token routing, and deeper-model activation distributions differ from the proxy.

## Measurement
The CP1 controls expose a severe full-indexer-forward slowdown (~26 seconds per full-indexer block). Source inspection identifies a generic score-chunk path that uses per-head FP32 bmm, whereas the packed-CP path supplies causal offsets to the cuDNN indexer. Runtime-trace verification is recorded in RESULTS.md; the benchmark does not patch this path.

CP8EP1 with HybridEP failed during the 64-token startup warmup: CUDA cooperative launch too large in dispatch_with_permute. The failure log is retained under cp8ep1/failed-hybridep-startup.log. Both EP1 benchmarks use alltoall; comparison against EP8 changes dispatcher implementation as well as topology and is not a pure communication-only ablation.

Run-only sitecustomize instruments Megatron CheckpointFunction and backend forward_backward.
CUDA events bracket entire blocks (attention + MLP), original forward, recomputation, and backward.
No device synchronizations are inserted between blocks. Events resolve once after forward_backward.
Controls are Kineto-unprofiled but contain lightweight CUDA-event timing and disabled-profiler annotations.
TPS includes the instrumentation overhead; no overhead correction is claimed.
backward_including_recompute_ms includes the complete checkpoint backward.
backward_after_recompute_ms spans recompute-end to checkpoint-backward-end.
backward_excluding_recompute_ms subtracts recompute from total, retaining boundary/RNG setup.
Per-rank records are retained. Rank0 and max-rank-per-block summaries are separate; the latter is a conservative proxy, not an exact global critical path.
The server also performs a 64-token startup warmup. Collection aligns the final eight worker calls with the explicit driver protocol; the five controls are protocol steps 2 through 6 (raw worker steps are retained).
Peak allocator memory is the server-reported distributed maximum, not just rank0.
Runtime/memory snapshots use the service's default recording ranks (rank0); they are separate steps.

## Reproduction
Use lifecycle/start_trainer.sh, wait_trainer_health.sh, stop_trainer.sh with explicit topology config, NUM_GPUS, CUDA_VISIBLE_DEVICES and LAYER_TIMING_DIR.
Run profile_driver.py with --label glm53-1d2m-<topology>-262k-c5 --seq-len 262144 --datums 1 --num-gpus <8 or 1> --lora-rank 32 --control-repeats 5 --memory-profile --runtime-profile.
Locally: python3 collect_results.py <topology>.
