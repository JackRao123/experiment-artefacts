# GLM native-FP8 selective-MoE recompute

## Contract

- Date: 2026-08-31
- Devbox: `tj-w6172jq`, one 8xB300 node
- Branch: `jackrao/lps-1062-native-fp8-partial-recompute`
- PR: https://github.com/basetenlabs/trainers/pull/1251
- Goal: replace native-FP8's full-layer recompute requirement with bounded selective whole-MoE recompute; then evaluate microbatch interleave.
- Protocol: matched debug-model A/B first, then full GLM-5.3 correctness, throughput, runtime, and memory.

## Log

- Stopped the previous GLM-5.3 trainer and verified all eight GPUs drained.
- Confirmed native FP8 does not numerically require full recompute. The constraint bounds temporary BF16 expert weights retained by TE autograd for dgrad.
- Pushed `88d50a63b`: allow selective `modules=["moe"]` while retaining the full/uniform/1 default and rejecting unsafe modes.
- Local validation: 186 model tests and full pre-push `make check` passed.
- Debug selective-MoE A/B: stabilized 12,632 -> 12,968 tok/s/GPU (+2.7%), numerically matched, but full 78-layer selective-MoE OOMed at 131K.
- Implemented frozen expert BF16 rematerialization in MCore; 14/14 targeted tests pass on all eight B300 ranks, including nested-checkpoint and bit-exact mixed-output dgrad tests.
- Full-model block72 is memory-safe but perf-neutral (~1,195 tok/s/GPU); block64 OOMs. Six direct layers add 51.9 GiB transient memory (~8.7 GiB/layer).
- De-risked combined-1F1B with native FP8 and D2. K76 (two overlapped layers) is perf-neutral; more overlap requires activation offload/recompute and regresses badly. Combined/offload experiment was not pushed.
- Replaced GLM FP32 SIMT LM-head projection with BF16 tensor-core operands and FP32 GEMM output. Frozen-linear dgrad uses BF16x3 emulation and matches the FP32 reference after BF16 cast.
- Final steady result on clean PR stack: 1,224 mean / 1,231 median tok/s/GPU versus 1,183 / 1,189 matched FP32 baseline (+3.5%). Profiled wall improved 14.99 -> 13.60 s; peak transient memory improved 48.2 -> 38.5 GiB/rank.
- Final review fixed E8M0 signed-zero handling, nonstandard NVFP4 fallback, quantized-autocast fallback, TE checkpoint symmetry, and FP32 sequence-parallel reduce-scatter. The 8-rank MCore target set passes 32/32 and the Bridge Kimi suite passes 17/17.
- Added grouped-linear private-API compatibility for both the deployed TE signature and MCore's pinned TE 2.17.1 explicit-splits signature; the deployed-contract B300 suite remains green.
