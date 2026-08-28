# GLM-5.2 native blockwise FP8 training

## Scope

- Base trainer SHA: `7fe28a0b27cdb82c7bc4fa25b12c8b1e632053a3`
  (PR #1210 tensorwise-FP8 baseline).
- Branch: `jackrao/lps-1062-glm52-native-blockwise-fp8`.
- Local worktree:
  `/Users/jackrao/Documents/trainers-wt-lps1062-glm52-vpp-balanced`.
- Devbox: `tj-w5y89m3`, 2 nodes x 8 B300 GPUs.
- Iteration workload: GLM-5.2 0d1m debug model, sequence length 131072,
  one datum, TP1/PP1/CP8/EP8/ETP1/DP1.
- Precision contract: retain eligible frozen base weights directly in their
  native HF blockwise E4M3 representation; BF16 LoRA adapters and ordinary
  activations; existing BF16 custom DSA and FP32 output head.

## Log

20260828 00:27 PDT Inspected the published checkpoint and installed Transformer
Engine 2.16 formats before implementation. The native checkpoint stores linear
weights as `torch.float8_e4m3fn` plus FP32 `weight_scale_inv` metadata on a
ceil(M/128) x ceil(N/128) grid. The existing 0d1m debug snapshot is BF16 and has
no scales. Transformer Engine has a distinct `Float8BlockwiseQTensor` and
`Float8BlockQuantizer`; its tensorwise `Float8Tensor` carries one inverse scale
and is not byte-compatible with the HF blockwise format. Therefore direct load
must preserve and transform both FP8 payload and the 128x128 scale grid.

20260828 00:27 PDT Created this run and a dedicated implementation branch from
the exact pushed tensorwise-FP8 baseline SHA above. No profiling has run yet.

20260828 00:58 PDT Native scales are not byte-compatible with TE's B300
blockwise parameters: only 0.087% of 4,608 sampled production gate-projection
scales are exact powers of two, while TE 2.16 reports that Blackwell emulates
2D block scaling with MXFP8 and requires power-of-two scales. A raw metadata
copy would silently change values. On the representative production tensor
`model.layers.0.mlp.gate_proj.weight` (`12288x6144`), numerically correct
native FP8+scale reconstruction followed by TE blockwise quantization gives
cosine similarity 0.9996483, mean absolute error 0.00024565, and max absolute
error 0.01171875 versus the native logical FP32 weight. This is marginally
better than the old BF16-to-tensorwise path (cosine 0.9996468, mean absolute
error 0.00024682).

20260828 00:58 PDT Implemented the smallest correct conversion: GLM's bridge
now reconstructs each streamed native blockwise weight in FP32, performs the
existing mapping once, and copies directly into TE's selected parameter
format. The B300 131K golden config now selects TE blockwise FP8 parameters,
so the load path is native blockwise FP8 -> FP32 logical mapping -> TE
blockwise FP8, with no BF16 or tensorwise intermediate. Blockwise startup
warmup rounds the global token count to `128 * CP` so each CP shard is kernel
aligned. LoRA, residual/pipeline activations, custom DSA, and the output head
retain their existing BF16/FP32 behavior.

20260828 00:58 PDT Pushed Megatron-Bridge SHA
`6b5928d4f61c9ffbfcafa46bf64610fb1a046363` and exact trainers candidate SHA
`daedf9ad610c09a5b9ec74d945a125fa7b1a518b`. The Megatron-Bridge mandatory
all-files pre-commit passes. Trainers pre-push `make check` passes lint,
format, and all package typechecks. Focused tests pass: 37 trainer backend/
precision tests, 145 model/controller/registry tests, the benchmark golden
projection, and the native-FP8 FP32-loading regression. The full GLM bridge
test file has four unrelated environment-version failures because this devbox
has Transformers 5.8.1 while those existing tests expect the affected 5.11-
5.12 config shape and a raw-config fixture. No profiling has run before this
pushed trainers SHA.

20260828 01:08 PDT Built a real native-FP8 0d1m checkpoint from production
layer 6: 778 E4M3 payload tensors, 779 FP32 scale tensors, 11 BF16 tensors,
13.688 GB total. The blockwise candidate at exact pushed SHA
`daedf9ad610c09a5b9ec74d945a125fa7b1a518b` completed eight real 131K
training steps. Loss stayed finite at 13.5194-13.5304 and gradient norms were
finite/nonzero at 0.0286-0.0432. Warming controls 1-4 averaged 12,640.31
tok/s/GPU. Runtime and all-rank memory profiles were retained.

20260828 01:28 PDT Matched debug tensorwise controls on the same native
checkpoint averaged 12,718.18 tok/s/GPU, making blockwise 0.61% slower. The
all-rank allocator profiles show blockwise at 39.555 GB peak allocated and
42.075 GB reserved versus tensorwise at 38.134/40.540 GB: +1.421/+1.535 GB.
The blockwise trace proves UE8M0 x E4M3 block-scaled NVJet GEMMs and TE
block-scale cast/transpose kernels executed. One initial tensorwise driver run
completed all windows but could not write its final JSON because the project
cache quota was full; its trace and all-rank snapshots were recovered, and an
independent ten-control JSON was written to node-local scratch.

20260828 01:52 PDT Full GLM-5.2 blockwise completed 17 optimizer steps on one
8xB300 node at the same exact candidate SHA. Ten steady controls averaged
1,237.41 tok/s/GPU with 13.24052 s mean FB. Loss declined from 12.2782 to
12.1792; gradient norms remained finite/nonzero. Peak across all ranks was
273.909 GB allocated and 278.024 GB reserved out of 287.429 GB, leaving only
9.405 GB (3.27%) reserved headroom. The 13.566 s trace is retained and proves
the full 78-layer blockwise path.

20260828 02:09 PDT Exact final-code tensorwise control at pushed SHA
`fd40df6663df9b0a2e80fbf0dbad7fc7244e1f5e` completed the same one-datum
workload. Ten steady controls averaged 1,300.56 tok/s/GPU with 12.59761 s mean
FB. Peak was 166.684 GB allocated and 170.463 GB reserved. Native-to-blockwise
is therefore 4.86% slower while adding 107.23 GB peak allocated and 107.56 GB
reserved. Rejected blockwise as the golden default; retained it as a correct
opt-in path and restored the validated tensorwise golden config.

20260828 02:13 PDT Final behavioral trainers head
`ab882f27f183b7387152ca60d3905bea4cf493cb` and profiled Megatron-Bridge head
`6b5928d4f61c9ffbfcafa46bf64610fb1a046363` were pushed. The final trainers
commit only corrects the documented matched regression from 4.6% to 4.9%; the
behavioral final SHA measured above is its parent. Trainers pre-push `make
check`, 145 model/controller/registry tests, focused benchmark projection, 37
backend/precision tests, and Megatron-Bridge all-files pre-commit pass. Stopped
the trainer through the generated lifecycle script; Slurm is empty and all 16
devbox GPUs are idle.

20260828 09:10 PDT Created clean draft PRs after rebasing the one
Megatron-Bridge change onto current `trainers-main`: Megatron-Bridge PR #53
(`d64ee30fd18e038e6adf0395f068404441c6dcb5`) and stacked trainers PR #1211
(`4333166b5eed867b3778a28c15b50f0b871199fa`). The repin does not change the profiled code. Trainers `make
check` and Megatron-Bridge changed-file pre-commit hooks pass on the clean PR
branches.

## Final Result

GLM-5.2 now supports the requested native HF blockwise checkpoint -> FP32
logical mapping -> TE blockwise parameter path without BF16 or tensorwise FP8
loading intermediates. Raw byte reuse is numerically invalid on B300 because
native arbitrary scales do not match TE's power-of-two scale contract. The
correct blockwise path trains, profiles, and fits PP1, but its 4.86% throughput
regression and 9.405 GB memory headroom make it unsuitable as the production
default. Tensorwise remains golden; native-to-blockwise remains available for
further TE/kernel work.
