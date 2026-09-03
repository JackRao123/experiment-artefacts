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

20260828 14:00 PDT Completed isolated Phase 1A without a custom kernel using
production checkpoint layer 6, routed expert 0. FC1 native gate/up payloads and
scale grids were concatenated along the output dimension; FC2 used the native
down projection directly. The persistent-BF16 reference and native-FP8 plus
temporary-BF16 reconstruction produced bitwise-identical BF16 outputs for all
`M=256,512,1024,2048,4096,8192` cases: 165,150,720 output elements total, all
at 0 ULP with zero absolute error. At balanced `M=4096`, the naive path was
2.75x slower for FC1 and 2.79x slower for FC2. Native payload plus scales saves
49.988% persistent projection storage, but vectorized temporary reconstruction
peaked 288 MiB above baseline for FC1 and 144 MiB for FC2. No trainer source,
custom kernel, model integration, or Phase 2+ path was touched. Full method and
tables are in `PHASE1A.md`; raw measurements are in `phase1a_results.json`.

20260828 15:13 PDT Optimized only the full-weight dequantization step while
retaining the simple BF16-materialization followed by BF16 `torch.mm`
algorithm. Compact-scale broadcasting improved the matched `M=4096` combined
ratio from 2.615x to 2.333x; compiling the same exact expression improved it to
1.122x. A final paired benchmark cycled eight real layer-6 experts to exceed
single-weight cache residency and alternated arm order. The compiled path was
1.124x persistent BF16 at balanced `M=4096` and 1.085x at `M=8192`; it remained
3.788x at `M=256` because full-weight materialization is a fixed cost. All
weights and outputs were bitwise equal, including independent direct checks of
16 native scale blocks. The compiled path's peak allocation consists only of
the BF16 temporary plus output: 80 MiB FC1 and 72 MiB FC2 at `M=4096`. No
trainer code, handwritten custom kernel, backward path, or Phase 2 integration
was changed. See `PHASE1A_DEQUANT_OPTIMIZATION.md` and
`dequant_compiled_cold_sweep.json`.

20260828 19:29 PDT Finalized Phase 2 on the simpler Transformer Engine generic
BF16 fallback. The old PR stack was replaced by draft trainers PR #1222,
Megatron-Bridge PR #54, and Megatron-LM PR #67. The final exact pushed SHAs are
trainers `2b94a5ef5f5706ec133cb6afc837b838f02b2ace`, Bridge
`f5dfc08c1446cdbe8fb9b868ea870f5ea2b131f2`, and MCore
`8f5ac1e4efe051209ec20a69fd53fd6ef19c27bb`. Qualified module names now apply
the routed-expert recipe before parameter construction; frozen quantized expert
initialization runs under no-grad so TE creates only rowwise native E4M3 payload
and FP32 scales. Direct loading preserves checkpoint payloads/scales without
requantization. Startup fails unless all routed weights are rowwise-only native
FP8, LoRA is enabled, and full-layer recompute is selected. Experiment boundary
hooks remain available behind `BT_PHASE2_CAPTURE_DIR`.

20260828 19:29 PDT The exact pushed stack completed one warmup plus five control
forward/backward+optimizer steps on the real native-FP8 0d1m checkpoint at
131072 tokens, TP1/PP1/CP8/EP8/ETP1/DP1. Loss stayed finite at
13.5194-13.5305 and every gradient norm was finite/nonzero at 0.0287-0.0378.
Controls warmed from 11,266 to 12,542 tok/s/GPU and averaged 12,188. The server
reached step 6, stopped through the pinned lifecycle script, and all eight GPUs
returned to zero model allocation. Root `make check`, 18 controller tests, 29
policy/wiring tests, 5 direct-loader tests, 7 MCore recipe tests, and 3 direct
TE BF16 forward/dgrad/LoRA/recompute GPU tests pass. See
`PHASE2_TE_GENERIC.md`, `phase2_te_generic_final_pushed_steady5.json`, and
`phase2_te_generic_final_pushed_trainer.log`.

20260828 21:19 PDT Full GLM-5.2 Phase 4 validation passed on the exact pushed
TE-generic stack at TP1/PP1/CP8/EP8/ETP1/DP1, sequence length 131072, one datum,
and 8xB300 on node 0. The model became healthy with 135.136 GB allocated on
rank 0 after startup. The first full forward/backward control reached 1,020
tok/s/GPU. Five warmed controls averaged 1,120.9 tok/s/GPU with a narrow
1,116.7-1,123.9 range, passing the >1,000 gate. All losses were finite
(12.2777-12.3078 across warmed controls) and all gradient norms were
finite/nonzero (0.3903-0.4727). A warmed rank-0 allocator profile measured
132.973 GB start allocated, 183.669 GB peak allocated, and 187.213 GB final
reserved out of 287.429 GB, leaving 100.216 GB physical headroom. The trainer
completed 11 total optimizer steps and stopped cleanly. See
`PHASE4_FULL_MODEL.md` and `phase4_full_native_te_generic_steady5.json`.

20260828 21:48 PDT Captured a 452 MB rank-0 runtime trace from one warmed full
TE-generic step. The profile covered 15.119 s wall with 14.010 s summed GPU busy
time and 7.34% GPU idle. Ranked GPU classes were NVJet BF16 GEMMs 2.710 s,
HybridEP dispatch/combine/sync 2.467 s, DSA attention/indexer 2.246 s,
FP8-dequant/cast candidates 2.221 s, permutation/concatenation 1.044 s, FP32
output-head GEMMs 0.925 s, and NCCL 0.293 s. The top single kernels were DSA
backward at 1.681 s and HybridEP device sync at 1.620 s. CPU slices showed heavy
dynamic-routing synchronization (`aten::nonzero` 8.417 s and `aten::index`
5.082 s, nested/overlapping). The 78 checkpointed layers averaged 56.3 ms
forward and 123.3 ms backward. See the runtime section in
`PHASE4_FULL_MODEL.md` and `phase4_full_native_te_generic_runtime.pt.trace.json`.

20260828 18:24 PDT Replaced the stale global TE FP8-compute PR stack with a
fresh native-storage/BF16-compute stack from current main: trainers draft PR
#1222, Megatron-Bridge draft PR #54, and Megatron-LM draft PR #67. Closed
superseded trainers PR #1211 and Bridge PR #53. The first pushed checkpoint
adds the explicit `expert_weight_storage=native_fp8` policy, arbitrary-FP32
block-scale TE parameter construction, direct GLM-5.2 routed-expert payload and
scale loading, BF16 execution tests, and fail-closed LoRA/config validation.
Root `make check` passed; B300 tests passed for MCore recipe construction (7),
Bridge native import (5), policy/wiring (26), and TE BF16 forward/dgrad/LoRA
with direct and full-recompute paths (3).

20260828 18:24 PDT Found and corrected a critical integration false positive:
the legacy GLM provider did not pass module names during TransformerLayer
construction, so the per-module quantization recipe initially matched nothing.
Earlier candidate timing/memory runs before this fix were invalid native-FP8
measurements and are retained only as rejected diagnostics. Added top-down
`decoder.layers.N` names, rowwise-only no-grad quantized initialization, and a
startup assertion that every routed grouped-linear parameter is native E4M3 +
FP32 scale storage. With that assertion active, the real 0d1m candidate loaded
native bytes/scales and saved 1.208 GB/rank persistent and peak allocated memory
versus BF16 for one MoE layer, matching the expected approximately 1.125 GiB
routed-weight saving.

20260828 18:24 PDT Completed real Phase 2 one-layer integration with compiled
full grouped BF16 materialization feeding TE's private BF16 grouped-linear
autograd path (`fp8=False`). Five warmed 131K controls averaged 12,630
tok/s/GPU versus 12,674 for matched persistent BF16, a 0.35% regression inside
run variance. The first matched warmup loss was bitwise/effectively identical;
subsequent matched losses differed at roughly 1e-5 with finite nonzero gradient
norms. Captured FC1, FC2, MoE, transformer-block, and sampled logits on all eight
ranks: every head/tail sample was bitwise equal, shapes/dtypes matched, and
full-tensor scalar reductions differed only by small reduction-order noise.
Raw results are `phase2_native_real_steady5.json`,
`phase2_bf16_control_steady5.json`, `phase2_boundary_comparison.json`, and the
paired rank-0 memory snapshots. Current TE autograd retains the grouped BF16
temporary until dgrad; backward rematerialization remains Phase 3.

20260828 18:24 PDT Began a standalone 32-real-expert comparison of three
forward arms without trainer startup: persistent BF16 + TE grouped GEMM,
compiled native-FP8-to-BF16 materialization + the same TE grouped GEMM, and TE
blockwise W8A8 autocast. A two-expert smoke passed and confirmed W8A8 output
drift from activation quantization. Full FC1/FC2 `M=256..8192` sweep is in
progress via `phase2_te_w8a8_comparison.py`.

## Final Result

GLM-5.2 now supports the requested native HF blockwise checkpoint -> FP32
logical mapping -> TE blockwise parameter path without BF16 or tensorwise FP8
loading intermediates. Raw byte reuse is numerically invalid on B300 because
native arbitrary scales do not match TE's power-of-two scale contract. The
correct blockwise path trains, profiles, and fits PP1, but its 4.86% throughput
regression and 9.405 GB memory headroom make it unsuitable as the production
default. Tensorwise remains golden; native-to-blockwise remains available for
further TE/kernel work.
