# GLM-5.2 One-GPU Expert Runtime Benchmark

## Conclusion

The custom native-FP8 storage runtime is close to persistent BF16, but it is not
free. Across two fresh processes and 20 controls per runtime, it was 4.87%
slower than persistent BF16 by mean full-recompute forward/backward latency.
The ordinary Transformer Engine generic fallback was 61.29% slower than pooled
BF16 and 53.81% slower than the pooled custom result.

This rejects the narrow claim that custom materialization has identical expert
runtime to persistent BF16. It supports the claim that the custom path avoids
most of the generic fallback penalty.

## Design

- Device: one B300 exposed as `NVIDIA L20D`, compute capability 10.3.
- Node: `b300-1-s58nc356-0011`; node-0 was not used.
- Checkpoint: real GLM-5.2 native-FP8 0d1m checkpoint, layer 0.
- Shape: hidden 6144, expert hidden 2048, 32 local experts, top-k 8.
- Workload: balanced EP8-rank equivalent, 4,096 rows per expert and 131,072 routed rows total, corresponding to 16,384 original tokens per GPU.
- Timed path: grouped FC1, SwiGLU, grouped FC2, full activation recompute, and activation backward.
- Weights are frozen; no weight gradients are computed.
- Each arm ran in a fresh process with one full warm step followed by ten controls.
- Order: BF16, custom, generic, custom repeat.
- Seed: 1062 for weights/input protocol, identical across arms.
- PyTorch 2.11.0+cu130, CUDA 13.0, Transformer Engine 2.16.0.
- Script SHA-256: `0db1ff2d17504eee613bdf7a6b3d304f401316811664f50dfbdf857618174f85`.

The persistent BF16 baseline dequantizes the exact native checkpoint values
once before timing. The custom arm keeps native rowwise E4M3 payloads and FP32
scales, uses a compiled full grouped BF16 materialization on every forward, and
passes those external weights to TE private `_GroupedLinear` with `fp8=False`.
The generic arm keeps the same native storage and invokes ordinary
`GroupedLinear.forward` outside FP8 autocast.

## Results

| Arm | Processes | Controls | Mean ms | Median ms | Min-max ms | Effective original tok/s/GPU | Change vs BF16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16, run 1 | 1 | 10 | 24.2743 | 24.6123 | 22.2688-25.8300 | 674,954 | component |
| BF16, run 2 | 1 | 10 | 23.7985 | 24.2626 | 22.1963-24.8420 | 688,446 | component |
| BF16, pooled | 2 | 20 | 24.0364 | 24.3697 | 22.1963-25.8300 | 681,633 | baseline |
| Custom, run 1 | 1 | 10 | 25.3926 | 25.5793 | 23.8447-25.9762 | 645,227 | +5.64% latency |
| TE generic | 1 | 10 | 38.7690 | 38.7466 | 38.5247-39.0826 | 422,606 | +61.29% latency |
| Custom, run 2 | 1 | 10 | 25.0189 | 25.2785 | 23.7841-25.8490 | 654,864 | +4.09% latency |
| Custom, pooled | 2 | 20 | 25.2058 | 25.5312 | 23.7841-25.9762 | 650,010 | +4.87% latency |

The second custom mean was 1.47% faster than the first, and the second BF16 mean
was 1.96% faster than the first. The robust conclusion is therefore "roughly 5%
slower," not that 4.87% is precise below the 1-2 percentage-point level.

## Every Control

| Control | BF16 run 1 ms | Custom run 1 ms | Generic ms | Custom run 2 ms | BF16 run 2 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.2518 | 24.8749 | 38.6499 | 24.1827 | 22.8698 |
| 2 | 22.2688 | 23.8447 | 38.5247 | 23.7910 | 22.1963 |
| 3 | 23.7840 | 25.0213 | 38.8671 | 23.7841 | 22.8314 |
| 4 | 24.0581 | 25.4939 | 38.8260 | 24.7974 | 22.6940 |
| 5 | 25.8300 | 25.5687 | 38.5587 | 25.0727 | 24.3303 |
| 6 | 24.9178 | 25.5900 | 38.9986 | 25.8195 | 24.4092 |
| 7 | 24.6932 | 25.7479 | 38.7569 | 25.7681 | 24.8294 |
| 8 | 24.7141 | 25.8405 | 38.7364 | 25.8490 | 24.8420 |
| 9 | 24.6540 | 25.9679 | 38.6891 | 25.6406 | 24.7879 |
| 10 | 24.5707 | 25.9762 | 39.0826 | 25.4842 | 24.1949 |

## Memory

| Arm | Model allocated GiB | Initial with input/grad GiB | Peak allocated GiB |
|---|---:|---:|---:|
| Persistent BF16 | 2.2500 | 5.2500 | 10.3750 |
| Custom | 1.1253 | 4.1253 | 11.5003 |
| TE generic | 1.1253 | 4.1253 | 11.5003 |

Native storage saves 1.1247 GiB persistently for the 32 local experts. Both
native arms peak 1.1253 GiB above BF16 because full BF16 materializations coexist
with the native payload while autograd consumes them. This microbenchmark shows
a persistent-memory benefit, not a peak-memory benefit, for one recomputed MoE
layer.

## Correctness

- All three arms produced BF16 output and BF16 activation gradients.
- Output sample statistics and values matched exactly across all four processes.
- Activation-gradient sample statistics and values matched exactly.
- Full-output loss proxy matched exactly: `0.00048052039346657693`.
- Full activation-gradient norm matched exactly: `42545.640625`.
- Output and activation gradients were finite in every arm.
- Native storage remained rowwise `uint8` with FP32 scales before and after backward.
- No columnwise payload or scale copy appeared.
- Native payload/scale version counters did not change, and all expert weight gradients remained `None`.

## Profiler Validation

Separate non-headline runs captured one full-recompute step with
`torch.profiler` and a CUDA allocator history snapshot. Profiling was not
enabled during the timing controls above.

| CUDA kernel class | BF16 ms | Custom ms | Generic ms |
|---|---:|---:|---:|
| Same TE BF16 grouped GEMMs | 21.4690 | 21.6280 | 21.5662 |
| Compiled custom materialization | 0 | 1.6259 | 0 |
| Generic dequantization/casts | 0 | 0 | 16.3138 |
| Common activation work | 4.7297 | 4.7269 | 4.7262 |
| Total summed kernel time | 26.1988 | 27.9809 | 42.6062 |

The grouped GEMM time is effectively unchanged across arms. The custom path
really does materialize all 64 local FC1/FC2 expert weights on each of the two
recomputed forwards: the trace contains 128 fused Triton materialization kernel
calls totaling 1.6259 ms. That is small next to roughly 21.6 ms of grouped GEMM
work, which explains why custom remains close to BF16.

The generic path launches three conversion stages per expert weight. Its
128 FP32 multiply kernels, 128 FP8-to-FP32 copies, and 128 FP32-to-BF16 copies
consume 16.3138 ms before the same grouped GEMMs. This independently explains
the generic arm's large penalty.

The allocator histories also distinguish the implementations:

| Arm | Allocation events | All history events | Profiled peak GiB |
|---|---:|---:|---:|
| Persistent BF16 | 14 | 42 | 11.8750 |
| Custom | 18 | 57 | 13.0003 |
| TE generic | 398 | 1,226 | 13.0003 |

Custom allocates one 1.5 GiB grouped FC1 materialization and one 0.75 GiB
grouped FC2 materialization on each forward. Generic instead records 384
per-expert dequantization allocations across its two forwards. The profiled
peaks are 1.5 GiB above the headline peaks because the profiling script retains
the output and activation gradient while dumping the snapshot; the relative
1.1253 GiB native-vs-BF16 peak delta is unchanged.

## Scope

The effective tokens/s numbers are expert-block microbenchmark rates, not
full-model training throughput. This benchmark intentionally excludes routing,
EP all-to-all, attention, shared experts, LoRA, optimizer work, and communication
overlap. It uses a balanced 4,096-row split per expert rather than a captured
real router distribution. Those exclusions make the storage/runtime comparison
clean, but they prevent translating 674,954 tok/s/GPU into a trainer headline.

## Artifacts

- `one_gpu_expert_runtime_benchmark.py`
- `one_gpu_bf16_32x4096_run1.json`
- `one_gpu_bf16_32x4096_run2.json`
- `one_gpu_custom_32x4096_run1.json`
- `one_gpu_generic_32x4096_run1.json`
- `one_gpu_custom_32x4096_run2.json`
- `one_gpu_expert_runtime_profile.py`
- `one_gpu_profiles/*.pt.trace.json`
- `one_gpu_profiles/*.memory.pickle`
- `one_gpu_profiles/*.profile.json`
- `one_gpu_profiles/analysis.json`
