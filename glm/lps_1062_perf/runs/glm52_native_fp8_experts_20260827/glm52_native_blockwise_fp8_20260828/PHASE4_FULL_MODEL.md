# Phase 4: Full GLM-5.2 Validation

## Configuration

```text
model: zai-org/GLM-5.2-FP8
sequence length: 131072
datums per step: 1
topology: TP1 / PP1 / CP8 / EP8 / ETP1 / DP1
hardware: 8 x B300 on node 0 of tj-w5y89m3
expert storage: checkpoint-native rowwise E4M3 + FP32 128x128 scales
compute: TE generic BF16 fallback, FP8 autocast disabled
LoRA: rank 32, alpha 32, routed experts excluded
recompute: full layer
```

Exact pushed stack:

- trainers `2b94a5ef5f5706ec133cb6afc837b838f02b2ace`
- Megatron-Bridge `f5dfc08c1446cdbe8fb9b868ea870f5ea2b131f2`
- Megatron-LM `8f5ac1e4efe051209ec20a69fd53fd6ef19c27bb`

## Throughput

The initial fit smoke passed at 1,020 tok/s/GPU. Five subsequent warmed controls
were tightly grouped:

| Control | FB seconds | tok/s/GPU | Loss | Grad norm |
|---:|---:|---:|---:|---:|
| 0 | 14.631 | 1,119.8 | 12.303819 | 0.398763 |
| 1 | 14.672 | 1,116.7 | 12.307822 | 0.390325 |
| 2 | 14.591 | 1,122.9 | 12.298895 | 0.434902 |
| 3 | 14.613 | 1,121.2 | 12.307299 | 0.472733 |
| 4 | 14.578 | 1,123.9 | 12.277666 | 0.407372 |

Mean steady throughput was **1,120.9 tok/s/GPU**, with a 7.2 tok/s/GPU total
range. This passes the plan's >1,000 tok/s/GPU gate. Every loss and gradient norm
was finite, and every gradient norm was nonzero.

## Memory

The warmed rank-0 allocator profile measured:

```text
start allocated: 132.973 GB
peak allocated:  183.669 GB
final reserved:  187.213 GB
device total:    287.429 GB
headroom:        100.216 GB
```

Comparison with the earlier full-model profiles:

| Full-model path | Start allocated | Peak allocated | Reserved |
|---|---:|---:|---:|
| Tensorwise FP8 compute | 115.988 GB | 166.684 GB | 170.463 GB |
| Native FP8 storage, BF16 compute | 132.973 GB | 183.669 GB | 187.213 GB |
| Old TE blockwise FP8 compute | 223.213 GB | 273.909 GB | 278.024 GB |

The native BF16-compute path uses about 17 GB more peak allocation than the
tensorwise W8A8 path, but about 90.2 GB less than the old blockwise path and
retains over 100 GB of physical headroom.

## Lifecycle

The trainer completed 11 total forward/backward+optimizer steps across smoke,
steady, and memory-profile windows. It reached step 11, then stopped through the
pinned lifecycle script. All GPU model allocations drained after teardown.

## Result

The full model fits TP1/PP1/CP8/EP8 on one 8xB300 node, trains at 131K context,
passes the throughput gate, and retains substantial memory headroom. Phase 4 is
accepted for the TE-generic implementation.

## Runtime Profile

One warmed full-model step produced a 452 MB rank-0 Kineto trace. The profiled
forward/backward took 15.1 seconds. Kernel wall time was 15.119 seconds, summed
GPU busy time was 14.010 seconds, and GPU idle time was 7.34%.

| GPU kernel class | Total time | Percent of wall |
|---|---:|---:|
| NVJet BF16 GEMMs | 2.710 s | 17.9% |
| HybridEP dispatch/combine/sync | 2.467 s | 16.3% |
| DSA attention/indexer | 2.246 s | 14.9% |
| FP8 dequant/cast candidate kernels | 2.221 s | 14.7% |
| Permutation/concatenation | 1.044 s | 6.9% |
| FP32 output-head GEMMs | 0.925 s | 6.1% |
| NCCL collectives | 0.293 s | 1.9% |

The dequant/cast class is a pattern-based upper estimate: it groups FP32
multiply and FP8/BF16 copy kernels that can also occur outside expert
dequantization. The strongest direct dequant signal is 5,618
`aten::repeat_interleave` calls (244 ms CPU operator time), plus the dominant
FP32 direct-copy, multiply, and BF16-copy GPU kernels.

The top single GPU kernels were DSA backward (1.681 s), HybridEP device sync
(1.620 s), expert/dense NVJet GEMMs, and the two FP32 output-head GEMMs
(0.925 s combined). CPU-side critical-path slices were dominated by
`cudaStreamSynchronize` (10.263 s inclusive), `aten::nonzero` (8.417 s), and
`aten::index` (5.082 s), primarily around dynamic routing/HybridEP metadata.
These nested durations overlap and must not be summed.

Full recompute behaved as expected: 78 checkpointed layers averaged 56.3 ms
forward and 123.3 ms backward, a 2.19x ratio. NCCL is not a primary bottleneck;
ReduceScatter, AllReduce, and AllGather totaled only 293 ms.

## Artifacts

- `phase4_full_native_te_generic.json`
- `phase4_full_native_te_generic_smoke.json`
- `phase4_full_native_te_generic_steady5.json`
- `phase4_full_native_te_generic_memory_result.json`
- `phase4_full_native_te_generic_memory.rank0.pickle`
- `phase4_full_native_te_generic_trainer.log`
- `phase4_full_native_te_generic_runtime_result.json`
- `phase4_full_native_te_generic_runtime.pt.trace.json`
- `phase4_kernel_categories.sql`
- `phase4_gemm_breakdown.sql`
- `phase4_nccl_breakdown.sql`
- `phase4_operator_breakdown.sql`
- `phase4_per_layer.sql`
