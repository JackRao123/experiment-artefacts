# Phase 1A Dequantization Optimization

## Scope

The algorithm remains deliberately simple:

```text
1. Materialize the complete BF16 weight from native E4M3 + FP32 block scales.
2. Run the same BF16 torch.mm used by the persistent-BF16 reference.
```

There is no fused dequantization-GEMM kernel and no trainer integration. Only
step 1 changed. GPU-operation durations use CUDA events; CPU wall time and
one-time `torch.compile` cost are excluded.

## Iterations At M=4096

The first three rows use one repeatedly accessed real expert so implementation
changes can be compared in one matched run. The final row cycles eight real
experts and uses paired, alternating measurements to reduce cache and GPU
performance-state bias.

| Iteration | Dequantization | FC1 + FC2 ratio vs persistent BF16 | Exact output |
|---|---|---:|---|
| 0 | `repeat_interleave` expanded FP32 scales | 2.615x | Yes |
| 1 | Eager block view + compact-scale broadcasting | 2.333x | Yes |
| 2 | `torch.compile` over the broadcast expression | 1.122x | Yes |
| 3 | Compiled broadcast, paired eight-expert cycle | 1.124x | Yes |

Iteration 1 removes the full-size expanded scale matrix but still writes an
FP32 weight and FP32 product to global memory. Iteration 2 fuses FP8 conversion,
compact scale lookup, FP32 multiplication, and BF16 rounding into generated GPU
code. It writes only the final full BF16 temporary before `torch.mm`.

At `M=4096`, the final paired eight-expert result is:

| Projection | Persistent BF16 | Compiled dequant + GEMM | Paired ratio | Dequant only |
|---|---:|---:|---:|---:|
| FC1 gate + up | 0.1548 ms | 0.1737 ms | 1.121x | 0.0307 ms |
| FC2 down | 0.0779 ms | 0.0882 ms | 1.133x | 0.0301 ms |
| Combined | 0.2327 ms | 0.2619 ms | 1.124x | 0.0608 ms |

The combined duration is not the arithmetic sum of dequant-only and old GEMM
durations because the freshly written BF16 temporary is immediately reused by
the GEMM through the memory hierarchy.

## Final Eight-Expert Sweep

Each timing repeat cycles experts 0-7 from production layer 6. The BF16 working
sets are 384 MiB for FC1 and 192 MiB for FC2; the corresponding native FP8 plus
scale working sets are approximately 192 MiB and 96 MiB. Each reported ratio is
the median of 11 paired ratios, with old-first and candidate-first order
alternated. Every pair contains 128 operations, or 16 uses of each expert.

| M | Persistent BF16 FC1 + FC2 | Compiled dequant + GEMM | Paired ratio |
|---:|---:|---:|---:|
| 256 | 0.0248 ms | 0.0940 ms | 3.788x |
| 512 | 0.0300 ms | 0.0931 ms | 3.112x |
| 1024 | 0.0554 ms | 0.1011 ms | 1.833x |
| 2048 | 0.1148 ms | 0.1438 ms | 1.255x |
| 4096 | 0.2327 ms | 0.2619 ms | 1.124x |
| 8192 | 0.4343 ms | 0.4698 ms | 1.085x |

The fixed cost of materializing two complete weights dominates at low expert
token counts. At the expected balanced `M=4096` point, it costs 12.4%. At
`M=8192`, it costs 8.5%.

## Correctness

- Compiled BF16 weights are bitwise equal to
  `BF16(FP32(qweight) * FP32(scale))` for both projections.
- Eight independently selected 128x128 blocks per projection were reconstructed
  directly from their scalar scale entries; all 262,144 sampled BF16 values
  matched at 0 ULP.
- All FC1 and FC2 outputs across the six-point sweep are bitwise equal, with
  zero maximum and mean absolute error.

The independent block check avoids validating the broadcast layout solely by
comparing it against the same helper implementation.

## Memory

At `M=4096`, incremental peak allocation is 80 MiB for FC1 and 72 MiB for FC2.
Those figures are exactly the full BF16 temporary plus output sizes. The
compiled path eliminates the original full-size FP32 scale, FP32 weight, and
FP32 product allocations, but it still materializes the complete BF16 weight
as required by this experiment.

## Conclusion

The original roughly 2.6x slowdown was an implementation artifact, not an
intrinsic cost of native FP8 storage. A simple compiled dequantization pass
reduces the expected `M=4096` projection overhead to 12.4% while preserving the
old BF16 numerical contract exactly. This is substantially better, but it does
not yet meet the plan's no-material-forward-throughput-regression target.

Small-M performance remains poor because the entire expert weight is rebuilt
regardless of token count. Removing that fixed cost requires tile-level fusion
with GEMM, which is outside this simple implementation.

## Artifacts

- `phase1a_optimize_dequant.py`: hot-expert iteration comparison.
- `dequant_optimization_m4096.json`: raw iteration 0-2 measurements.
- `phase1a_compiled_cold_sweep.py`: paired eight-expert benchmark.
- `dequant_compiled_cold_m4096_paired.json`: focused stability run.
- `dequant_compiled_cold_sweep.json`: final six-point paired sweep.
