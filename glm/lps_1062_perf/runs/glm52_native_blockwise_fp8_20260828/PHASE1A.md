# Phase 1A: Reference And Naive Dequantization

## Scope

Phase 1A benchmarks the two implementations that require no custom kernel:

1. Old reference: reconstruct the native checkpoint weight once as
   `BF16(FP32(qweight) * scale_inv)` and retain it for `torch.mm`.
2. Naive temporary: retain native E4M3 payload and FP32 scales, reconstruct a
   temporary BF16 weight on every invocation, then call the same `torch.mm`.

The fused target kernel from the full Phase 1 plan is intentionally excluded.
No trainer, model, grouped-GEMM, backward, or LoRA integration was performed.

## Inputs

- Devbox: `tj-w5y89m3`, one B300 (`SM103`) GPU used; all 16 GPUs were idle
  before and after the run.
- PyTorch: `2.11.0+cu130`.
- Native checkpoint: `zai-org/GLM-5.2-FP8`, snapshot
  `ba978f7d347eaf65d22f1a86833408afdb953541`.
- Real routed expert: layer 6, expert 0.
- FC1 gate + up: concatenate two native `[2048, 6144]` E4M3 payloads and two
  `[16, 48]` FP32 scale grids into `[4096, 6144]` and `[32, 48]`.
- FC2 down: native `[6144, 2048]` E4M3 payload and `[48, 16]` FP32 scale grid.
- Activations and outputs: BF16.
- Fixed seed: `1062`.
- Timing: 10 warmups, 50 invocations per repeat, 5 repeats. Tables report the
  median per-invocation GPU-operation duration measured with CUDA events.
- Useful MFU assumes 2,250 dense BF16 TFLOP/s peak for one B300.

## Correctness

Every comparison passed bitwise equality:

| Projection | M values | Compared BF16 elements | Max abs error | Mean abs error | ULP histogram |
|---|---|---:|---:|---:|---|
| FC1 gate + up | 256-8192 | 66,060,288 | 0 | 0 | all at 0 ULP |
| FC2 down | 256-8192 | 99,090,432 | 0 | 0 | all at 0 ULP |

This establishes that the native payload/scale reconstruction and FC1
concatenation reproduce the old-main BF16 operand and forward output exactly.
It does not establish the correctness of a future fused kernel.

## FC1 Gate + Up

Useful FLOPs are `2 * M * 6144 * 4096`.

| M | Old ms | Naive ms | Slowdown | Old TFLOP/s | Naive TFLOP/s | Old MFU | Naive MFU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0127 | 0.2174 | 17.17x | 1,017.6 | 59.3 | 45.23% | 2.63% |
| 512 | 0.0167 | 0.2214 | 13.25x | 1,542.2 | 116.4 | 68.54% | 5.17% |
| 1024 | 0.0317 | 0.2337 | 7.37x | 1,625.3 | 220.6 | 72.24% | 9.80% |
| 2048 | 0.0610 | 0.2630 | 4.31x | 1,689.2 | 391.9 | 75.08% | 17.42% |
| 4096 | 0.1239 | 0.3409 | 2.75x | 1,663.8 | 604.7 | 73.94% | 26.88% |
| 8192 | 0.2472 | 0.4677 | 1.89x | 1,668.1 | 881.6 | 74.14% | 39.18% |

At the balanced `M=4096` point, temporary dequantization adds 0.2170 ms and
makes the projection 2.75x slower than the persistent-BF16 reference.

## FC2 Down

Useful FLOPs are `2 * M * 2048 * 6144`.

| M | Old ms | Naive ms | Slowdown | Old TFLOP/s | Naive TFLOP/s | Old MFU | Naive MFU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0098 | 0.1207 | 12.34x | 658.8 | 53.4 | 29.28% | 2.37% |
| 512 | 0.0100 | 0.1155 | 11.54x | 1,286.8 | 111.5 | 57.19% | 4.96% |
| 1024 | 0.0160 | 0.1201 | 7.50x | 1,607.7 | 214.5 | 71.45% | 9.53% |
| 2048 | 0.0320 | 0.1332 | 4.16x | 1,608.4 | 386.9 | 71.48% | 17.20% |
| 4096 | 0.0619 | 0.1727 | 2.79x | 1,665.8 | 596.9 | 74.03% | 26.53% |
| 8192 | 0.1176 | 0.2483 | 2.11x | 1,753.6 | 830.1 | 77.94% | 36.89% |

At `M=4096`, temporary dequantization adds 0.1108 ms and makes the projection
2.79x slower than the persistent-BF16 reference.

## Memory

Persistent figures below include only the projection weights and scale grids.
They exclude activations, outputs, allocator overhead, and GEMM workspace.

| Projection | Persistent BF16 | Native FP8 + scales | Saving | Naive incremental peak allocation |
|---|---:|---:|---:|---:|
| FC1 gate + up | 48.000 MiB | 24.006 MiB | 23.994 MiB (49.988%) | 288 MiB |
| FC2 down | 24.000 MiB | 12.003 MiB | 11.997 MiB (49.988%) | 144 MiB |
| One expert total | 72.000 MiB | 36.009 MiB | 35.991 MiB (49.988%) | N/A |

The naive peak is much larger than one BF16 weight because the production
vectorized reconstruction creates expanded FP32 scales, an FP32 copy of the
E4M3 payload, and an FP32 product before casting to BF16. It is a correctness
arm and worst-case overhead bound, not a viable storage implementation.

## Conclusion

- Native E4M3 payload plus arbitrary FP32 128x128 scales is sufficient to
  reproduce the old BF16 projection output bit-for-bit.
- Native persistent storage is effectively half the BF16 weight storage.
- Materializing BF16 on every call is decisively too slow across the full sweep,
  including at the balanced `M=4096` point.
- A fused or tiled implementation that avoids global temporary expansion is
  required before this path can compete with persistent BF16.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 /root/.devbox-venvs/server/bin/python \
  /root/.cache/user_artifacts/lps1062_native_blockwise_20260828/phase1a/phase1a_benchmark.py \
  /root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541 \
  /root/.cache/user_artifacts/lps1062_native_blockwise_20260828/phase1a/phase1a_results.json \
  --layer 6 --expert 0 --warmup 10 --iterations 50 --repeats 5
```

Raw measurements, including all repeat durations and allocator counters, are
in `phase1a_results.json`. The exact harness is `phase1a_benchmark.py`.
