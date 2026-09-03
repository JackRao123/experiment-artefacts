# Results

## Decision

Native HF blockwise FP8 loading is enabled and numerically correct, but TE
blockwise parameters remain opt-in. Tensorwise FP8 remains the B300 131K golden
default because matched full-model testing found blockwise 4.86% slower and
107.56 GB larger at peak reserved memory.

## Integration

The published checkpoint stores E4M3 payloads with arbitrary FP32 inverse
scales on a 128x128 grid. Transformer Engine 2.16 uses a distinct
`Float8BlockwiseQTensor`, and B300 emulates its 2D scaling through MXFP8 with
power-of-two scales. Only 0.087% of the sampled native scales were powers of
two, so copying the bytes and scale metadata directly would silently change
the model.

The implemented path streams each native weight and reconstructs its logical
value in FP32, applies the existing HF-to-Megatron mapping once, then copies
directly into the selected TE parameter format:

```text
HF E4M3 + FP32 128x128 scales -> FP32 logical shard -> TE parameter
```

For `fp8_recipe="blockwise"`, the TE parameter is blockwise FP8. There is no
BF16 or tensorwise FP8 loading intermediate and no preconverted checkpoint.
The FP32 temporary is one streamed weight, not a full-model copy. LoRA,
residual/pipeline activations, custom DSA, and the output head retain their
existing BF16/FP32 behavior.

## Weight Parity

Representative production weight:
`model.layers.0.mlp.gate_proj.weight`, shape `12288x6144`.

| Target | Cosine vs native logical FP32 | Mean abs error | Max abs error |
|---|---:|---:|---:|
| TE blockwise from FP32 reconstruction | 0.9996483 | 0.00024565 | 0.01171875 |
| prior BF16 -> tensorwise path | 0.9996468 | 0.00024682 | not retained |

The blockwise conversion is not bitwise-identical because Blackwell requires
power-of-two scales, but it preserves slightly better correlation than the old
BF16/tensorwise loading path.

## Debug 0d1m

Both arms loaded the same real native-FP8 layer-6 checkpoint reduced to one
MoE layer, at sequence length 131072 and TP1/PP1/CP8/EP8/DP1.

| Metric | Blockwise | Tensorwise | Blockwise change |
|---|---:|---:|---:|
| Warming policy | controls 1-4 | 10 steady controls | |
| Throughput/GPU | 12,640.31 tok/s | 12,718.18 tok/s | -0.61% |
| Mean FB | 1.29617 s | 1.28823 s | +0.62% |
| Peak allocated | 39.555 GB | 38.134 GB | +1.421 GB |
| Peak reserved | 42.075 GB | 40.540 GB | +1.535 GB |

Blockwise completed eight real training steps with finite loss
`13.5194-13.5304` and nonzero finite gradient norms `0.0286-0.0432`.
The trace contains TE block-scale cast/transpose kernels and block-scaled
UE8M0 x E4M3 NVJet GEMMs, proving the selected format executed.

## Full GLM-5.2

Matched final topology and workload: sequence length 131072, one datum,
TP1/PP1/CP8/EP8/ETP1/DP1, eight B300 GPUs, full recompute.

| Metric | Native -> TE blockwise | Native -> TE tensorwise | Blockwise change |
|---|---:|---:|---:|
| Exact trainers SHA | `daedf9ad610c09a5b9ec74d945a125fa7b1a518b` | `fd40df6663df9b0a2e80fbf0dbad7fc7244e1f5e` | |
| Steady controls | 10 | 10 | |
| Throughput/GPU | 1,237.41 tok/s | 1,300.56 tok/s | **-4.86%** |
| Mean FB | 13.24052 s | 12.59761 s | +5.10% |
| MFU | 11.26% | 11.84% | -0.58 points |
| Startup max reserved | 225.50 GB | 118.35 GB | +107.14 GB |
| Peak allocated | 273.91 GB | 166.68 GB | +107.23 GB |
| Peak reserved | 278.02 GB | 170.46 GB | +107.56 GB |
| Device headroom at peak reserved | 9.41 GB | 116.97 GB | -107.56 GB |

The blockwise run completed 17 optimizer steps. Its ten-control loss declined
from 12.2782 to 12.1792; gradient norms stayed finite and nonzero at
0.2712-0.3802. The matched tensorwise run followed the same trajectory and
completed at the final behavioral SHA.

The blockwise runtime trace spans 13.566 s. Dominant kernels are DSA backward
(1.668 s), HybridEP device synchronization (1.534 s), blockwise NVJet GEMMs
(at least 1.03 s across the two leading families), DSA forward/indexer
(0.991 s), and block-scale cast/transpose kernels (0.580 s across leading
E4M3/E5M2 variants). The format is functioning, but its extra parameter
orientation storage and cast work erase the theoretical blockwise GEMM benefit.

## Conclusion

The technical goal is met: GLM-5.2 can train from the native HF blockwise
checkpoint into TE blockwise parameters without a BF16/tensorwise conversion
chain. The experiment also demonstrates why this should not be the default on
B300: it is slower and leaves only 3.27% device-memory headroom. The final
trainers branch retains the correct opt-in path, aligned startup warmup, tests,
and documentation while keeping the validated tensorwise golden configuration.
