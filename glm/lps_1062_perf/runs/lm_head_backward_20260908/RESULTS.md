# GLM-5.2 LM-head backward precision and runtime

## Setup

- Devbox: `tj-q098o4q`
- Hardware: one B300 (reported by the driver as `NVIDIA L20D`)
- PyTorch: 2.11.0, CUDA 13
- Checkpoint: `zai-org/GLM-5.2-FP8`
- Real BF16 `lm_head.weight`: `[154880, 6144]`
- Real BF16 `model.norm.weight`, epsilon `1e-5`
- Fixed synthetic BF16 hidden state, RMS-normalized before projection
- Synthetic rank-32 BF16 LM-head LoRA
- 4,096 tokens
- `grad_logits` calculated from mean cross entropy over FP32 base logits plus
  the BF16 LoRA delta
- Five warmups and twenty timed repetitions

The old timing excludes the one-time 3.545 GiB BF16-to-FP32 weight conversion.

## Results

### Precision versus the old FP32 GEMM

- FP32 intermediate maximum absolute error: `3.92465154e-08`
- FP32 intermediate mean absolute error: `8.16980594e-10`
- FP32 intermediate relative L2 error: `0.00026502099`
- Final BF16 bitwise-identical fraction: `95.405304%`
- Final BF16 maximum absolute error: `2.38418579e-07`
- Final BF16 mean absolute error: `8.19447954e-10`

Relative L2 error is:

```text
||new - old||₂ / ||old||₂
```

The measured value means the aggregate difference magnitude is approximately
`0.0265%` of the old FP32 gradient magnitude.

### End-to-end backward timing

- Old FP32 GEMM: median `232.680 ms`, mean `232.710 ms`, minimum `232.633 ms`
- New split-BF16 GEMMs: median `13.297 ms`, mean `13.182 ms`, minimum `12.552 ms`
- Median speedup: `17.499x`

## Limitations

- Hidden states, labels, and LoRA weights are synthetic.
- The LM-head and RMSNorm weights are real.
- This isolates the base-head hidden-gradient calculation; it is not a complete
  transformer forward/backward or training-trajectory comparison.
