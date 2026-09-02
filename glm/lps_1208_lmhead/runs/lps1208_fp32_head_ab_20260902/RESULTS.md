# GLM-5.2 BF16 vs FP32 LM-head boundary test

## Setup

- Hardware: one 8xB300 node (`tj-w6172jq`; GPUs report as NVIDIA L20D)
- Weight: real GLM-5.2 BF16 `lm_head.weight`, shape `[154880, 6144]`
- Trainer shape: TP1, `M=4096`
- Sampler shape: TP8, local weight `[19360, 6144]`, `M=1/8/32/256`
- Inputs: identical synthetic BF16 hidden vectors on all ranks
- FP32 control: true FP32 (`float32_matmul_precision=highest`, TF32 disabled)
- LoRA arm: synthetic rank-32 BF16 LoRA over the real base head

## Results

Trainer `M=4096` versus sampler `M=1`, over 256 identical hidden rows:

| Path | Mean KL | Max KL | Max logit error | Top-1 changes |
|---|---:|---:|---:|---:|
| BF16 base | 1.06e-7 | 7.16e-7 | 3.125e-2 | 0 |
| FP32 base | 4.16e-12 | 9.34e-11 | 2.77e-5 | 0 |
| BF16 + LoRA | 1.12e-7 | 7.39e-7 | 4.6875e-2 | 0 |
| FP32 base + BF16 LoRA | 5.88e-11 | 7.59e-9 | 9.80e-4 | 0 |

For the base head, sampler batches of 32 or 256 were bit-identical to the
trainer-shaped result in both BF16 and FP32. The shape-dependent boundary
appeared at sampler batches 1 and 8. LoRA retained much smaller shape effects
at larger batches because its BF16 GEMMs are still shape-dependent.

## Conclusion

FP32 does measurably reduce the isolated LM-head shape/TP boundary. It reduces
mean base-head KL by roughly 25,000x and the LoRA-arm boundary by roughly
1,900x.

However, the BF16 boundary is already only about `1.1e-7` KL, roughly 70,000x
smaller than the approximately `8e-3` end-to-end KL reported in PR #722. No
top-1 token changed. This experiment therefore does not support attributing the
PR's large end-to-end KL improvement to base LM-head BF16 rounding alone.

The likely interpretation is that FP32 lowers one real numerical floor, while
the reported end-to-end change was dominated by other differences or an
uncontrolled experiment.

## Limitations

- The base head weight is real, but hidden vectors are synthetic.
- The LoRA weights are synthetic, not a trained adapter.
- The sampler side emulates TP8 and sampler-sized GEMMs with PyTorch/cuBLAS; it
  does not execute vLLM's Punica LoRA kernels.
- This isolates the head boundary and does not test upstream transformer,
  routing, or weight-sync differences.
- A full both-BF16 versus both-FP32 trainer/sampler trajectory remains necessary
  before removing the production FP32 path.
