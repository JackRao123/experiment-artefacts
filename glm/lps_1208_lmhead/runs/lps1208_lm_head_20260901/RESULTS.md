# LPS-1208 LM-head benchmark

## Shape

- GPU: one B300 (`NVIDIA L20D`)
- TP1 / CP1
- Sequence: 16,384 tokens
- Hidden size: 6,144
- Vocabulary: 154,880
- Production chunk size: 4,096 (four chunks)
- Weight: `glm52-native-debug-0d1m/lm_head.weight`, BF16
- Runtime: forward + backward, seven steady controls
- Memory: incremental peak allocated above persistent weight/input baseline

## Production LoRA path

These arms use the real `chunked_lm_head.py`, Transformer Engine CE, and a
minimal production-compatible FP32-base + BF16-LoRA-r32 head.

| Method | Runtime | Incremental peak | Result |
|---|---:|---:|---|
| Current | 963.79 ms | 29.542 GiB | Baseline |
| Hoist FP32 base-weight cast | 956.68 ms | 18.907 GiB | Exact loss/gradient match |
| Hoist + checkpoint projection/TE CE | 1,433.82 ms | 12.014 GiB | Exact loss/gradient match |

Hoisting is a strict win: `-10.635 GiB` and `0.74%` faster. Generic activation
checkpointing saves another `6.893 GiB`, but makes the head operation `49.9%`
slower than hoist alone.

## Base classifier kernel comparison

Stock CCE and Liger accept one `hidden @ weight` classifier. They cannot express
the production sum of an FP32 base projection and a separately rounded BF16
LoRA projection, so this comparison removes LoRA and measures the base classifier
fairly.

| Method | Operand regime | Runtime | Incremental peak | Notes |
|---|---|---:|---:|---|
| Full FP32 linear + CE reference | FP32 | 926.39 ms | 31.904 GiB | Exact reference |
| Checkpointed TE CE | FP32 | 1,402.33 ms | 6.189 GiB | Exact, slow |
| CCE exact | BF16 inputs, FP32 dot accumulation | **348.75 ms** | **0.563 GiB** | Best viable stock kernel |
| CCE exact, weighted per-token loss | Same | **348.47 ms** | **0.563 GiB** | No penalty for arbitrary token weights |
| CCE default | BF16 inputs, filtered gradients | 268.24 ms | 0.189 GiB | Gradient filtering changes training gradients materially |
| Liger FP32 | FP32 | 1,003.90 ms | 5.479 GiB | Scalar CE exact; slower than reference |
| Liger BF16 | BF16 logits | 46.26 ms | 0.778 GiB | Fast but reintroduces the old BF16-logit rounding mismatch |

## Precision probe

The untimed probe compares 256 per-token NLLs and an arbitrary weighted
per-token backward against the full FP32 reference.

| Method | NLL max abs error | Hidden-grad max abs error | Assessment |
|---|---:|---:|---|
| CCE exact | 1.20e-4 | 9.77e-4 | Small accumulation-order difference; viable pending trainer/sampler parity |
| CCE default | 1.26e-4 | 9.99e-2 | Reject: approximate gradient filtering |
| Liger FP32 | 1.91e-6 | 3.95e-1 | Reject: `reduction=none` backward cannot correctly apply arbitrary token weights |
| Liger BF16 | 9.22e-3 | 3.96e-1 | Reject: BF16-logit mismatch plus weighted-backward limitation |

Liger precomputes classifier gradients during forward under the assumption that
CE is the terminal scalar loss. That is incompatible with PPO/CISPO/DPO and
other losses that apply arbitrary gradients to returned per-token logprobs.

## Recommendation

Do not ship the generic checkpoint implementation as PR 2. Use CCE exact as the
kernel architecture: it is 4.0x faster than checkpointed TE CE and uses 11x less
incremental memory in the base-head comparison.

Stock CCE is not directly usable in the trainer because it lacks:

- the separate `FP32(base) + FP32(BF16 LoRA delta)` projection contract;
- vocabulary-parallel normalization for TP > 1;
- sparse multi-target and top-k reporting integration.

The production implementation should extend the CCE exact path with a second
low-rank projection. It must compute the BF16 LoRA intermediate and delta with
the same rounding as the sampler, add that delta to the FP32 base accumulator,
and return gradients only for hidden/LoRA A/LoRA B (the frozen base classifier
must not receive a full `[V, H]` gradient). Start with TP1, then add TP-global
max/sum-exp and target ownership for vocab parallelism.

## Artifacts

- `lps1208-standalone-*.json`: production LoRA arms
- `lps1208-fused-*.json`: base-classifier kernel arms
- matching `*.memory.pickle`: allocator snapshots
- benchmark drivers:
  - `../../tools/lps1208_lm_head_benchmark.py`
  - `../../tools/lps1208_fused_ce_benchmark.py`
