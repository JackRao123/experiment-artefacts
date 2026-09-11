## Summary

GLM-5.3 CP/EP and FSDP experiments, implemented in branch code. Draft research
work; production readiness is not claimed. No run-local model patches or
sitecustomize are required to enable these paths.

- Fix the CP1 causal DSA indexer path and singleton expert dispatch/sorting.
- Add opt-in long-lived Python GC freezing after warmup.
- Wire Megatron FSDP into trainer configuration, meta initialization, post-wrap
  sharded Hugging Face import, gradient-buffer handling, and empty-DP alignment.
- Expose persistent FSDP buffers and parameter-prefetch controls.
- Integrate bounded checkpoint-reader caching and GPU load-time dequantization
  through the pinned Bridge source; no global loader monkeypatches.
- Include the optional frozen BF16 grouped-MM implementation in pinned Core.
- Expose the lower-memory FP32 LM-head fusion and sequence chunk size.

Dependency implementations are committed in basetenlabs/Megatron-LM#76 and
basetenlabs/Megatron-Bridge#84 and selected by the submodule pin in this PR.

## Reproduction settings

Check out this branch and initialize its pinned submodules. Build/use the
normal Megatron Bridge trainer environment and devbox-up lifecycle scripts.
Use a GLM-5.3 trainer config with BF16 expert storage, LoRA rank/alpha32,
full uniform one-layer recomputation, TP=PP=EP=ETP=1, seqlen131072 and CP8.

```bash
export BT_EXPERIMENTAL_FSDP=1
export BT_MULTI_ADAPTER_ENABLED=0
export CUDA_DEVICE_MAX_CONNECTIONS=32
export BT_FREEZE_GC_AFTER_WARMUP=1
export BT_FSDP_PERSISTENT_BUFFERS=1
export BT_FSDP_PREFETCH=1
export BT_FSDP_FAST_LOAD=1
export BT_FSDP_GROUPED_MM=0
export BT_MEMORY_EFFICIENT_LM_HEAD=0
export BT_LM_HEAD_SEQ_CHUNK=4096
```

For the lower-memory CP4 recipe, change the config to CP4 and use two
131072-token datums on eight GPUs. Set BT_FSDP_PREFETCH=0,
BT_MEMORY_EFFICIENT_LM_HEAD=1 and BT_LM_HEAD_SEQ_CHUNK=2048.

BT_FSDP_GROUPED_MM=1 selects the separate frozen-expert grouped-MM experiment;
it is not enabled in the standard full-model measurements. Setting
BT_FSDP_PERSISTENT_BUFFERS=0 or BT_FSDP_FAST_LOAD=0 reproduces the respective
buffer/import ablations. All these implementations live in committed code.

## Artifact boundary

Run configs, drivers, traces, analysis scripts and reports belong exclusively
to the separate JackRao123/experiment-artefacts repository. None are tracked by
this trainers PR; the previously added docs/performance report is also removed.
A root ignore rule protects that boundary during normal staging.

## Validation

- Trainer make check (lint, formatting and type checking) passes.
- Changed Core files pass formatting, import sorting and lint checks.
- Changed Bridge files pass pre-commit checks.
- No tests added or modified, as requested.
- Fresh-checkout GPU reproduction passed for three debug variants: standard
  CP8 FSDP, CP8 frozen grouped-MM, and CP4 no-prefetch with the fused/smaller
  head. Each completed three 131072-token warmups, five controls, memory
  capture and all-eight-rank runtime capture, with finite loss and gradients.
- Full 743.6B GLM-5.3 CP8EP1 also completed three 131072-token warmups and five
  controls at approximately 1399 TPS/GPU, reproducing the earlier full-model
  throughput. This integration check did not record another full-model trace.
- Runtime import inspection confirms backend, reader and grouped-MM code all
  come from the clean pinned checkout. No experiment directory is on its
  Python path; only Ubuntu's standard crash-handler sitecustomize is present.

Single-adapter GLM/BF16 experiments only. Save/load, multi-adapter FSDP and
other parallelism/quantization combinations are not claimed validated.
