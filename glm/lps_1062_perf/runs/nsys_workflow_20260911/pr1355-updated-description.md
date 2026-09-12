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

## Full-model FSDP CP8 Nsight results — 2026-09-11

**TE + EP8 remains the fastest tested FSDP configuration.** The grouped-MM
offset-upload fix removes a real synchronization pathology, but does not
establish an end-to-end throughput win; keep the grouped path opt-in.

Full GLM-5.3, 8×HGX B300, 131072 input tokens, one fixed synthetic SFT datum,
CP8/TP1/PP1/ETP1, BF16 routed experts, LoRA rank/alpha 32, full uniform
one-layer recompute. Three warmups, then five unprofiled controls per case.
TPS/GPU is `131072 / (8 × mean forward_backward seconds)`; optimizer time
is excluded. These are actual full-model measurements, not extrapolations.

| EP | Expert path | Mean FB ± sample SD (s) | TPS/GPU | Peak allocated GiB |
|---:|---|---:|---:|---:|
| 1 | TE, fresh repeat | 11.837 ± 0.192 | 1384.1 | 248.10 |
| 8 | TE | 10.569 ± 0.219 | 1550.1 | 213.62 |
| 1 | Grouped-MM, synchronous offsets | 11.789 ± 0.079 | 1389.8 | 247.98 |
| 8 | Grouped-MM, synchronous offsets | 11.328 ± 0.297 | 1446.3 | 213.49 |
| 1 | Grouped-MM, asynchronous offsets | 11.874 ± 0.200 | 1379.9 | 247.98 |
| 8 | Grouped-MM, asynchronous offsets | 11.200 ± 0.182 | 1462.9 | 213.49 |

TE EP8 is 12.0% higher TPS than TE EP1. Fixed grouped-MM is effectively tied
with TE on EP1 and 5.6% below TE on EP8. Relative to unfixed grouped-MM,
the fix changes mean TPS by −0.7% / +1.1% on EP1 / EP8, within observed variability.

### Trace findings

- CPU-list-to-CUDA offset construction caused **300 stream synchronizations
  per GPU per FB**. On EP8, those API calls occupied 1.095–1.247 s/rank.
  Pinned CPU staging plus nonblocking H2D reduces the count to **zero on all
  eight ranks in both layouts**. The microbenchmark pre-created offsets and
  omitted this integration cost. API wait residence is not recoverable wall time.
- Rank-0 expert GPU union (forward + recompute + real backward): TE EP1/EP8
  **2.120 / 1.244 s**; fixed grouped EP1/EP8 **2.415 / 1.422 s**.
  Actual CUTLASS grouped kernels ran; this was not a silent TE fallback.
- EP1 has expert-DP size eight and gathers weights. All **149/149 comparable
  steady expert gathers** arrived by the preceding block's compute end on each
  rank. That does not make their resource cost free. EP8 has expert-DP size one
  and no expert-weight gather. Dispatch/combine accounting includes local packing,
  not only network traffic; overlapping category times must not be added.
- The initial EP1 trace's 0.632 s allocator-coincident idle did not recur in
  the fresh repeat (maximum 0.012 s), while its TPS and substantial GPU gaps
  remained. Allocator churn is not the persistent explanation.
- Fastest-case EP8 TE rank-0 exclusive observed costs, ranked: attention
  **3.952 s**, dispatch/combine **2.796 s**, expert GEMM **1.244 s**, other
  compute **0.702 s**, other GEMM **0.556 s**. These are investigation priorities,
  not guaranteed speedups. The report contains all ranks.

### Reproduction and validation notes

- Baselines: trainers `13137ef1a`, Bridge `60b1570f`, Core `cf81782b2`.
  Fixed cases: trainers **`57b9a3ef4`**, Bridge **`a3438227`**, Core **`fe3976282`**.
  The only implementation change between grouped cases is offset staging/upload
  and the dependency pins; GPU cumsum, GEMMs, routing and instrumentation remain unchanged.
- Same Torch 2.11.0+cu130 / TE 2.16.0 environment and built-in nsys 2025.3.1.
  Persistent buffers, prefetch and fast import on; head chunk 4096, memory-efficient
  head off, CUDA graphs off (zero graph launches observed). **Unlike the earlier
  reproduction block above, this sweep leaves `BT_FREEZE_GC_AFTER_WARMUP` unset/off.**
  Allocator: `expandable_segments:True,garbage_collection_threshold:0.95`.
- Separate all-rank CUDA/NVTX timing and 10 kHz GPU-metrics captures. Collection
  is off during controls, although the launcher/NVTX wrappers remain.
  All ranks and 100% kernel/runtime correlation were present; Nsight still emitted
  potential event-loss warnings. Metric passes can perturb timing and are not the TPS source.
- All runs completed with finite losses/gradient norms. Manual GPU checks confirmed
  identical offsets for zero/uneven/EP1/EP8 counts. Trainer `make check` and changed-Core
  checks pass; Bridge all-file read-only checks show pre-existing unrelated lint/format
  issues, which were not changed. **No tests added or modified.** This is a five-control
  systems screen, not convergence or production validation.

[Committed workflow, configurations, manifests and full report](https://github.com/JackRao123/experiment-artefacts/tree/b98f8824d8e8e4e3252387da2396560ec230c8bc/glm/lps_1062_perf/runs/nsys_workflow_20260911).
All raw nsys reports/SQLite exports are on the Mac with SHA256 verification;
large raw/derived trace artifacts are intentionally kept out of Git and out of trainers.
Single-command analysis: `python3 workflow.py analyze CASE/timing.sqlite --benchmark CASE/benchmark.json`.
The supplied devbox remains available; the final trainer process is stopped.
