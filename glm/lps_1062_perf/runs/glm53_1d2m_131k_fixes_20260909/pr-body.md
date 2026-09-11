## Summary

Experimental GLM-5.3 1d2m fixes and 131072-token measurements, stacked on #1157 for all-rank runtime profiling.

- Route plain-causal CP1 indexer chunks through cuDNN rather than the per-head FP32 fallback (Megatron-Core #76, Bridge #84).
- Use the existing local dispatcher for singleton HybridEP groups, avoiding the cooperative-launch-too-large failure with 256 local experts.
- Add default-off `BT_FREEZE_GC_AFTER_WARMUP=1`: collect dead warmup objects once, freeze surviving long-lived objects, synchronize the one-time cleanup across ranks, and unfreeze at resource teardown. Automatic GC remains enabled for new objects.

## Measurements

HGX B300, LoRA rank/alpha 32, full one-block recompute, native-FP8 expert storage. Three complete warmups, five unprofiled controls per topology, separate memory and all-rank runtime captures. Values are for the three-block proxy, not full-model TPS.

| Topology | GPUs | FB mean ± SD (s) | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| CP8EP8 | 8 | 0.6448 ± 0.0653 | 25,410 | 24.512 |
| CP8EP1 | 8 | 0.6563 ± 0.0143 | 24,964 | 53.609 |
| CP1EP1 | 1 | 3.5149 ± 0.0174 | 37,291 | 136.151 |

Full attention/MLP forward, recompute, backward breakdowns and weighted full-model block estimates are in `docs/performance/glm-131k-experiment.md`.

## Evidence and validation

- Baseline all-rank capture: generation-2 GC paused a rank for 591 ms while peers waited; slow controls recorded 598–636 ms GC pauses. No CUDA allocation/free driver calls accompanied that captured stall.
- Fixed CP1 trace: 128 cuDNN indexer-forward calls, zero old per-head FP32 fallback signature, zero NCCL kernels.
- Numerical causal/key-count probes passed; selected-key overlap 1.0 at 2k/4k, 0.99999994 at 8k.
- 20 additional controls per topology; stable allocated/reserved peaks. Stability TPS/GPU: 26,427 / 25,074 / 37,307 respectively.
- 17 rank-qualified runtime traces and three memory snapshots retrieved locally, hashed and parsed.
- No tests added or modified, per researcher request. Pre-push lint/format/type checks passed. GC policy remains opt-in/default-off pending longer full-model and resource-reload validation; this draft is not a production-rollout approval.

Tested source: `c9a723bf431621ca05580726f4acd01ec618326a`; later commits only add result documentation.
