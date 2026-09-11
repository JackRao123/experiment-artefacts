# GLM-5.3 singleton-EP root cause: 131072-token BF16 1d2m

IN PROGRESS: FSDP and isolated GEMM follow-ups are separate experiments.

Same checkpoint values, LoRA rank/alpha32, full one-block recompute, TP1/PP1/ETP1. Core identity-sort elimination is the only model-code change from the previous BF16 comparison. No tests or original driver/MFU tools changed.

## Whole-request controls

| Topology | Five-control FB seconds ± SD | Five-control TPS/GPU | Additional 20 FB seconds ± SD | Additional 20 TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| cp1ep1 | 3.4245 ± 0.0095 | 38,275 | 3.4399 ± 0.0413 | 38,104 | 142.098 |
| cp8ep1 | 0.6270 ± 0.0343 | 26,130 | 0.5946 ± 0.0205 | 27,556 | 58.261 |
| cp8ep8 | 0.6497 ± 0.0769 | 25,218 | 0.6143 ± 0.0144 | 26,672 | 26.761 |

Do not substitute profiler durations for controls. The five-control and 20-control populations are separate and retained in full. TPS uses HTTP forward/backward time, excluding optimizer, as in all previous runs.

## Timing boundary decomposition (20 controls)

The backend metric includes FB + optimizer. Compare it to HTTP FB + optimizer, not HTTP FB alone. The difference includes serialization, dispatch, return values, polling, and instrumentation outside the backend timer; it is not a direct measurement of network time.

| Topology | HTTP FB+optim ms | Backend FB+optim ms | Outside backend ms |
|---|---:|---:|---:|
| cp1ep1 | 3491.06 | 3338.16 | 152.90 |
| cp8ep1 | 623.67 | 494.78 | 128.88 |
| cp8ep8 | 641.25 | 513.45 | 127.80 |

## Attention / MLP (20 controls, rank0 mean milliseconds)

| Topology | Layer type | Fwd attn | Fwd MLP | RC attn | RC MLP | Bwd attn interval | Bwd MLP interval |
|---|---|---:|---:|---:|---:|---:|---:|
| cp1ep1 | Dense/indexer | 240.94 | 48.83 | 110.06 | 47.74 | 292.01 | 65.16 |
| cp1ep1 | MoE/shared | 110.96 | 76.95 | 100.73 | 76.10 | 296.04 | 96.36 |
| cp1ep1 | MoE/indexer | 259.91 | 85.55 | 113.70 | 76.61 | 299.15 | 99.06 |
| cp8ep1 | Dense/indexer | 41.48 | 5.78 | 14.96 | 5.81 | 35.52 | 7.40 |
| cp8ep1 | MoE/shared | 14.04 | 15.56 | 12.61 | 14.02 | 38.18 | 14.83 |
| cp8ep1 | MoE/indexer | 32.88 | 17.53 | 16.82 | 13.17 | 39.13 | 18.28 |
| cp8ep8 | Dense/indexer | 42.48 | 5.78 | 15.17 | 5.83 | 35.05 | 7.43 |
| cp8ep8 | MoE/shared | 14.11 | 21.52 | 12.55 | 21.58 | 35.29 | 22.57 |
| cp8ep8 | MoE/indexer | 31.64 | 18.67 | 16.92 | 20.54 | 35.77 | 18.70 |

Backward intervals partition the gradient dependency path and include waits. They are not exclusive ownership of backward kernels. accounting.json retains all-rank distributions, not just rank0.

## Evidence

- Identity sorts are unnecessary only when EP=ETP=1. The ordinary token permutation remains. identity-sort-probe.log verifies bitwise output/probability/gradient equivalence. New EP1 traces must contain zero _sort_chunks_by_map_kernel launches.
- Runtime traces and memory snapshots live under each topology/result/runtime and topology/result/memory. Artifact sizes and SHA256 digests are in result/summary.json.
- The old BF16 EP8 trace has approximately 20.5 ms summed dispatch/combine kernel duration for a complete step. This is a few percent of a ~0.6-second request, not most of its time. Kernel overlap and peer waits mean the sum is not an exact hypothetical wall-time saving.
- EP1 still performs routing, local permutation and expert GEMMs. It has 256 resident experts per GPU instead of 32. At balanced routing, CP8EP1 has 512 rows/expert versus CP8EP8 4096. Isolated probes distinguish shape efficiency from communication.
- No hardware SM/tensor-core counters have been collected in these Kineto runs; occupancy or kernel duration alone is not a saturation measurement.

## Why CP1 has higher TPS/GPU

Normalize the CP1 run by dividing its latency by eight: that is the same amount of token work per GPU as CP8. This is an efficiency comparison, not the actual request latency (CP1 is still much slower to finish one request).

On matching FB+optimizer boundaries, normalized HTTP time is 436.38 ms for CP1 versus 623.67 ms for CP8. The gap is 187.28 ms; 109.77 ms (58.6%) lies outside the backend timer.

| Normalized component, ms | CP1EP1 / 8 | CP8EP1 | CP8 penalty |
|---|---:|---:|---:|
| forward_attention_ms | 76.48 | 88.40 | +11.92 |
| forward_mlp_ms | 26.42 | 38.87 | +12.46 |
| recompute_attention_ms | 40.56 | 44.40 | +3.84 |
| recompute_mlp_ms | 25.06 | 32.99 | +7.93 |
| attention_backward_ms | 110.90 | 112.82 | +1.92 |
| mlp_backward_ms | 32.57 | 40.52 | +7.94 |
| Backend remainder, including optimizer | 105.28 | 136.78 | +31.50 |

The MLP penalty is consistent with more local experts, smaller GEMMs, repeated weight reads, and launch/scheduling overhead. The isolated GEMM probe quantifies only the base-GEMM shape effect; it does not measure hardware saturation or the actual unbalanced routing distribution.

The proxy has just three blocks but pays a complete LM head/loss and request-processing cost. Extrapolate block categories separately; do not extrapolate its raw TPS to a 78-block model.
