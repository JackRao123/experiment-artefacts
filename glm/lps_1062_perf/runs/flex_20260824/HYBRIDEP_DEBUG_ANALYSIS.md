# HybridEP Flex Debug Analysis

## Scope

This analysis compares the EP=8, 8-GPU, sequence-length 8192, 65,536-token runs in `artifacts/`:

- `debug-ep8-hybridep-sms16`
- `debug-ep8-hybridep-sms20`
- `debug-ep8-hybridep-sms32`
- `debug-ep8-deepep-sms16`
- `debug-ep8-deepep-sms20`
- `debug-ep8-alltoall`
- `debug-ep8-hybridep-sms16-failed` (log-only failed attempt)

Evidence comes from each successful run's `result.json`, `trainer_srun.log`, and all eight `memory.rank*.pickle` snapshots. Runtime trace files were not needed for the requested comparison. All byte values below use decimal GB.

The artifact names are the only serialized evidence for the sms setting; this report assumes `sms16`, `sms20`, and `sms32` identify the communication SM allocation used by the corresponding DeepEP or HybridEP run.

## Executive Summary

- **Correctness passes.** Every successful run completed five steps. HybridEP losses differ from alltoall by at most `1.91e-6`, and grad norms differ by at most `4.51e-9` (17.7 ppm). There are no NaNs, infinities, OOMs, or successful-run exceptions.
- **HybridEP/Flex is fastest without the profiler at sms16 and sms20.** Control forward/backward means are 826.49 ms and 825.34 ms, respectively, or about 31% more throughput than alltoall. sms20 is only 0.14% faster than sms16, which is below the strength of this short experiment.
- **sms32 is worse.** It raises HybridEP control forward/backward mean by 5.37% versus sms20 and raises CV from 0.41% to 3.85%. The extra communication SM allocation likely takes useful capacity from model compute without improving communication enough to compensate.
- **The profiler interacts badly with Flex at higher sms.** HybridEP traced-step overhead rises from 17.0% at sms16 to 71.3% at sms20 and 106.5% at sms32. Control steps do not show this scaling. Traced timings therefore cannot be used to rank these Flex settings.
- **HybridEP has no PyTorch-managed CUDA-memory leak.** All 24 HybridEP rank histories show one expected 130.695 MB increase after the first optimizer step, then exactly flat end-of-step live allocations for the remaining three snapshots. DeepEP and alltoall show the identical one-time increase.
- **HybridEP does not buy speed by using more live memory.** Its mean reconstructed peak live allocation is 27.486 GB, byte-equivalent to DeepEP. HybridEP's mean allocator reserve is about 1.29 GB lower than DeepEP's.
- **Flex adds startup/JIT sensitivity.** Successful HybridEP warmup forward/backward takes 24.9-27.3 seconds versus 5.1-7.5 seconds for the baselines. A preceding sms16 attempt failed because HybridEP runtime metadata preprocessing tried to invoke `/bin/nvcc`, which was absent. That failure is a packaging/bootstrap problem, not an OOM or numerical failure.
- **Recommended setting: sms16.** sms20 ties it on steady-state speed but has much larger profiler interference and optimizer-phase jitter. sms32 is dominated.

## Correctness

All successful runs use the same model, topology, token count, and five-step workload. The table compares each run to the corresponding alltoall step sequence.

| Run | Steps | Maximum absolute loss delta vs alltoall | Maximum absolute grad-norm delta vs alltoall | Result |
|---|---:|---:|---:|---|
| alltoall | 5 | 0 | 0 | Reference |
| DeepEP sms16 | 5 | 1.908e-6 | 2.998e-9 | Pass |
| DeepEP sms20 | 5 | 2.861e-6 | 3.289e-9 | Pass |
| HybridEP sms16 | 5 | 1.908e-6 | 1.979e-9 | Pass |
| HybridEP sms20 | 5 | 9.538e-7 | 3.318e-9 | Pass |
| HybridEP sms32 | 5 | 9.538e-7 | 4.511e-9 | Pass |

The loss deltas occur in multiples near FP32 epsilon at this magnitude and are consistent with changed collective/kernel reduction order. The trajectories remain aligned across all five steps. The largest grad-norm delta is only 17.7 ppm of the alltoall grad norm.

Every completed step reports 65,536 input tokens and 65,528 loss tokens. The fixed eight-token difference is one masked/non-loss token per datum, not an emitted MoE drop count, and no log reports dropped tokens.

`final_status.last_loss` and `final_status.grad_norm` are null in every `result.json`, including alltoall; correctness must therefore be read from the per-window records, not those final-status fields.

## Timing and Variance

Control forward/backward timing is the least contaminated steady-state metric in these artifacts. CV is the population standard deviation divided by the mean over the three control windows.

| Run | Control FB mean (ms) | Control TPS/GPU | FB CV | Control FB range / mean | Warmup FB (s) | Traced FB (ms) | Reported Kineto overhead |
|---|---:|---:|---:|---:|---:|---:|---:|
| alltoall | 1083.94 | 7,557.6 | 16.66% | 38.93% | 7.30 | 1192.39 | 10.0% |
| DeepEP sms16 | 973.01 | 8,419.2 | 5.79% | 13.81% | 7.50 | 916.14 | -5.8% |
| DeepEP sms20 | 852.95 | 9,604.4 | 1.52% | 3.70% | 5.11 | 1099.62 | 28.9% |
| HybridEP sms16 | 826.49 | 9,911.8 | **0.26%** | **0.58%** | 26.58 | 967.20 | 17.0% |
| HybridEP sms20 | **825.34** | **9,925.6** | 0.41% | 1.00% | 24.85 | 1413.82 | 71.3% |
| HybridEP sms32 | 869.66 | 9,419.8 | 3.85% | 9.25% | 27.31 | 1795.41 | 106.5% |

### Steady-state effects

- HybridEP sms16 is 31.15% higher-throughput than alltoall and 17.73% higher-throughput than DeepEP sms16.
- HybridEP sms20 is 31.33% higher-throughput than alltoall and 3.34% higher-throughput than DeepEP sms20.
- DeepEP benefits materially from 20 versus 16 SMs: 14.08% higher throughput. HybridEP does not: sms20 improves over sms16 by only 0.14%.
- HybridEP sms32 is 5.10% lower-throughput than sms20. Its middle control forward/backward is 905.41 ms versus 824.95 and 878.61 ms around it, producing substantially more variance.
- HybridEP sms16 and sms20 are dramatically more stable than alltoall and more stable than either DeepEP setting in this sample.

The likely Flex interpretation is that 16 SMs already saturate the useful communication work. Increasing to 20 produces no meaningful control-path gain, while 32 begins to starve or interfere with model compute. This is supported by both the control slowdown and the rising variance, but the run does not contain isolated communication timing, so it is not a direct proof of the mechanism.

### Profiler interaction

The traced-step result is one sample per run, and every log warns that the profiler has no warmup and may skew results. DeepEP sms16 even reports negative overhead because its traced sample happens to be faster than the three-control mean.

HybridEP's overhead scales sharply with sms while its control timing does not:

- sms16: 17.0%
- sms20: 71.3%
- sms32: 106.5%

This is a Flex-specific measurement effect in this set, not evidence that normal sms20 execution is 71% slower. The higher-SM communication path and Kineto instrumentation appear to contend or synchronize in a way absent from control execution. Use unprofiled control timing to choose sms; use the trace only for qualitative kernel attribution unless the profiler interaction is removed.

### Optimizer timing

Optimizer measurements are noisy and likely absorb asynchronous work or synchronization from the preceding forward/backward:

| Run | Control optimizer mean (ms) | Optimizer CV | Individual controls (ms) |
|---|---:|---:|---|
| alltoall | 26.57 | 64.3% | 11.87, 17.32, 50.52 |
| DeepEP sms16 | 25.31 | 57.7% | 18.98, 11.47, 45.50 |
| DeepEP sms20 | 8.77 | 13.7% | 10.28, 7.33, 8.71 |
| HybridEP sms16 | 10.08 | 23.9% | 8.05, 13.47, 8.72 |
| HybridEP sms20 | 33.03 | 67.2% | 8.37, 28.55, 62.17 |
| HybridEP sms32 | 41.97 | 66.9% | 39.39, 77.60, 8.93 |

The sms20/sms32 HybridEP optimizer spikes are a warning sign for synchronization spillover, but three samples are insufficient to call a deterministic regression. The full server-step CV remains 1.08% for sms20 and 3.80% for sms32, versus 0.90% for sms16.

## CUDA Memory

Peak live allocation was reconstructed from each pickle's allocator action history and agrees with `peak_allocated_bytes` in the logs. End allocation and reserve come from the final segment/block state. Values are mean across eight ranks, with the maximum rank shown separately.

| Run | End live mean (GB) | Peak live mean (GB) | Peak live max (GB) | Reserve mean (GB) | Reserve max (GB) | End inactive cache mean (GB) |
|---|---:|---:|---:|---:|---:|---:|
| alltoall | 8.020 | 27.504 | 27.519 | 30.014 | 30.761 | 21.994 |
| DeepEP sms16 | 8.002 | 27.486 | 27.493 | 31.136 | 32.227 | 23.134 |
| DeepEP sms20 | 8.002 | 27.486 | 27.493 | 31.131 | 32.227 | 23.129 |
| HybridEP sms16 | 8.002 | 27.486 | 27.493 | 29.839 | 30.652 | 21.837 |
| HybridEP sms20 | 8.002 | 27.486 | 27.493 | 29.845 | 30.652 | 21.842 |
| HybridEP sms32 | 8.002 | 27.486 | 27.493 | 29.860 | 30.652 | 21.858 |

### Memory effects

- HybridEP sms16/20/32 have effectively identical live memory. Changing sms does not change model, activation, optimizer, or persistent communication-buffer footprint in these snapshots.
- HybridEP and DeepEP have byte-equivalent peak live allocation. HybridEP's speedup is therefore not purchased with extra live CUDA memory.
- HybridEP reserves about 1.29 GB less allocator memory on average than DeepEP and has about 1.29 GB less inactive cache. Its maximum rank reserve is 1.575 GB lower.
- alltoall has about 18 MB more end live memory and peak live memory than HybridEP/DeepEP, but its mean reserve remains about 175 MB above HybridEP.
- The top tracked allocations at peak are identical across implementations and are dominated by logits projection, linear forward buffers, forward activations, bias/dropout/add, and top-k sorting. No HybridEP-specific allocation dominates peak.
- All runs use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The 21.8-23.1 GB inactive values are cached allocator slack after a 27.5 GB activation peak, not by themselves proof of harmful fragmentation. There is no OOM and total device memory is 287.4 GB.

The snapshots cover PyTorch caching-allocator segments and events. If HybridEP or DeepEP owns CUDA/NVLink/RDMA buffers through an external allocator, those bytes are invisible here. The pickles rule out a retained PyTorch-managed Flex buffer but cannot quantify or exclude a leak in externally managed communication memory.

`result.json.aggregates.peak_gpu_memory_bytes` is not the actual peak live allocation. It matches a current/final reserved-memory reading from status. The allocator histories and log `peak_allocated_bytes`/`peak_reserved_bytes` fields are the reliable sources for peak analysis.

## Leak Analysis

Every memory history contains five end-of-operation snapshot markers. Mean end-live sequences are:

- alltoall: 7.889645, 8.020340, 8.020340, 8.020340, 8.020340 GB
- DeepEP sms16/20: 7.871536, 8.002231, 8.002231, 8.002231, 8.002231 GB
- HybridEP sms16/20/32: 7.871539, 8.002234, 8.002234, 8.002234, 8.002234 GB

Each rank in every run grows by exactly 130.695168 MB between the first and second markers and then remains byte-for-byte flat. The retained 130.695 MB is exactly attributable to lazy optimizer setup on rank 0:

- `_initialize_state`: 87.130 MB
- `_copy_model_grads_to_main_grads`: 43.565 MB

Transient top-k allocations are replaced across the boundary but have zero net effect. There are no `oom` events in any successful pickle and no monotonic per-step growth. **Conclusion: no PyTorch-managed CUDA tensor leak is present in HybridEP, DeepEP, or alltoall over the measured window.** Externally allocated communication memory remains outside the evidence boundary.

The failed HybridEP attempt's `destroy_process_group() was not called` message describes NCCL process-group resources left by abrupt termination. It is not evidence of a CUDA tensor leak in the successful training path.

## Warnings and Failures

### Common to all successful implementations

- Eight per-rank `AccumulateGrad node's stream does not match` warnings. These can add synchronization and interfere with CUDA graph capture, but they are not HybridEP-specific.
- One `External init callback must run in same thread as registerClient` error during profiler startup. It appears in alltoall, both DeepEP runs, and all HybridEP runs; profiling still starts and produces a trace.
- Profiler warnings about no warmup and clearing events at cycle boundaries.
- General startup warnings: Apex fallback, deprecated APIs, `pynvml`, experimental modelopt/transformers support, barrier device selection, trusted remote code, and missing checkpoint keys. None terminate a successful run.

### HybridEP/Flex-specific warnings

- All three successful HybridEP logs emit a Dynamo graph break on every rank in `token_dispatcher.py:setup_metadata` at `int(max_num_tokens_across_ep.item())`.
- The graph break is absent from alltoall and both DeepEP logs. It is a genuine Flex-path difference and prevents that metadata setup from remaining in one compiled graph. It may contribute to startup overhead and profiler sensitivity, though the control steps are fast and stable at sms16/20.

### Failed sms16 attempt

The earlier `debug-ep8-hybridep-sms16-failed` run fails during startup warmup in:

`hybrid_ep_dispatch -> dispatch_with_permute -> metadata_preprocessing`

HybridEP tries to JIT-compile a generated CUDA source with `/bin/nvcc`; `/bin/nvcc` does not exist, so all ranks raise `RuntimeError: Failed to compile the code`. Torch elastic terminates the other ranks, and rank 0 warns that its NCCL process group was not destroyed.

This failure establishes an operational Flex dependency: the image must either contain the expected CUDA compiler path or ship/cache the generated HybridEP kernels. It does not indicate a numerical bug, OOM, or steady-state leak. The subsequent successful sms16 run demonstrates that the training path works once a usable compiled kernel or compiler path is available.

## Conclusions and Recommendation

1. **Use HybridEP sms16 as the default from this set.** It has the best evidence balance: 31.15% throughput gain over alltoall, 17.73% over DeepEP sms16, 0.26% FB CV, correct numerics, no PyTorch-managed leak, and lower allocator reserve than DeepEP.
2. **Treat sms20 as a tie, not a win.** Its 0.14% throughput advantage over sms16 is smaller than the experiment's resolution, while profiler overhead and optimizer-phase jitter are much worse.
3. **Reject sms32 for this workload.** It is 5.10% lower-throughput and materially more variable than sms20, with no correctness or memory benefit.
4. **Do not benchmark Flex from the traced step.** The profiler perturbs HybridEP increasingly as sms rises. Collect more unprofiled controls or profile with a method validated not to contend with HybridEP communication resources.
5. **Keep the HybridEP JIT dependency explicit.** A missing `/bin/nvcc` causes an all-rank startup failure. Prebuilding/caching the generated kernels should remove the JIT component of the 25-27 second first-step penalty and reduce deployment fragility.
6. **Investigate the HybridEP-only `.item()` graph break separately.** It does not block steady-state performance here, but it is the clearest Flex-specific compile warning and a plausible contributor to instrumentation sensitivity.

The main uncertainty is sample count: there are only three unprofiled control windows and one traced window per run. The sms16-versus-sms20 tie needs a longer unprofiled run to resolve, but sms32's regression and the absence of a PyTorch-managed memory leak are clear in the available evidence.
