IN PROGRESS: opt-in run-local Megatron FSDP, BF16, meta initialization then shard-aware HF import. CP8EP1, 1d2m, 131072 tokens. Single-adapter only; this does not claim multi-adapter swapping or save/load support. Full one-layer recompute and same profiling protocol. No tests changed.

Attempt 1 successfully built the sharded model and imported all 1074 HF
parameters, reaching roughly 8 GiB/GPU before startup failed in the backend
DDP type check. The exported FullyShardedDataParallel symbol is a factory,
so the experiment now uses the concrete V1 type in the accepted-type tuple.
This was an experiment-wiring error, not a failed sharding/import path.

GC observation is now Python-only. Entering torch.profiler from a GC callback
could re-enter FX during tracer construction and print ignored graph-attribute
exceptions. Layer CUDA-event timers remain; startup diagnostics are archived.
The new instrumentation also saves the actual per-expert row-count distributions.
