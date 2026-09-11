# Full GLM-5.3: persistent FSDP buffers

CP8 COMPLETE; CP4 WITH LOOKAHEAD OOM. See RESULTS.md. Same full-model BF16 LoRA workload, 131072 tokens per datum,
TP1/PP1/EP1/ETP1, full one-layer recompute. CP8 D1, then CP4 D2 / CP2 D4 if feasible.

The previous full CP8 run completes but takes roughly 80 seconds per control.
A live backward stack points at StorageResizeBasedBucketAllocator allocating
an all-gather buffer. Peak allocated is about 239 GiB, while reserved memory
is near capacity. Its five controls and separate profiles are retained as a
diagnostic baseline, not silently discarded.

This run explicitly enables existing fsdp_double_buffer and
megatron_fsdp_max_pool_double_buffer. Max-pool buffering handles heterogeneous
dense/MoE/indexer units. Non-unit modules retain the dynamic fallback to avoid
extra persistent allocations. No expert compute kernel change is enabled.

Startup-only changes: allocation-demand-based reclamation, cached root parameter
count (Core e6c86ea3b75674dfff3769c7c92141629864dc0c), bounded safetensors readers,
indexed exact-key lookup, and GPU load-time FP8-to-BF16 conversion. Real-weight
probes showed bitwise equality; loader changes are scoped to the immutable HF
snapshot and restored after import. They do not change runtime expert storage.

Use devbox-up-derived start/wait/stop scripts. The stop copy allows 40 seconds
for torchrun's worker cleanup; verify previous GPU contexts are gone before
launching another full model. No tests or original profiling/MFU tools changed.
