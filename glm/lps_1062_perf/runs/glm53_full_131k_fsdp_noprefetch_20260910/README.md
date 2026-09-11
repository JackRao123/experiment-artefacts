# Full GLM-5.3: CP4 without parameter lookahead

CP4 COMPLETE BUT MEMORY-MARGINAL; CP2 OOM. See RESULTS.md and cp2ep1/FAILED.md.
Same BF16 weights, LoRA, full one-layer recompute, 131072 tokens and
persistent max-pool buffers as the validated CP8 run. CP4 uses two datums for
two DP replicas. This experiment forces prefetch=False on required FSDP
unshard calls. It does not skip required all-gathers or alter model arithmetic.

Purpose: see whether avoiding next-layer parameter residency makes CP4 fit,
and measure the overlap/throughput tradeoff. The ordinary overlap setting is
forced on internally by MFSDP for parameter sharding, so this run-local hook
controls the actual lookahead argument instead of relying on a config no-op.

An OOM observer is attached before startup forward, even without a normal
memory-profile session. Such a pre-profile OOM snapshot contains live state
but may lack allocation history. No tests or original profiler/MFU tools changed.
