# Full CP4EP1: reduce transient LM-head memory

COMPLETE; see RESULTS.md. Same 131072-token sequences, two datums on eight GPUs, BF16 FSDP,
persistent buffers, no parameter lookahead, LoRA32 and full one-layer recompute.

Changes: opt-in BT_MEMORY_EFFICIENT_LM_HEAD=1 fuses the FP32 gradient split
and LoRA-logit conversion/addition; head chunks are 2048 instead of 4096 tokens.
FP32 logits are retained. The fused split explicitly emulates precision casts:
default compiler widening otherwise destroys the required BF16/FP16 residual.
Real-size synthetic head probes passed bitwise output/gradient equality and
reduced incremental peak from 8.33 to 5.91 GiB for a 4096-token chunk. The
smaller chunk adds headroom without changing the model sequence length.

Purpose: remove allocator retries that remained on some CP4 ranks, despite
rank0's previous memory snapshot showing no unmaps. All-rank counters and
separate memory/runtime profiles are required before calling this healthy.
No tests or original profiling/MFU tools are changed.
