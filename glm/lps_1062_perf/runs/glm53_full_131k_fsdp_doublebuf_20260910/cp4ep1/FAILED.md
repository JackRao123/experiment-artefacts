# CP4EP1 with parameter lookahead: capacity failure

Full GLM-5.3, BF16 FSDP with persistent max-pool double buffers, full one-layer
recompute, two 131072-token datums on eight GPUs (DP2, CP4). Startup completed,
including its one-datum warmup through the newly aligned masked DP replica.

The first maximum-length warmup OOMed. No valid control TPS, normal memory
profile, or runtime trace was produced for this case.

Rank0's error: 2.36 GiB allocation requested with 1.16 GiB free; 260.75 GiB
allocated by PyTorch, only 372.34 MiB reserved-but-unused. GPU4 reported the
same requested allocation and live PyTorch footprint. The stack ends at
FP32LMHead.__call__, adding the LoRA logits to the FP32 base logits. This is
capacity exhaustion under these settings, not the large inactive-cache regime
of the earlier dynamic-buffer run.

The standard OOM observer attaches only after memory-profile start, so this
pre-profile failure did not produce an OOM snapshot. Preserve result.log,
result.trainer.log, startup_timings and source-pins.json as the evidence.

A spawned helper survived the failed trainer and retained CUDA resources.
Its command, process group and stdout identified it as belonging to this run.
The run-local stop script now terminates worker/helper process groups scoped
to its exact trainer log, not only the torchrun launcher's group. All GPUs
were verified clear before the next run.

Follow-up: the sibling no-prefetch experiment retains persistent storage but
removes parameter lookahead to test a lower-memory CP4 schedule. CP2 has not
been measured; this failure is not a claim that every possible CP4 strategy
is impossible.
