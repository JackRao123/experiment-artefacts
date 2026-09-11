# CP2EP1 without lookahead: capacity failure

Full GLM-5.3, BF16 FSDP, persistent buffers, no parameter lookahead, full
one-layer recompute, four 131072-token datums on eight GPUs (DP4, CP2).
Startup completed; the first maximum-length warmup did not.

Ranks 0 and 2 OOMed requesting a 6-GiB expert down-projection output. GPU0
reported 5.08 GiB free, 256.71 GiB live PyTorch allocations, and 503.65 MiB
reserved-but-unused. The failure is in TE GroupedLinear's output allocation,
before the LM-head stage, so an LM-head-only memory optimization does not
address this particular failure.

The early OOM observer saved memory.rank0.oom.pickle and memory.rank2.oom.pickle
under failure_timings/. They capture live allocator state but not a complete
allocation history, since ordinary memory profiling had not begun.
result.log and result.trainer.log preserve the driver and exact stack traces.
No control TPS or regular runtime profile is claimed for this case.

The owned trainer/helper groups were stopped after failure. No tests or
original profiling/MFU scripts were changed.
