# Full GLM-5.3 without parameter lookahead

CP4 completes, but the original 4096-token FP32 LM-head chunking leaves
allocator pressure. Treat this as a marginal-memory diagnostic, not a clean
steady-state topology comparison. CP2 fails capacity before the LM head.

| CP4EP1, DP2, two 131072-token datums | Result |
|---|---:|
| Five-control FB mean ± sample SD | 26.425 ± 2.240 s |
| TPS/GPU | 1240.06 |
| Peak allocated / reserved | 256.810 / 261.674 GiB |
| Allocation retries across 40 rank-controls | 16 |
| Device allocs / frees across those controls | 481 / 431 |
| Allocator-wide stream synchronizations | 16 |

The rank0 memory-profile window itself mapped only 0.254 GiB and unmapped
nothing. That single snapshot did not represent every control: all eight
ranks had at least one retry in the five unprofiled controls. Per-step,
all-rank allocator counters were necessary to catch this.

The runtime trace still has 150 expert-weight all-gathers (2700 GiB logical
output per rank), 4.134 seconds summed on rank0. CP all-gather and
reduce-scatter sum to 0.087 and 0.569 seconds respectively. Kernel sums include
waits and overlap, and cannot be added as exclusive wall time.

CP2 with four 131072-token datums OOMed requesting a 6-GiB expert-output buffer,
with 256.71 GiB already live. Its live-state OOM snapshots and exact stack are
retained under cp2ep1/failure_timings and cp2ep1/result.trainer.log; no CP2 TPS
or regular runtime profile is reported. See cp2ep1/FAILED.md.

Next: the headmem sibling run reduces transient FP32 LM-head allocations and
uses 2048-token head chunks. This preserves the 131072-token model sequences
and is separately benchmarked. No tests or original profiler/MFU tools changed.
