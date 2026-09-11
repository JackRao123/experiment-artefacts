# Full GLM-5.3 CP8EP1: diagnostic dynamic-buffer baseline

Functional but pathological. BF16 FSDP, 131072 tokens, one datum on eight
HGX B300 GPUs, LoRA rank/alpha32, full one-layer recompute. Three warmups,
five controls, one memory snapshot and eight Kineto traces. All artifacts are
local under cp8ep1/result; this result is retained, not used as a healthy
topology-performance claim.

| Metric | Result |
|---|---:|
| Unprofiled FB mean ± sample SD | 84.1999 ± 4.5005 s |
| Control FB range | 77.5289–89.2269 s |
| TPS/GPU | 194.58 |
| Peak allocated | 239.132 GiB |
| Peak reserved | 262.795 GiB |

The small proxy predicted roughly 10.7 seconds of full-model block work.
That extrapolation did not survive this change in memory-management regime.

## Evidence for buffer-management overhead

- A live backward stack caught StorageResizeBasedBucketAllocator allocating
  an all-gather buffer through tensor storage resize.
- The memory-profile window has 24 segment_map events totaling 93.346 GiB
  and 35 segment_unmap events totaling 92.193 GiB on rank0. These are mappings,
  not tensor-copy bandwidth. The same FSDP proxy snapshot had zero map/unmap
  events. Logical tensor alloc/free traffic is much larger and mostly reuses
  memory; do not confuse it with physical mapping activity.
- Allocator settings are expandable_segments=True and
  garbage_collection_threshold=0.95. Reserved memory is near the device limit.
  Proactive cache reclamation may contribute; that specific contribution has
  not been separately isolated.
- Maximum recorded Python GC pause across the five controls/all ranks is
  14.22 ms. It does not explain an 80-second step.
- Kineto's allocator-API rollup did not expose cudaMalloc/cuMem calls, so the
  memory snapshot and live stack are necessary evidence. Absence of those
  Kineto events alone is not proof of zero allocator work.

## Rank0 communication in the separate runtime trace

| Recorded process-group operation | Calls | Logical output GiB | Summed GPU kernel seconds |
|---|---:|---:|---:|
| Expert-data-parallel weight all-gather | 150 | 2700.0 | 7.964 |
| DP+CP non-expert parameter all-gather | 159 | 66.607 | 0.813 |
| CP all-gather | 199 | 23.250 | 0.731 |
| CP reduce-scatter | 78 | 1.371 | 2.202 |

These durations overlap computation and include peer waits. They are not
additive exposed communication costs. An 8-byte CP all-reduce takes 5.401
seconds on rank0: that is straggler waiting, not bandwidth transfer of 8 bytes.
There is no expert dispatch/combine at EP1, but FSDP communicates expert weights.

Next run: glm53_full_131k_fsdp_doublebuf_20260910 enables existing persistent
max-pool double buffers and records per-step allocator counters. Startup-only
improvements are reported separately in STARTUP.md and the import probes.
No tests or original profiling/MFU tools were changed.
