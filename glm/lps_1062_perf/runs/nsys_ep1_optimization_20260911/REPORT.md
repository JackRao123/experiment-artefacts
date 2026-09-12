# Why FSDP CP8EP1 trails CP8EP8

**Reduced FSDP host overhead and CUDA allocator stalls; EP1 has not been shown
to outperform EP8.** Metadata caching improved the earlier unprofiled EP1
screen by4.3%. A separate, matched single-FB allocator experiment improved
profiled throughput by8.1%. These gains use different contracts and must not
be added or compounded.

At the user's direction, the current workflow is one untimed FB warmup and
one measured/profiled FB, with zero optimizer steps. The older long validations
below had already finished before that instruction. No further benchmark calls
are queued; the trainer was stopped and the supplied devbox remains available.

## Single-FB allocator result

Same V2 source, checkpoint, input hash, configuration, one warmup, no optimizer,
and software CUDA/NVTX capture; only the allocator setting differs.

| Expandable CUDA segments | Measured profiled FB (s) | TPS/GPU |
|---|---:|---:|
| True |12.4625|1314.7|
| False |11.5272|1421.3|

That is8.1% higher profiled TPS in this one-sample comparison, **not an
independently measured unprofiled-training gain or variance estimate**.
Both runs remained at optimizer step0 and had finite loss.

The trace confirms the mechanism:139 `cuMemCreate`,139 `cuMemMap` and34
`cuMemSetAccess` calls became zero. Maximum allocation/mapping-coincident GPU
idle across ranks fell771.7→46.4ms. For example, rank2 previously spent449ms
in one `cuMemSetAccess` call before the next kernel could be launched.
The analyzer had omitted `cuMemSetAccess`; that accounting bug is fixed.
Reprocessing the earlier TE captures corrects the fresh-repeat maximum from
12 ms to 165 ms (initial run: 633 ms). The historical report is corrected; its
TPS, GPU-idle and GEMM timing numbers are unchanged.

The trade-off is higher reserved memory: GPU0's cumulative reserved high-water
mark rose252.27→257.06GiB (+4.78GiB). Allocated peak was not collected in the
no-optimizer protocol. Treat this as a fixed-shape experimental setting:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.95`.
See `allocator-single-comparison.json` for contract checks and exact numbers.

## Completed measurements

Full GLM 5.3, 131072 input tokens, 8 HGX B300, LoRA32, BF16 expert storage,
TE expert kernels, CP8/TP1/PP1/ETP1, full one-layer recompute, persistent
FSDP double buffers and prefetch. Same fixed synthetic input, natural routing
(not forced balance). Three warmups and five initial controls; TPS excludes
the separate optimizer request. No CUDA graphs. No tests added or modified.

| Case | Mean FB ± sample SD (s) | TPS/GPU | Peak allocated GiB | Mean optimizer (s) |
|---|---:|---:|---:|---:|
| Previous EP1 TE repeat |11.837 ±0.192|1384.1|248.10|0.511|
| Previous EP8 TE |10.569 ±0.219|1550.1|213.62|0.162|
| EP1 GC freezing only |11.785 ±0.257|1390.3|248.10|—|
| EP1 metadata cache V1 |11.344 ±0.222|1444.3|248.10|0.205|
| EP8 metadata cache + bucket dedup V2 |10.655 ±0.370|1537.6|213.62|0.112|

Peak memory is the controller's cumulative allocated high-water mark,
MAX-reduced across all ranks, as reported during the initial controls.
EP8 V2's longer validation later raised that high-water mark to 215.48 GiB
and then plateaued; it completed all 55 optimizer steps with finite losses
and gradients. Its last 10 validation steps averaged 10.7070 s
(1530.22 TPS/GPU). High-water statistics alone are not a leak diagnosis.

V1 raises initial-control FB throughput 4.3%, and throughput including optimizer
from 1326.8 to 1418.6 TPS/GPU (+6.9%). GC freezing alone did not establish a gain.

V1 also completed 50 additional unprofiled steps, with finite loss/gradient norms
throughout and the same 248.10 GiB peak. Its last 10 validation steps averaged
10.9647s (1494.25 TPS/GPU). These start at a later optimizer state and are
**not substituted into the five-control A/B**. See `metadata-v1-stats.json`.

## What was found and changed

1. **FSDP had substantial CPU bookkeeping overhead, amplified by EP1's 256
   local experts versus EP8's 32.** Before each gather it recreated per-parameter
   buffer views. At step boundaries it resolved every parameter's dotted module
   path to swap raw and distributed Parameters. V1 caches persistent frozen BF16
   views and static parameter owners, preserving the actual swaps, tied aliases,
   tensor values, kernels and communication. It is opt-in for static module trees.
2. **The initial trace gaps shrink in the expected places.** Rank 0's final
   GPU-idle tail falls 333.86→109.84 ms; V1's new NVTX scope directly places
   98.83 ms of residual idle inside raw→distributed parameter replacement.
   Material idle gaps preceding collective launches fall 762.62→407.30 ms.
   These next-launch associations are not complete dependency-based attribution.
3. **Expert GEMMs are not faster from this change.** Rank 0 expert GPU union
   is 2.120 s before and 2.180 s after; attention-tagged union 4.414→4.428 s.
   V1's category-exclusive expert Tensor Active remains 53.56% on rank 0
   (11487 exclusive samples; 52.5% of expert samples exclusive).
   This supports a host-overhead improvement, not a GEMM-efficiency improvement.
4. **A second redundancy was source-verified.** Readiness/release are bucket
   operations, but the existing hooks invoked them for every parameter in the
   bucket. V2 visits each distinct bucket once, preserving first-use order and
   FP8 postprocessing order. The orchestration recorder replay maps 768 parameters
   sharing 3 buckets to 3 instead of 768 calls, with identical state transitions.
   V2 ran on both full-model layouts. Its incremental TPS benefit over V1 has
   not been isolated under a common measurement contract; do not claim one.

CPU-only experiments load the exact source methods: parameter swapping for 38400
toy expert parameters drops 176.51→41.36 ms (reverse 179.67→41.52 ms), and binding
768 small BF16 parameters drops 2.70→0.12 ms. Identity, tied-alias and pool-slot
reassignment checks pass. These are metadata microbenchmarks, not GPU or TPS
measurements; `microbench-results.json` retains the outputs.

## Why removing expert dispatch does not make EP1 communication-free

EP1 on these 8 GPUs means expert-DP8: each rank gathers expert weights through
FSDP. EP8 means expert-DP1: expert weights remain local, but tokens are dispatched
and combined. EP1 also divides the expert GEMM work among more, smaller matrices.

The earlier statement that weight gathers were "fully hidden" was too strong.
The verified result was that 149/149 comparable prefetches finished before the
preceding block's compute ended—not that the transfers had zero cost. In the
previous EP1 rank 0 trace, expert gathers occupied 4.466 s, with 1.151 s exclusive
of every other classified device category. V1 has 4.190 s and 0.881 s respectively.
Gathers also consume GPU/memory/link resources while overlapping computation.

**Do not add category unions or treat all exposed time as removable time.**
The prior EP8 trace's dispatcher union 2.950 s includes packing/control work, not
only transport; expert union 1.244 s is materially lower than EP1's. With V2,
EP8 still has1.245s of expert GPU work; the final no-expand EP1 trace has2.119s.
The latest layouts used different measurement contracts, so this is operator
timing evidence—not a matched headline-TPS comparison. Routed expert storage
is BF16 in all of these cases; routed-expert dequantization is not the measured
cause of the gap.

Final EP1 rank0 largest **exclusive observed costs**, seconds per traced FB:
attention3.028, expert GEMM1.192, expert-weight gathers0.908, GPU idle0.857,
other compute0.534. These are investigation priorities, not guaranteed
recoverable time. All-rank reports retain the full overlapping budget.

## Reproduction and limitations

- Implementation: trainers PR1355, Bridge PR84 and Core PR76. V1 pins:
  trainers 734081da3 / Bridge 33ccd9cb4 / Core 6851f7895. V2 pins:
  trainers c699f8990 / Bridge 28622cfb7 / Core 963a47acd.
- Enable with `BT_FSDP_CACHE_PARAMETER_METADATA=1` alongside the existing
  experimental FSDP settings. Other backends/defaults remain unchanged.
- Source is committed, not patched by the driver. Original `profile_driver.py`
  and `mfu.py` are untouched. All artifacts remain in the separate artifact repo.
- The initial before/after comparison spans source revisions; V1 adds GC/FSDP
  NVTX ranges and contains the inactive grouped-MM offset fix. Source-change
  caveats are explicit in `ep1-metadata-comparison.md`. An annotated baseline
  config is prepared but has not run.
- All 8 ranks are captured with full runtime/kernel correlation. Nsight warns
  some CUDA/NVTX events may be missing. Correlation is not Python stack coverage.
  Profiles are single-step attribution; use unprofiled controls for throughput.
  The new single-FB results are explicitly profiled measurements without an
  independent unprofiled reference. The old/new protocols are not mixed.
  The old TE-repeat timing capture was 3.06% above its control median; V1 was
  0.84% below. Their traced step delta is not the unprofiled gain.
- Raw `.nsys-rep` and SQLite files are verified on the Mac and excluded from
  Git. Scripts, configs, manifests, readable reports and numeric summaries are
  committed. This is a synthetic systems experiment, not convergence validation.
- `devbox-ep1-te-host-baseline`, `devbox-ep8-te-host-baseline` and
  `devbox-ep8-te-gcfreeze` were only prepared, never benchmarked. Their configs
  are retained as alternatives; no measurements are claimed for them.

`../nsys_workflow_20260911/workflow.py` handles capture/collect/analyze/compare.
`idle_breakdown.py` adds gap and FSDP/GC host-scope diagnostics. `summarize_runs.py`
keeps initial controls and late stability windows separate.
Current fast runner: copy `single_fb.py` beside `capture.py` on the devbox,
then run `python single_fb.py CASE`. It enforces one warmup, one timing trace,
zero optimizer steps, and no hardware-metrics or stability pass.
