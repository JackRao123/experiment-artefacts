# PCIe D2H/H2D bandwidth benchmark — activation-offload feasibility (2026-08-15)

**Box:** tj-q9v0p03 (1×1 B300, ali, project jrao123-ali). GPU reports as
"NVIDIA L20D", cc 10.3 (sm_103), 148 SMs. Host: 2× Xeon 6767P (Granite
Rapids, 256 threads), 4 TB RAM, 2 NUMA nodes; GPU on node 1.
**Tool:** `bw_bench.cu` (this folder's sibling; also `/root/bw_bench.cu` on
box). cudaMemcpyAsync on dedicated streams, 4 GiB buffers, 32 GiB moved per
measurement. Compiled `-gencode arch=compute_100,code=compute_100` (nvcc
12.8 rejects sm_103 — papercut pc_ba0ec9738fd2).

## Results

| measurement | GB/s |
|---|---:|
| D2H pinned, sustained (chunk-size-flat 64MB→4GB) | **57.3** |
| H2D pinned, sustained | **55.8** |
| Bidirectional simultaneous, aggregate | **98.9** (~49.5/dir) |
| D2H / H2D while SMs saturated (FMA kernel) | 57.2 / 55.8 — **kernel interference +0.1%** |
| Pageable (naive) D2H / H2D | 11.7 / 10.5 |
| NUMA local vs remote socket | identical (57.3 both) |

- Link reports **PCIe Gen6 x16** (`nvidia-smi`), but end-to-end delivery is
  **Gen5-class** — the Granite Rapids root port is Gen5; the Gen6
  negotiation is GPU↔switch only. Do not trust the link-gen readout for
  offload sizing.
- Copy engines are genuinely free under compute (+0.1%): the
  hide-copies-behind-compute premise holds at the hardware level given
  pinned memory.
- Pageable is 5× worse — a naive/staging implementation forfeits the plan.

## Verdict vs the offload plan (numbers from A2A_EXPOSURE_DECOMPOSITION /
blockK: S_eager 2.94 GiB/layer/mb, fwd ~42 ms and bwd ~51 ms per layer-mb
at d16/131k)

- **Offload-everything: copy-bound.** Needs ~74 GB/s D2H vs ~50-57
  available → fwd/bwd stretch ~25-50%; net win shrinks from +45% ceiling to
  roughly +15%.
- **Selective offload (free ~40-100 GiB, ~1 GiB/layer-mb of the biggest
  tensors): fits with ~2× margin** (~24 GB/s) → full no-recompute prize
  intact.
- **FP8-compressed stash of everything also fits** (~1.6 GiB/layer-mb →
  ~32 ms copy < 42 ms fwd compute).

## 8-GPU node results (2026-08-15, box w6xmmgw, fresh 1×8 B300 ali)

Both caveats below are now CLOSED. Tools: `bw_multi.cu` + `run_multi.sh`
(this folder). Each worker pinned to 8 physical cores on its GPU's local
NUMA node (GPUs 0-3 → node0, 4-7 → node1). 20 s sustained per mode.
Single-GPU suite re-run on this box class first: **identical to the 1-GPU
box** (57.3 / 55.6 / 98.9 bidi / +0% kernel interference / pageable 11.8)
— box-class confirmed representative, Gen5-class end-to-end here too.

| mode | per-GPU | node aggregate |
|---|---:|---:|
| all-8 D2H | 57.1–57.2 | **457 GB/s — zero degradation vs solo** |
| all-8 H2D | 55.0–55.1 | **441 GB/s — zero degradation** |
| all-8 bidirectional | 56.2–60.3 (both dirs summed) | **464 GB/s** |

**The surprise: the node has a hard ~460 GB/s total ceiling regardless of
direction mix.** Unidirectional 8-way scales perfectly (4×57 ≈ 229
GB/s/socket sits exactly at the ceiling). But bidirectional per-GPU
collapses 98.9 → ~58 GB/s (i.e. **~29 GB/s per direction per GPU** when
the whole node runs both directions at once) — the same ~460 node total,
just split across directions.

## Planning numbers for the offload design

- Stash-only or prefetch-only phases: budget **~55-57 GB/s per GPU**.
- Phases where all ranks stash AND prefetch simultaneously: budget
  **~29 GB/s per direction per GPU**. Under 1F1B fwd/bwd alternation the
  truth lands between the two; design to the 29 floor.
- Verdicts restated against the floor: offload-everything (74 GB/s) —
  infeasible, 2.5× short. Selective ~1 GiB/layer-mb (≈26 GB/s) — fits at
  the floor, comfortable at the alternating average. FP8-compressed
  selective (~13 GB/s) — trivial. FP8-compressed-everything (~37 GB/s) —
  fits only if directions mostly alternate; do not design to it without a
  schedule-aware prefetch throttle.
- The ~460 GB/s node ceiling is shared with everything else host-side
  (dataloader, host caches); leave margin.
