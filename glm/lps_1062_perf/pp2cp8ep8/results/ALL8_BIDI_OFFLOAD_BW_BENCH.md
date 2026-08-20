# All-8-GPU Bidirectional Host Memory Bandwidth Bench

**Date:** 2026-08-20
**Box:** `baseten-training-job-wlmlkeq-multinode-0` (SSH alias `tj-wlmlkeq`), 8 GPUs
(275040 MiB each), 2x Intel Xeon 6767P, CUDA 12.8 (nvcc V12.8.93).
**Purpose:** measure sustained GPU<->host-RAM copy bandwidth in the all-8-GPUs-at-once
*bidirectional* regime, which had never been measured on this box, and quantify the
multi-GPU cross-socket (NUMA) penalty.

This document reports measurements only. It draws no conclusion about any design or
project decision.

---

## Terminology (defined on first use)

| Term | Meaning |
|---|---|
| **D2H** | Device-to-host. Copying data from GPU memory to CPU (host) RAM. |
| **H2D** | Host-to-device. Copying data from CPU RAM to GPU memory. |
| **Pinned (page-locked) memory** | Host RAM that the OS is forbidden from swapping or relocating. Required for the GPU's DMA (direct memory access) engines to copy asynchronously at full speed. All numbers here use pinned memory; ordinary "pageable" memory is roughly 5x slower and is not used. |
| **Bidirectional / bidi** | D2H and H2D running *at the same time* on the same GPU, on two separate CUDA streams, so both DMA directions are busy concurrently. |
| **Unidirectional / unidir** | Only one direction active. |
| **NUMA** | Non-uniform memory access. Each CPU socket has its own directly-attached RAM ("NUMA node"). Reaching the *other* socket's RAM requires crossing the inter-socket link (Intel UPI, Ultra Path Interconnect), which is slower and has finite bandwidth. |
| **NUMA-local** | A GPU's host buffer lives in the RAM attached to the CPU socket that GPU is plugged into. |
| **NUMA-remote** | A GPU's host buffer lives deliberately in the *other* socket's RAM, so every byte crosses the inter-socket link. |
| **NUMA-interleave** | Host buffer pages alternate between both sockets (Linux `MPOL_INTERLEAVE`), so roughly half the traffic crosses the inter-socket link. This is what a process gets when it does not bind memory deliberately. |
| **Sustained** | Rate averaged over a continuous multi-second run, not a short burst peak. |
| **Aggregate** | Sum across all 8 GPUs. |
| **GB/s** | Gigabytes per second, decimal (1e9 bytes/s). |

All bandwidths below are **decimal GB/s** and are **sustained** (>= 10 s of continuous
copying, and 60 s for the headline pattern), not peaks.

---

## Verified machine topology

Confirmed on the box, not assumed:

```
nvidia-smi topo -m            # CPU Affinity / NUMA Affinity columns
cat /sys/devices/system/node/node*/cpulist
cat /sys/fs/cgroup/memory.max
```

- **GPU 0,1,2,3 -> NUMA node 0**, whose CPUs are `0-63,128-191`
- **GPU 4,5,6,7 -> NUMA node 1**, whose CPUs are `64-127,192-255`
- 256 logical CPUs total; node 0 RAM 2113159176 kB, node 1 RAM 2111050048 kB (~1.97 TiB each)
- cgroup `memory.max` = 2662556762112 bytes (2.42 TiB)
- All 8 GPUs are mutually NVLink-connected (`NV18`); cross-socket GPU pairs show `SYS`
  (traffic traverses PCIe plus the inter-socket link).
- `numactl` is **not installed** on this box, so NUMA placement was controlled by
  pinning each worker thread to a CPU on the target node before allocating, letting
  Linux first-touch policy place the pages there (plus `mbind(MPOL_INTERLEAVE)` for the
  interleave case). Placement was then **verified**, not assumed: every run samples 64
  pages of every buffer with the `move_pages` syscall and prints how many landed on
  node 0 vs node 1. Those counts appear in every result table below and were correct in
  every run.
- GPUs report as `NVIDIA L20D` in `nvidia-smi` (a masked/renamed SKU string in this
  environment), 275040 MiB each, matching the expected 8-GPU shape.
- Box was idle before the runs: all 8 GPUs at 0% utilization, 0 MiB used, no compute
  processes.

---

## Benchmark source and why it was extended

Prior art on the box: `/tmp/cuda_host_bw.cu` (written by agent *huygens* earlier the
same day). It already implemented pinned D2H/H2D/bidirectional copies, CPU pinning
before allocation, and `move_pages` placement verification. It could **not** express
what was needed here, for four reasons:

1. **Duration was hardcoded** at 40 iterations of 256 MiB = ~10 GiB per direction,
   i.e. about **0.19 s** at these rates. That is a burst peak, not a sustained rate,
   and it cannot show whether throughput sags over time.
2. **A single `alloc-cpu` argument applied to all 8 workers**, so it could not express
   the per-GPU NUMA-remote mapping needed (GPUs 0-3 -> node 1 *and* GPUs 4-7 -> node 0
   simultaneously).
3. **Bidirectional rates were reported as a combined figure divided by two**, which
   assumes the two directions are symmetric. They are not (H2D is consistently ~5%
   faster than D2H here), so per-direction rates had to be measured independently.
4. **No rate limiting**, needed for the asymmetric-demand pattern.

So it was extended into `all8_bidi_bw.cu` (archived next to this document as
`all8_bidi_bw.cu`, and on the box at `/tmp/all8_bidi_bw.cu`). Changes:

- **Wall-clock duration** (`--seconds`) instead of a fixed iteration count.
- **One thread per (GPU, direction)** — so 16 threads for the bidirectional patterns,
  each with its own CUDA stream, its own pinned host buffer and its own device buffer.
  Both directions therefore stay busy for the *entire* run; neither can finish early
  and hand the other an uncontended tail (which would inflate the survivor's number).
- **Per-direction rates measured independently** rather than a combined figure halved.
- **Per-window stability sampling** (`--interval`), so sag over the run is visible.
- **NUMA policy selector** (`--numa=local|remote|interleave`) with the correct per-GPU
  mapping, and `mbind(MPOL_INTERLEAVE)` + `cudaHostRegister` for the interleave case.
- **Optional rate limiting** (`--target-d2h`, `--target-h2d`) with per-copy completion
  latency percentiles.
- Warm-up copies before the timed region, and a thread barrier so all 16 workers start
  together.

Memory footprint per run: 16 workers x 256 MiB pinned host = 4 GiB host, plus 512 MiB
device per GPU. Far below the 2.42 TiB cgroup limit.

### Build

```bash
scp all8_bidi_bw.cu tj-wlmlkeq:/tmp/all8_bidi_bw.cu
ssh tj-wlmlkeq 'nvcc -O3 -std=c++17 -o /tmp/all8_bidi_bw /tmp/all8_bidi_bw.cu'
```

(One benign warning about deprecated pre-sm_75 offline compilation targets. Built clean.)

### Exact commands run

```bash
# (0) cross-validation: all-8 unidirectional, NUMA-local
/tmp/all8_bidi_bw --gpus=8 --mode=d2h    --numa=local      --seconds=10 --interval=1 --buf-mib=256 --chunk=8
/tmp/all8_bidi_bw --gpus=8 --mode=h2d    --numa=local      --seconds=10 --interval=1 --buf-mib=256 --chunk=8

# (2) all-8 bidirectional, NUMA-remote, and the interleaved variant
/tmp/all8_bidi_bw --gpus=8 --mode=bidir  --numa=remote     --seconds=20 --interval=2 --buf-mib=256 --chunk=8
/tmp/all8_bidi_bw --gpus=8 --mode=bidir  --numa=interleave --seconds=15 --interval=5 --buf-mib=256 --chunk=8

# (1) all-8 bidirectional, NUMA-local  (headline; plus a 60 s soak)
/tmp/all8_bidi_bw --gpus=8 --mode=bidir  --numa=local      --seconds=20 --interval=2 --buf-mib=256 --chunk=8
/tmp/all8_bidi_bw --gpus=8 --mode=bidir  --numa=local      --seconds=60 --interval=5 --buf-mib=256 --chunk=8

# controls, to prove the remote collapse is real and not a harness artifact
/tmp/all8_bidi_bw --gpus=1 --mode=bidir  --numa=local      --seconds=8  --interval=4 --buf-mib=256 --chunk=8
/tmp/all8_bidi_bw --gpus=1 --mode=bidir  --numa=remote     --seconds=8  --interval=4 --buf-mib=256 --chunk=8
/tmp/all8_bidi_bw --gpus=8 --mode=d2h    --numa=remote     --seconds=10 --interval=5 --buf-mib=256 --chunk=8

# (3) all-8 bidirectional, NUMA-local, rate-limited asymmetric demand
/tmp/all8_bidi_bw --gpus=8 --mode=bidir  --numa=local --seconds=15 --interval=5 \
                  --buf-mib=32 --target-d2h=22 --target-h2d=24
```

---

## HEADLINE ANSWER

> **With all 8 GPUs active simultaneously, both directions running at once, and pinned
> host buffers on each GPU's local NUMA node, the sustained per-GPU bidirectional
> bandwidth is:**
>
> ## **27.5 GB/s D2H and 28.8 GB/s H2D, per GPU, simultaneously**
>
> **Aggregate across 8 GPUs: 219.6 GB/s D2H + 230.6 GB/s H2D = 450.1 GB/s total.**
> Held flat for 60 s with under +/-1% window-to-window variation. No degradation.

The important structural observation is that **the total is capped at ~450 GB/s
regardless of how it is split between directions.** All-8 unidirectional D2H alone
reaches 454.7 GB/s; all-8 bidirectional reaches 450.1 GB/s *combined across both
directions*. Turning on the second direction at 8 GPUs does not add throughput — it
splits the same ceiling. That is different from the single-GPU behaviour, where the two
directions *are* largely additive (48.4 + 50.7 = 99.1 GB/s on one GPU).

---

## Results

### (0) Cross-validation — all-8 UNIdirectional, NUMA-local

Purpose: reproduce the previously measured numbers with this harness before trusting it
on the unknown patterns.

| Direction | Per-GPU (mean) | Per-GPU range | Aggregate (this bench) | Aggregate (prior bench) | Match |
|---|---|---|---|---|---|
| D2H | **56.84 GB/s** | 56.41 - 57.19 | **454.73 GB/s** | 456 GB/s | within 0.3% |
| H2D | **54.55 GB/s** | 54.01 - 54.98 | **436.39 GB/s** | 439 GB/s | within 0.6% |

**Plainly: yes, this harness reproduces the known numbers.** ~57 GB/s per GPU D2H and
~55 GB/s per GPU H2D, with no per-GPU collapse (spread across the 8 GPUs is under 1.5%).
Prior figures were 57.3 D2H / 55.7 H2D per GPU; measured here 56.84 / 54.55, i.e. 0.8%
and 2.1% lower, consistent with this being a *sustained* 10 s average rather than a
0.19 s burst.

Stability: D2H windows 446.3 (first, includes ramp) then 454.4-456.3 for the remaining
nine. H2D windows 432.7-439.6 with no trend. Stable.

### (1) and (2) — all-8 BIDIRECTIONAL, by NUMA placement

Every row is all 8 GPUs running D2H and H2D simultaneously. Per-GPU figures are the
mean over the 8 GPUs.

| Pattern | Host buffer placement | Per-GPU D2H | Per-GPU H2D | Aggregate D2H | Aggregate H2D | Aggregate BOTH | Duration |
|---|---|---|---|---|---|---|---|
| **(1) NUMA-local** | each GPU's own socket | **27.46 GB/s** | **28.94 GB/s** | **219.65 GB/s** | **231.54 GB/s** | **451.19 GB/s** | 20 s |
| **(1) NUMA-local, soak** | each GPU's own socket | **27.45 GB/s** | **28.82 GB/s** | **219.56 GB/s** | **230.55 GB/s** | **450.11 GB/s** | **60 s** |
| **(2b) interleaved** | pages alternating both sockets | 15.60 GB/s | 16.83 GB/s | 124.83 GB/s | 134.63 GB/s | 259.46 GB/s | 15 s |
| **(2) NUMA-remote** | deliberately the wrong socket | 7.39 GB/s | 7.82 GB/s | 59.12 GB/s | 62.54 GB/s | **121.66 GB/s** | 20 s |

Per-GPU spread within the NUMA-local run is very tight: D2H 27.14-27.91, H2D 28.73-29.20.
No individual GPU collapses.

Per-GPU spread within the NUMA-remote run is wider — D2H 5.77-8.98, H2D 6.20-9.21 —
i.e. once the inter-socket link is the bottleneck, the GPUs no longer share it evenly.
GPU0 gets the least (5.77 D2H), GPU7 the most (8.98 D2H), a 1.56x spread.

### Stability over the run

| Pattern | Window size | D2H window range | H2D window range | Verdict |
|---|---|---|---|---|
| (0) all-8 unidir D2H, local, 10 s | 1 s | 446.3 - 456.3 | - | stable after 1st window |
| (0) all-8 unidir H2D, local, 10 s | 1 s | - | 432.7 - 439.6 | stable, no trend |
| (1) all-8 bidir local, 20 s | 2 s | 218.4 - 221.3 | 229.6 - 233.1 | **stable, +/-0.7%** |
| **(1) all-8 bidir local, 60 s soak** | 5 s | **218.2 - 222.2** | **227.5 - 231.5** | **stable, +/-0.9%, no sag** |
| (2b) all-8 bidir interleave, 15 s | 5 s | 124.5 - 125.3 | 133.6 - 135.3 | stable, +/-0.4% |
| (2) all-8 bidir remote, 20 s | 2 s | 57.6 - 61.7 | 61.1 - 64.9 | stable, +/-3.5% jitter, no trend |

**Answer to "does it sag as host memory controllers saturate": no.** In the 60 s
NUMA-local bidirectional soak, the first 5 s window (219.8 D2H / 230.5 H2D) and the last
(222.2 / 227.5) are within 1% of each other, and the twelve windows show no monotonic
trend in either direction. The 20 s and 60 s runs agree to within 0.25% on every figure.
The NUMA-remote case is noisier (~3.5% window-to-window) but likewise shows no decline
from start to finish. Rates here are ceilings that are hit immediately and held, not
thermal or controller-saturation effects that build up.

### Multi-GPU NUMA cross-socket penalty

Penalty = reduction in sustained bandwidth caused purely by moving the host buffers to
the wrong socket, all else identical.

| Comparison | Local | Remote | **Penalty** |
|---|---|---|---|
| **All-8 bidirectional, total** | 451.19 GB/s | 121.66 GB/s | **-73.0%** (3.71x slower) |
| All-8 bidirectional, D2H only | 219.65 GB/s | 59.12 GB/s | **-73.1%** |
| All-8 bidirectional, H2D only | 231.54 GB/s | 62.54 GB/s | **-73.0%** |
| All-8 bidirectional, interleaved vs local | 451.19 GB/s | 259.46 GB/s | **-42.5%** |
| All-8 **unidirectional** D2H | 454.73 GB/s | 144.02 GB/s | **-68.3%** |
| **Single-GPU** bidirectional (control) | 99.09 GB/s | 99.12 GB/s | **+0.03%, i.e. none** |

**The previously reported "<1% NUMA penalty" figure is confirmed — but it holds only for
the single-GPU case, and it does not survive to 8 GPUs.** At 1 GPU the penalty is
literally unmeasurable (99.09 vs 99.12 GB/s, remote marginally *faster*, i.e. inside
run-to-run noise). At 8 GPUs the penalty is **73%**. The interleaved case, which is what
an unbound process gets by default, sits in between at **42.5%**, roughly consistent with
half its traffic crossing the link.

### Controls proving the remote collapse is real

The 73% figure is large enough to warrant checking it is not a harness bug. Three
independent checks:

1. **Page placement was verified, per buffer, in every run** via `move_pages`. In the
   NUMA-remote run, all 8 buffers belonging to GPUs 0-3 sampled 0/64 pages on node 0
   and 64/64 on node 1 (correctly remote), and all 8 buffers of GPUs 4-7 sampled 64/64
   on node 0 (correctly remote). In the interleave run every buffer sampled exactly
   32/32. Placement was as intended in every case.
2. **Single-GPU bidirectional, local vs remote, same binary, same code path:**
   99.09 vs 99.12 GB/s — no penalty. If the remote code path were broken, this would
   have collapsed too. It did not. The collapse appears *only* under 8-GPU contention.
3. **All-8 unidirectional D2H remote:** 144.02 GB/s aggregate (18.00 GB/s per GPU),
   versus 454.73 GB/s local. So the cross-socket ceiling also binds with only one
   direction active, ruling out anything specific to the two-stream bidirectional setup.

These are mutually consistent: a single GPU pushing 99 GB/s across the inter-socket link
fits under the link's ceiling and pays nothing, whereas 8 GPUs demanding ~450 GB/s across
it are clamped to roughly **122-144 GB/s aggregate**. Bidirectional cross-socket traffic
(121.66 GB/s) is slightly *less* efficient than unidirectional (144.02 GB/s), consistent
with contention on a shared link.

### (3) Asymmetric realistic mix — rate-limited to 22 GB/s D2H / 24 GB/s H2D per GPU

All 8 GPUs, bidirectional, NUMA-local, each direction paced to a target rate with
32 MiB chunks (smaller chunks give finer pacing granularity), 15 s.

| Direction | Target per GPU | Achieved per GPU | Achieved aggregate | Target met? |
|---|---|---|---|---|
| D2H | 22.0 GB/s | **21.95 GB/s** | 175.58 GB/s | **Yes** (99.8% of target) |
| H2D | 24.0 GB/s | **23.89 GB/s** | 191.08 GB/s | **Yes** (99.5% of target) |
| Both | 46.0 GB/s | 45.84 GB/s | 366.67 GB/s | Yes |

The 0.2-0.5% shortfall is pacing-loop granularity, not a capacity limit — individual
GPUs hit 22.00 and 24.00 exactly. Demand was met on every one of the 8 GPUs in both
directions, and held flat across all three 5 s windows (D2H 174.8-176.0, H2D 189.3-192.0).

**Headroom.** The unthrottled all-8 bidirectional NUMA-local ceiling is 27.46 D2H /
28.94 H2D per GPU, so against a 22 / 24 demand:

| Direction | Demand | Ceiling | Headroom |
|---|---|---|---|
| D2H | 22 GB/s | 27.46 GB/s | **+5.46 GB/s (+24.8%)** |
| H2D | 24 GB/s | 28.94 GB/s | **+4.94 GB/s (+20.6%)** |

Per-copy completion latency for a 32 MiB chunk under this load (time from copy
enqueue to DMA completion, ~9800-10700 samples per stream):

| Percentile | D2H | H2D |
|---|---|---|
| p50 | 0.733 - 0.783 ms | 0.731 - 0.766 ms |
| p90 | 0.874 - 0.964 ms | 0.851 - 0.925 ms |
| p99 | 0.94 - 5.25 ms | 0.93 - 3.62 ms |
| max | 1.74 - 123 ms | 0.97 - 125 ms |

Median latency is ~0.75 ms, i.e. a 32 MiB chunk moves at ~45 GB/s instantaneous when it
is its turn, and the pacing loop leaves the link idle the rest of the time — there is no
standing queue at this demand level. The tail is worth flagging honestly: p99 stays
around 1-5 ms, but 14 of the 16 streams recorded a single-digit number of ~100 ms
outliers (max 125 ms). At ~10000 samples per stream these are ~0.01% events and do not
move the throughput figures; they were not investigated further and their cause is
unknown (plausibly host scheduling, since GPU0's two streams — on CPUs 0 and 1 — show no
such outliers at all).

---

## Summary table (all patterns, sustained, per-GPU and aggregate)

| # | Pattern | GPUs | Directions | NUMA | Per-GPU D2H | Per-GPU H2D | Agg D2H | Agg H2D | Agg total | Stable? |
|---|---|---|---|---|---|---|---|---|---|---|
| 0a | unidirectional D2H | 8 | D2H | local | 56.84 | - | 454.73 | - | 454.73 | yes |
| 0b | unidirectional H2D | 8 | H2D | local | - | 54.55 | - | 436.39 | 436.39 | yes |
| **1** | **bidirectional** | **8** | **both** | **local** | **27.45** | **28.82** | **219.56** | **230.55** | **450.11** | **yes (60 s)** |
| 2b | bidirectional | 8 | both | interleave | 15.60 | 16.83 | 124.83 | 134.63 | 259.46 | yes |
| 2 | bidirectional | 8 | both | remote | 7.39 | 7.82 | 59.12 | 62.54 | 121.66 | yes (noisier) |
| 3 | bidirectional, paced 22/24 | 8 | both | local | 21.95 | 23.89 | 175.58 | 191.08 | 366.67 | yes |
| A1 | bidirectional (control) | 1 | both | local | 48.43 | 50.66 | 48.43 | 50.66 | 99.09 | yes |
| A2 | bidirectional (control) | 1 | both | remote | 48.13 | 50.99 | 48.13 | 50.99 | 99.12 | yes |
| B | unidirectional (control) | 8 | D2H | remote | 18.00 | - | 144.02 | - | 144.02 | yes |

---

## Reproduction notes / caveats

- Nothing on the box was rebooted, killed, or stopped. Another agent was doing read-only
  source archaeology concurrently; host load average was ~7.5 out of 256 logical CPUs
  during the runs, which is negligible, and all 8 GPUs were idle (0% utilization, 0 MiB
  allocated, no compute processes) before starting.
- All figures are pinned (page-locked) host memory. Pageable memory was not measured
  here; it is known to be roughly 5x worse.
- Bandwidths are decimal GB/s (1e9 bytes/s), computed as bytes moved divided by
  wall-clock elapsed time per worker thread.
- Chunk structure for unthrottled runs: batches of 8 copies of 256 MiB enqueued
  asynchronously then synchronized, repeated until the wall-clock deadline. The
  synchronize boundary costs microseconds against ~35 ms batches, so it is negligible.
- Source archived alongside this document as `all8_bidi_bw.cu`; on the box it is at
  `/tmp/all8_bidi_bw.cu` (binary `/tmp/all8_bidi_bw`). The original prior-art bench it
  extends is `/tmp/cuda_host_bw.cu`.
- `numactl` is not available on this box; NUMA placement is done by thread CPU affinity
  plus first-touch, and by `mbind(MPOL_INTERLEAVE)` for the interleave case. Placement
  is verified per-run with `move_pages` and the verification counts are printed in the
  raw output.
