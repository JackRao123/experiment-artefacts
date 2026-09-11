# GLM-5.3 topology investigation

PR: https://github.com/basetenlabs/trainers/pull/1355 (draft). Core dependency
PR #76 and Bridge dependency PR #84 are linked from it. Original profile_driver.py
and mfu.py are unchanged in this investigation; no tests were added or modified.

## Three debug-model measurements

Previously agreed 1d2m proxy, real checkpoint values, BF16 expert storage,
131072 tokens, LoRA rank/alpha32, full one-layer recompute, TP1/PP1/ETP1.
Three warmups precede five unprofiled controls. Separate memory and all-rank
runtime captures; twenty additional controls are retained as a separate check.

| Topology | GPUs | Five-control FB mean ± SD | Five-control TPS/GPU | Twenty-control TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| CP1EP1 | 1 | 3.4245 ± 0.0095 s | 38,275 | 38,104 | 142.10 |
| CP8EP1 | 8 | 0.6270 ± 0.0343 s | 26,130 | 27,556 | 58.26 |
| CP8EP8 | 8 | 0.6497 ± 0.0769 s | 25,218 | 26,672 | 26.76 |

The order holds here, but EP1 is only about 3.3% ahead of EP8 in the matching
twenty-control means. CP1 is more efficient per GPU, not faster in request
latency: one GPU takes about 3.44 seconds versus about 0.595 seconds on eight.

## Why removing dispatch/combine is not a huge whole-step win

- Dispatch/combine was only about 20.5 ms summed GPU kernel time in the
  complete EP8 proxy step, not most of a roughly 0.6-second request. Overlap
  and peer waits mean that sum is not an exact removable wall-time cost.
- EP1 still routes, permutes and computes experts. At balanced counts, it
  changes 32 experts x 4096 rows to 256 experts x 512 rows per GPU. A
  communication-free BF16 probe measured 10.35 ms versus 6.16 ms for equal
  forward arithmetic, and 10.00 versus 5.62 ms for input gradients. More
  small GEMMs and eight times the resident expert weight footprint offset
  some communication savings. Actual routing is much more uneven than this
  balanced diagnostic, with many empty or tiny experts.
- The expert MLP itself can improve substantially while the whole step
  barely moves. The shared-index MoE block's MLP forward/recompute/backward
  intervals are about 15.56/14.02/14.83 ms at EP1 versus 21.52/21.58/22.57 ms
  at EP8. Attention, head/loss and request work remain.
- No SM/tensor-core hardware counters were collected. Effective GEMM-path
  throughput is measured; individual-kernel saturation is not established.

## Why CP1 has higher TPS/GPU

The small proxy pays a complete LM head/loss and request-processing cost for
only three blocks. On matching FB+optimizer boundaries, CP1 latency divided
by eight is 436.38 ms versus CP8EP1's 623.67 ms. Of that 187.28-ms efficiency
gap, 109.77 ms (58.6%) lies outside the backend timer. The remainder consists
of MLP, attention and backend residual work. This is not a claim that 58.6%
of total training time is network overhead.

## Bugs and pathologies addressed

- CP1's indexer now uses the causal-offset cuDNN path rather than the slow
  per-head FP32 fallback (earlier fix retained).
- Singleton EP/ETP avoids invalid cooperative HybridEP dispatch and now
  skips source/expert chunk sorting that is mathematically identity.
- Warmed long-lived objects are excluded from repeated Python generation-2
  scans via an opt-in GC policy; new-object GC remains enabled.
- MFSDP unwrapping now reaches the model through its inner wrapper.
- FSDP startup no longer flushes GC/CUDA cache before every module merely
  because persistent shards exceed 50% of device memory. Root parameter
  counting is cached instead of repeated for every module.
- FSDP empty DP replicas enter loss-masked model calls so weight all-gathers
  remain aligned. The ordinary DDP shortcut of skipping the model is unsafe
  for FSDP.
- Full-model runtime uses persistent parameter buffers instead of repeated
  storage resizing. Scoped shutdown also cleans run-owned orphan helpers.

## Full-model FSDP result

Full 743.6B-parameter GLM-5.3, BF16 FSDP CP8EP1, one 131072-token datum:

| Buffer policy | FB mean ± SD | TPS/GPU | Peak allocated / reserved GiB |
|---|---:|---:|---:|
| Dynamic, diagnostic baseline | 84.200 ± 4.501 s | 194.58 | 239.13 / 262.79 |
| Persistent max-pool, validated | 11.714 ± 0.244 s | 1398.64 | 248.10 / 252.79 |

The speedup is 7.19x. Rank0's memory-profile mapping/unmapping changed from
93.35/92.19 GiB to 0.117/0 GiB. All forty rank-controls of the persistent-buffer
run had zero allocation retries, device frees and allocator-wide synchronizations.
FSDP still communicates weights: 2700 GiB logical expert-weight all-gather
output per rank per step, about 4.45 seconds summed in the separate trace.

## Lower CP boundary

- CP4 with parameter lookahead OOMed in FP32 LM-head logit addition.
- CP4 without lookahead completed at 26.425 ± 2.240 s / 1240 TPS/GPU for two
  datums, but all-rank counters found sixteen allocator retries. Retained as
  a marginal-memory diagnostic, not the final clean CP4 measurement.
- CP2 without lookahead OOMed on a 6-GiB expert output before the LM head.
  Live-state snapshots were captured; no control TPS is claimed.
- CP4 with opt-in, numerically checked LM-head fusion and 2048-token head
  chunks completed at 21.527 ± 0.360 seconds / 1522.15 TPS/GPU for two datums.
  Peak allocated/reserved is 252.64/258.38 GiB. All forty rank-controls have
  zero allocation retries, OOMs, device frees and allocator-wide stream syncs.
  CP4 is the lowest CP successfully measured; CP2's earlier expert-layer
  allocation failure is not addressed by an LM-head-only optimization.

The final CP4 recipe is 8.8% higher TPS/GPU than the earlier CP8 recipe, but
this is not a CP-only ablation: prefetch and head settings differ. All
experiment trainers/helpers are stopped; the devbox itself remains available.

See RESULTS.md and GEMM.md for the full debug accounting, and ARTIFACTS.md for
clickable Kineto and memory-snapshot paths. Every full-model variant has its
own sibling run directory, code fingerprints, raw controls and diagnostics.
