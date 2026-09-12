# Full GLM5.3 FSDP comparison — final ablation pending

**The first grouped-MM integration has a confirmed synchronous-offset-upload
pathology.** Its measurements below remain valid for that implementation, not
for the pending asynchronous fix. The original microbenchmark pre-created its
offset tensor outside the timed region and therefore did not exercise this cost.

131072 tokens, CP8, LoRA32, BF16 expert storage, full recompute. TPS/GPU below
is measured from five unprofiled forward/backward controls, not extrapolated.

| Machine / capture generation | EP | Mean FB | TPS/GPU | Peak allocated |
|---|---:|---:|---:|---:|
| Old profiler pod, preliminary | 1 | 11.5838 s | 1414.4 | 248.10 GiB |
| Old profiler pod, preliminary | 8 | 10.5831 s | 1548.1 | 213.62 GiB |
| New devbox, current annotations | 1 | 11.9001 s | 1376.8 | 248.10 GiB |
| New devbox, current annotations | 8 | 10.5694 s | 1550.1 | 213.62 GiB |

EP1 with the current grouped-MM implementation: **11.7888 s, 1389.8 TPS/GPU,
247.98 GiB**, five controls, SD0.0793s. This is only0.94% higher TPS than TE,
within the observed variation; it does not establish a useful end-to-end gain.
EP8 grouped-MM and a TE EP1 repeat are pending. The repeat removes the earlier
TE profiler/allocator anomaly and mapped-libc setup difference.

The current matched pair has **12.59% higher TPS/GPU with EP8**. Same source,
venv/package versions, profiler and GPU UUIDs; exact checkpoint revision restored
under a different cache root. The old pair independently showed a 9.46% advantage.
Do not mix old and new rows into an A/B. See `devbox-te-comparison.md` and
`devbox-ep1-grouped-comparison.md`.
Setup caveat: installing the debugger upgraded system libc while EP1 was already
running; it retained the older mapped libc. This is an uncontrolled CPU-runtime
difference, not an established cause of the performance gap.

## What the traces establish

- In the unfixed EP8 grouped path, every rank executes **300
  cudaStreamSynchronize calls per FB** while constructing expert offsets:
  **1.095–1.247s of host API residence**, versus none in the corresponding TE
  forward scopes. The CPU-list-to-CUDA tensor constructor synchronizes after
  H2D; the fix stages in pinned CPU memory and copies nonblocking. This removes
  an enqueue-ahead barrier, not necessarily that entire duration from wall time.
  The reusable analyzer now reports these per-scope API costs automatically.
  Manual GPU checks confirmed exact offset equality including zero/uneven
  expert counts. Full-model fixed-path captures are still pending.
- **Grouped-MM is active, but its EP1 GPU kernels are not faster in situ.**
  Rank0 forward+recompute expert GPU union: TE1.294s, grouped1.278s; real
  input-gradient backward: TE0.878s, grouped1.018s. Total2.173→2.296s.
  Kernel inventory confirms the256×256×64 CUTLASS grouped kernels. Host time
  outside recorded CUDA APIs within expert scopes falls2.394→0.163s, but these
  host durations are not additive critical-path costs. Thus the isolated
  microbenchmark gain did not transfer to the full trainer.
- EP1 grouped sampled Tensor Active rises to77–87% in exclusive expert samples,
  despite no GPU-time win. Its metrics pass was~14.5% slower than control median
  (timing pass only0.38% slower). Utilization is not useful-FLOP efficiency or
  evidence of a speedup. Padding, weight traffic, overlapping work and clock
  behavior require further controlled measurements to separate; do not claim
  a quantified cause from this counter alone. The exported GPC clock unit is
  inconsistent with its raw values; the analyzer now flags it rather than
  reporting an impossible MHz value.
- Current matched pair, rank0: **expert-path GPU union 2.173 s EP1 vs 1.244 s
  EP8**, attention 4.420 vs 4.020 s, GPU idle 1.725 vs 0.232 s. EP1 idle
  coincident with expert host scopes was 0.399 s vs 0.002 s. These differences
  overlap and are not additive speedup estimates. The traced step difference
  is 2.068 s while the unprofiled difference is 1.331 s: EP1's larger profiling
  perturbation must not be mistaken for production time saved.
- Current EP8 expert dispatcher/combine path occupies 2.950 s on rank0;
  CP collectives occupy 0.342 s. EP1's local dispatch bookkeeping is 0.295 s
  despite no expert cross-rank dispatch; CP collectives occupy 0.926 s.
  Dispatcher time includes packing/metadata, and collective residence includes
  waiting for peers. This does not measure raw network transfer time.
- Old matched pair, rank0: expert-path GPU union was **2.161 s EP1 vs 1.279 s
  EP8**. These include forward, recompute and input-gradient backward. EP1
  attention was also slower (4.473 vs 4.046 s) and had more idle time (~1.5 vs
  ~0.26 s). The difference cannot honestly be attributed solely to GEMMs.
- New EP1: expert-path union is **2.108–2.188 s across ranks**. Steady expert
  FSDP prefetches arrived by the preceding block's compute end in **149/149
  comparable cases on every rank**. One initial backward refill has no prior
  block to compare. This rules out late steady expert-weight delivery in this
  capture, not contention or all other synchronization.
- New EP1 expert Tensor Active in category-exclusive samples: **53.7–59.7%**;
  only ~51–57% of expert-resident samples were exclusive. Current matched EP8
  is **91.2–92.4%**, with all expert samples exclusive. This supports a real
  execution-efficiency gap; these are sampled device metrics, not NCU rooflines
  or proof that changing only the GEMM will recover the entire time difference.
- New EP1 rank1 had **0.632 s of GPU idle intersecting allocation/VMM API
  calls**, with long calls in the output head. Other ranks had zero such
  intersection in this capture. This is not proof of equivalent unprofiled
  allocator cost: the timing capture was **6.67% slower** than control median,
  versus **0.71%** for EP8. EP8 had no allocation-coincident idle on any rank.

Nsight warns that some CUDA/NVTX events may not have been collected. All eight
rank anchors and runtime correlations are present, but these do not prove zero
event loss. Communication residence/exposure is not recoverable wall time.

## Deliverables and remaining work

`workflow.py` provides capture, verified collection, cached analysis and matched
comparison. No manual SQL is needed for reruns. First ingestion takes tens of
seconds; unchanged cached analysis is immediate. `README.md` documents usage.

Trainer implementations and annotations: [PR1355](https://github.com/basetenlabs/trainers/pull/1355).
Baseline captures use13137ef1a; async-offset fix is committed in57b9a3ef4,
pinning Bridgea3438227 and Corefe3976282. Analysis/run files belong only to this separate artifact repo.
No tests were added or modified. Original tools remain unchanged by this workflow.

The new devbox uses the old venv and built-in nsys2025.3.1. The old profiler pod
was deleted after migration. The shared full checkpoint disappeared before EP8
startup; its exact HF revision has been restored to node-local disk. All EP1/EP8
TE runtime and metrics artifacts are now on the Mac, SHA256 verified and analyzed.
Finish both grouped-MM variants before claiming the proposed optimization gain.
