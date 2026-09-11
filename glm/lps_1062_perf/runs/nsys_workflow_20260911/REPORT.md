# Full GLM5.3 FSDP comparison — incomplete

131072 tokens, CP8, LoRA32, BF16 expert storage, full recompute. TPS/GPU below
is measured from five unprofiled forward/backward controls, not extrapolated.

| Machine / capture generation | EP | Mean FB | TPS/GPU | Peak allocated |
|---|---:|---:|---:|---:|
| Old profiler pod, preliminary | 1 | 11.5838 s | 1414.4 | 248.10 GiB |
| Old profiler pod, preliminary | 8 | 10.5831 s | 1548.1 | 213.62 GiB |
| New devbox, current annotations | 1 | 11.9001 s | 1376.8 | 248.10 GiB |
| New devbox, current annotations | 8 | pending | pending | pending |

The old pair is matched: EP8 was 9.46% faster in TPS. Do not combine the new
EP1 measurement with old EP8 as a same-machine A/B. Full-model grouped-MM
ablation has not run; its end-to-end benefit remains unmeasured.

## What the traces establish

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
  only ~51–57% of expert-resident samples were exclusive. Old EP8 was ~91–93%
  with almost all samples exclusive. This supports a GEMM-efficiency issue,
  but is a cross-machine/capture comparison, not a controlled NCU experiment.
- New EP1 rank1 had **0.632 s of GPU idle intersecting allocation/VMM API
  calls**, with long calls in the output head. Other ranks had zero such
  intersection in this capture. This is not proof of equivalent unprofiled
  allocator cost: the timing capture was **6.67% slower** than control median.

Nsight warns that some CUDA/NVTX events may not have been collected. All eight
rank anchors and runtime correlations are present, but these do not prove zero
event loss. Communication residence/exposure is not recoverable wall time.

## Deliverables and remaining work

`workflow.py` provides capture, verified collection, cached analysis and matched
comparison. No manual SQL is needed for reruns. First ingestion takes tens of
seconds; unchanged cached analysis is immediate. `README.md` documents usage.

Trainer implementations and annotations: [PR1355](https://github.com/basetenlabs/trainers/pull/1355),
commit13137ef1a. Analysis/run files belong only to this separate artifact repo.
No tests were added or modified. Original tools remain unchanged by this workflow.

The new devbox uses the old venv and built-in nsys2025.3.1. The old profiler pod
was deleted after migration. The shared full checkpoint disappeared before EP8
startup; its exact HF revision has been restored to node-local disk. Resume EP8,
then both grouped-MM variants, before claiming a final bottleneck comparison.
