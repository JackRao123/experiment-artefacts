# GLM-5.3 FSDP CP8 comparison

**TE + EP8 is the fastest tested FSDP configuration: 1550 TPS/GPU.**
EP1 does not become faster by removing token dispatch/combine: it instead
requires expert-weight FSDP gathers and has less efficient expert execution.
The grouped-MM upload fix is real and verified, but does not establish a
throughput improvement. Keep `BT_FSDP_GROUPED_MM=0` for this measured workload.

## Measurements

Full model, 131072 input tokens, one fixed synthetic SFT datum, eight HGX B300 s,
CP8/TP1/PP1/ETP1, BF16 routed-expert storage, LoRA rank/alpha 32, full uniform
one-layer recompute. Three warmups and five unprofiled controls per case.
TPS/GPU = 131072 / (8 × mean forward/backward request seconds); optimizer
time is excluded. These are real full-model measurements, not extrapolations.

| EP | Expert implementation | Mean FB ± sample SD (s) | TPS/GPU | Peak allocated GiB |
|---:|---|---:|---:|---:|
| 1 | TE, fresh repeat | 11.837 ± 0.192 | 1384.1 | 248.10 |
| 8 | TE | 10.569 ± 0.219 | 1550.1 | 213.62 |
| 1 | Grouped-MM, synchronous offsets | 11.789 ± 0.079 | 1389.8 | 247.98 |
| 8 | Grouped-MM, synchronous offsets | 11.328 ± 0.297 | 1446.3 | 213.49 |
| 1 | Grouped-MM, asynchronous offsets | 11.874 ± 0.200 | 1379.9 | 247.98 |
| 8 | Grouped-MM, asynchronous offsets | 11.200 ± 0.182 | 1462.9 | 213.49 |

TE EP8 has 12.0% higher TPS than TE EP1. Fixed grouped-MM is effectively tied
with TE on EP1 and 5.6% below TE on EP8. The upload fix changes grouped TPS by
−0.7% on EP1 and +1.1% on EP8, within the observed run variability.
Mean optimizer times were 0.511/0.162 s for TE EP1/EP8 and 0.468/0.176 s for
fixed grouped EP1/EP8; they are separate from the table's FB metric.

The initial devbox TE EP1 result was 1376.8 TPS/GPU; the fresh repeat above
removes its older mapped-libc setup difference and checks its allocator/profile
anomaly. It confirms the conclusion rather than selecting a different winner.
Earlier profiler-pod results remain historical, not mixed into this A/B.

## What the traces establish

1. **A real integration pathology was removed.** The original grouped path
   constructed CUDA offsets directly from a CPU list: 300 stream synchronizations
   per GPU per FB, including original/recomputed forwards. On EP8 these calls
   occupied 1.095–1.247 s of host API residence. Pinned CPU staging + nonblocking
   H2D reduces that count to **zero on every rank in both layouts**. The
   microbenchmark prepared offsets outside its timed region and missed this cost.
   Host wait residence is not automatically recoverable step time.
2. **The current grouped kernels are not faster in the full trainer.** Rank 0
   expert GPU union is 2.120 s with TE versus 2.415 s with fixed grouped-MM at EP1;
   at EP8 it is 1.244 s versus 1.422 s. Kernel inventory confirms actual
   PyTorch/CUTLASS grouped kernels, not a fallback. Lower host setup overhead
   therefore did not deliver the isolated microbenchmark's end-to-end gain.
3. **Weight-gather readiness is not the same as zero communication cost.**
   Every EP1 capture has 149/149 comparable steady expert gathers ready by the
   preceding block's compute end on every rank. One initial backward refill
   has no comparable predecessor. These gathers still consume GPU/memory/link
   resources. EP8 has expert-DP size 1 and no expert-weight FSDP gather.
4. **Allocator churn is not the persistent explanation.** The initial EP1
   trace had 0.632 s of allocation-coincident GPU idle on rank 1. The repeat's
   maximum was 0.012 s, but 1.22–1.75 s/rank of recorded GPU idle remained and
   unprofiled TPS stayed similar.
5. **Tensor Active is not useful model throughput.** TE exclusive expert
   samples show 53.2–58.6% at EP1 versus 91.2–92.4% at EP8. Fixed grouped-MM
   raises EP1 to 77.3–83.3% without a speedup. Only 43–50% of fixed EP1 expert
   samples are category-exclusive; all EP8 expert samples are exclusive.
   Separating padding, weight traffic, scheduling and clock effects needs
   further controlled kernel work; these samples alone do not prove that split.

Representative rank 0 GPU category unions (seconds per traced FB):

| Case | Expert GEMM | Attention-tagged | Expert gather | CP collective | Dispatch/combine path | GPU idle |
|---|---:|---:|---:|---:|---:|---:|
| EP1 TE repeat | 2.120 | 4.414 | 4.466 | 0.500 | 0.485 | 1.454 |
| EP8 TE | 1.244 | 4.020 | 0 | 0.342 | 2.950 | 0.232 |
| EP1 grouped async | 2.415 | 4.697 | 5.011 | 0.707 | 0.219 | 0.744 |
| EP8 grouped async | 1.422 | 4.055 | 0 | 0.410 | 3.210 | 0.261 |

These columns overlap and **must not be summed**. Dispatch/combine includes
packing/metadata; its EP1 value is not expert network communication. Collective
residence also includes waiting for peers. Attention-tagged work is not a claim
that every generic backward projection GEMM is separately attributed.

For the fastest case, EP8 TE, the five largest rank 0 **exclusive observed costs**
are attention 3.952 s, dispatch/combine 2.796 s, expert GEMM 1.244 s, other compute 0.702 s
and other GEMM0.556 s. These are investigation priorities, not guaranteed
recoverable milliseconds. The analyzer reports every rank, not just rank 0.

## Reproduction, quality and handoff

- Persistent FSDP buffers, prefetch and fast import on; memory-efficient head
  off; head chunk 4096; CUDA graphs off (zero graph launches observed).
- This sweep does **not** enable `BT_FREEZE_GC_AFTER_WARMUP`; it is unset/off,
  unlike the earlier PR reproduction block. Allocator uses expandable segments
  and garbage-collection threshold 0.95. Run scripts retain the exact settings.
- Checkpoint: `zai-org/GLM-5.3@187fb9fff6319062325ff825627ef6db084d9bc6`.
  Same old venv: Torch 2.11.0+cu130 / TE 2.16.0; built-in nsys 2025.3.1.
- Baseline source: trainers `13137ef1a` / Bridge `60b1570f` / Core `cf81782b2`.
  Fixed source: [PR1355](https://github.com/basetenlabs/trainers/pull/1355)
  at `57b9a3ef4`, [Bridge PR84](https://github.com/basetenlabs/Megatron-Bridge/pull/84)
  at `a3438227` and [Core PR76](https://github.com/basetenlabs/Megatron-LM/pull/76)
  at `fe3976282`. Only offset upload and its dependency pins changed.
- CUDA/NVTX timing capture and a separate all-GPU 10 kHz metrics capture per case.
  Collection is off during controls, although launcher/NVTX wrappers remain.
  Timing overhead versus control median is 3.06%/0.71% for TE EP1/EP8 and
  1.96%/−0.05% for fixed grouped EP1/EP8. Metrics passes can perturb more.
- All eight ranks captured with 100% kernel/runtime correlation. Nsight warns
  that some CUDA/NVTX events may not have been collected; anchors/correlation
  do not prove zero loss. Exported GPC-clock units are flagged as inconsistent.
- All runs completed with finite loss/gradient norms. No tests added or modified.
  Trainer `make check` and changed-Core checks passed. Bridge read-only all-file
  checks exposed pre-existing unrelated formatting/import errors; none were fixed.
  This is a five-control systems screen, not convergence or production validation.
- All raw reports/SQLite exports are on the Mac and SHA256-verified. Their
  manifests, configs, scripts, numeric results and reports are committed in the
  separate artifact repository; the large raw binaries are intentionally not in Git.
  No experiment artifacts are tracked in trainers. The final trainer is stopped;
  the supplied devbox remains available.

One-command analysis: `python3 workflow.py analyze CASE/timing.sqlite --benchmark CASE/benchmark.json`.
Use `workflow.py collect CASE --host tj-q9exk9w --analyze` for verified retrieval.
See [README](README.md), [TE comparison](refreshed-te-comparison.md),
[EP1 upload ablation](ep1-offset-upload-comparison.md),
[EP8 upload ablation](ep8-offset-upload-comparison.md), and
[fixed-layout comparison](final-async-comparison.md).
