# Handoff To Laplace

## User Contract

Read and execute this prompt verbatim:

`/Users/jackrao/Documents/trainers/prompts/g.md`

Key requirements:

- Work autonomously overnight; do not stop early.
- Put all evidence under this run directory.
- Maintain `WORKLOG.md`.
- Use `profile_driver.py`.
- Iterate with the 0d1m debug model at sequence length 131072, one datum,
  TP1/PP1/CP8/EP8/ETP1/DP1 on one 8xB300 node.
- Optimize the full 78-layer GLM-5.2 model; debug results are screens only.
- Start from tip of main.
- Push every profiled SHA and record it in the worklog.
- Create and keep updating a trainers PR.
- Report tok/s/GPU and MFU during progress. Include measured performance in
  later commit messages where possible; do not amend already-pushed commits.
- Use the `devbox-up` lifecycle scripts on `ssh tj-w5y89m3`.

## Current Local State

- Experiment root:
  `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm52_postmerge_optimization_20260829`
- Worklog exists at `WORKLOG.md`.
- Fresh worktree:
  `/Users/jackrao/Documents/trainers-wt-lps1062-postmerge-perf`
- Branch: `jackrao/lps-1062-postmerge-perf`
- Branch is pushed and tracks origin.
- Base SHA: `b1b28f8850bda62f70c4025b5df30555d4173e80`.
- All submodules are initialized.
- Pre-push `make check` passed after submodule initialization.
- No code changes and no PR yet because the branch currently equals main.
- Original `/Users/jackrao/Documents/trainers` contains untracked `prompts/`;
  do not modify or delete it.

## Main And Dependency State

- Tip main includes merged trainers #1222 at
  `b485c642b353d23b15116c574f6de1f018fe0a87`.
- Main pins merged Bridge #54 and MCore #67.
- Compiled native-FP8 optimization remains an open draft stack:
  - MCore #68, head `32ba8c0196d9b3729dd1e59f85a9b1ac5d66e185`
  - Bridge #55, head `9b258cd666480304a2896a29ae48ccf63bcba4c8`
  - trainers #1228, head `3e2131c252c8e2bf5f049bf7f34f8d8d686a3600`
- #1228 measured 1205.0 tok/s/GPU, 10.97% MFU, 13.596 s mean FB.
- Matched persistent-BF16 main measured 1216.1 tok/s/GPU, 11.07% MFU,
  13.473 s mean FB.
- TE-generic native FP8 measured 1117.7 tok/s/GPU, 10.17% MFU, 14.659 s.
- Thus compiled materialization is already validated at +7.82% over the generic
  runtime and is within 0.91% of persistent BF16, but it is not merged.

## Trace To Analyze/Use

Full DP1 trace:

`/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm52_native_blockwise_fp8_20260828/phase4_full_native_te_generic_runtime.pt.trace.json`

This is byte-identical to the 452 MB trace copied from `tj-w5y89m3`. It covers
one complete 15.168 s full-model DP1 step with all 78 checkpointed layers.

Related artifact directory:

`/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm52_native_blockwise_fp8_20260828`

Important files there:

- `PHASE4_FULL_MODEL.md`
- `optimized_runtime_full_steady10.json`
- `current_runtime_full_steady10.json`
- `main_runtime_full_steady10.json`
- `optimized_runtime_debug.pt.trace.json`
- `optimized_runtime_debug_steady10.json`
- all-rank memory profiles for optimized/current/main

## Corrected Trace Conclusions

Do not repeat the earlier mistake of treating inclusive CPU waits as additive
wall time:

- `aten::nonzero`: 8.417 s inclusive, but 7.997 s overlaps active GPU work;
  only 0.420 s coincides with GPU idle.
- `cudaStreamSynchronize`: 10.263 s inclusive, but 10.042 s overlaps active GPU
  work; only 0.221 s coincides with GPU idle.
- These idle portions overlap each other and cannot be added.

Critical-path interval accounting on the precompiled full trace:

| Class | Exclusive wall |
|---|---:|
| Copies/elementwise | 3.784 s |
| GEMM | 3.283 s |
| DSA attention/indexer | 2.826 s |
| HybridEP device sync | 1.620 s |
| HybridEP movement/permutation | 1.224 s |
| Other | 0.615 s |
| NCCL | 0.293 s |
| No-kernel gaps | 1.463 s |

HybridEP's device sync kernel has zero overlap with other GPU kernels, but it
may represent waiting for another rank rather than inefficient kernel code.
Capture all ranks before changing synchronization.

Full recompute evidence:

- 78 `CheckpointFunction` forwards total 4.389 s, 56.3 ms/layer.
- 78 checkpoint backwards total 9.615 s, 123.3 ms/layer.
- Backward includes full forward recomputation.

The two FP32 output-head SGEMMs total 0.925 s in the precompiled full trace and
are a promising contained optimization. Investigate BF16 tensor-core matmul
with FP32 loss accumulation before touching broad model precision.

## Prior PR #1197

PR #1197 is open and stacked on old work. Its 5.7% debug-model gain combines
no recompute and tensorwise FP8. It is not full-model acceptance evidence.
Potentially reusable knobs include HybridEP permutation/router fusion, trivial
router-group elision, shared-expert overlap, and scoped FP8, but screen each as
one variable against current main. Do not copy the 19-commit stack wholesale.

## Devbox State

- `tj-w5y89m3` is 2x8xB300; use one node.
- Previous trainer processes were stopped and GPUs were empty at handoff.
- The generated default devbox worktree is dirty from old experiments. Do not
  reset or overwrite it.
- Create/use an isolated exact-SHA checkout and a copied generated lifecycle
  script pointed at that checkout, following the `devbox-up` skill.
- A devbox global git `insteadOf` rule redirects Bridge/MCore GitHub URLs to the
  golden checkout. URLs without `.git` bypass it when an unmirrored PR SHA is
  required.

## Recommended Immediate Sequence

1. Benchmark fresh pushed base SHA `b1b28f885` on 0d1m with a stabilized window;
   report tok/s/GPU and MFU and copy result/config/log into this run directory.
2. Benchmark the full model baseline before accepting any debug gain.
3. Research output-head FP32 SGEMM and selective recompute in parallel.
4. Prefer a small training-equivalent output-head prototype as first new code.
5. Use #1228 as a known validated candidate, but keep its dependency-stack and
   private-TE-ABI risk explicit.
6. Push before every profile and record exact SHA in `WORKLOG.md`.

## Aborted Research Tasks

Three delegated research tasks were launched but aborted by the handoff:

- trace the output-head/logits/cross-entropy dtype path;
- assess memory-safe selective recompute with compiled native experts;
- evaluate which #1197 knobs have full-model evidence.

Restart those investigations directly or with subagents.
