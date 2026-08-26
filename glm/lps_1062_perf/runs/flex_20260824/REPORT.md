# LPS-1062 GLM-5.2 Flex report

## Verdict

Both Flex backends work on the requested 2x8 B300 devbox with GLM-5.2 at TP1/PP2/EP8/CP8.

- **HybridEP at 16 SMs is the winner:** 837.55 tok/s/GPU, **+8.86%** over alltoall.
- **DeepEP works but is not competitive:** its best tested setting, 20 SMs, reaches 734.86 tok/s/GPU, **-4.49%** versus alltoall.
- HybridEP also reduces PyTorch allocator reserve and improves PP0 memory balance. Neither Flex backend changes the fleet live-memory ceiling because PP1 remains dominant.

## Branches

Both branches started at `revert-1064-vram-release` (`6d66d40a6`) and are pushed.

| Branch | Pushed head | Changes |
|---|---|---|
| `lps1062-deepep-test` | `e974ba793` | B300 `L20D` name shim; explicit Flex backend and SM-count config |
| `lps1062-hybridep-test` | `bbdaa1410` | B300 `L20D` name shim; explicit Flex backend and SM-count config |

The runtime remains backward compatible: `moe_token_dispatcher="flex"` without a backend still selects DeepEP. HybridEP is selected explicitly with `moe_flex_dispatcher_backend="hybridep"`.

Validation:

- Repository pre-push checks passed on both branches, including all package type checks.
- `models/tests/test_control.py`: 16 passed on each branch.
- GPU name shim tests: 3 passed on each branch.
- On-box validation imported both `deep_ep.Buffer` and `HybridEPBuffer`, confirmed the `L20D` device is exposed as B300, and verified provider mutation for both backends and SM counts.

### Productionization status

HybridEP PR #1150 was subsequently completed at `fb605182a` so the production artifact matches the experiment:

- GLM-5.2 B300 131K publishes `flex + hybridep + 16 SMs` through the golden config and opaque runtime payload.
- The unvalidated B300 256K row remains alltoall.
- The production image sets `CUDA_HOME`/`CUDA_PATH`, retains `nvcc` for HybridEP JIT, and smoke-tests `Buffer` plus `HybridEPBuffer`.
- The existing pinned CUDA 13 `sm_103a` wheel is retained; no multinode wheel rebuild is required because each PP2/EP8 HybridEP communicator is node-local.
- Production image builds passed and published both lanes: `baseten/trainers-server:lps1062-hybridep-fb60518` and `baseten/trainers-server:lps1062-hybridep-fb60518-cu13` ([CI run](https://github.com/basetenlabs/trainers/actions/runs/32880233823)).
- DeepEP PR #1151 was intentionally left unchanged.

## Debug model

The debug checkpoint has exactly one dense and one MoE layer. Only layer-related architecture fields were changed:

- `num_hidden_layers=2`
- `first_k_dense_replace=1`
- `mlp_layer_types=["dense", "sparse"]`
- `indexer_types=["full", "full"]`
- `num_nextn_predict_layers=0`

GLM-5.2 dimensions are preserved: hidden size 6144, intermediate size 12288, MoE size 2048, 256 routed experts, top-k 8, 64 attention/KV heads, Q/KA LoRA ranks, head dimensions, vocabulary 154880, and position limit 1048576.

### Debug EP8 results

All debug runs use 8 B300 GPUs, sequence length 8192, eight datums, one traced step, and three untraced controls.

| Backend | SMs | tok/s/GPU | Delta vs alltoall | Control FB | Peak reported memory |
|---|---:|---:|---:|---:|---:|
| alltoall | n/a | 7,558 | baseline | 1.084 s | 28 GiB |
| DeepEP | 16 | 8,419 | +11.4% | 0.973 s | 29 GiB |
| DeepEP | 20 | **9,604** | **+27.1%** | **0.853 s** | 29 GiB |
| HybridEP | 16 | 9,912 | +31.1% | 0.826 s | 27 GiB |
| HybridEP | 20 | **9,926** | **+31.3%** | **0.825 s** | 27 GiB |
| HybridEP | 32 | 9,420 | +24.6% | 0.870 s | 27 GiB |

Debug conclusions:

- DeepEP benefits from 20 versus 16 SMs.
- HybridEP payload does not improve above 16 SMs. Sixteen and twenty are tied within 0.14%; 32 SMs is 5.2% slower and more variable. Use 16 SMs.
- Debug numerical parity is tight: maximum backend loss spread is 2.86e-6 for DeepEP and 1.91e-6 for HybridEP; gradients match within tiny relative error.
- No backend-specific token drops, NaN/Inf, OOM, CUDA/NCCL failure, or PyTorch-managed leak was observed.

HybridEP initially failed its JIT compile because the devbox exported neither `CUDA_HOME` nor `CUDA_PATH`, producing `/bin/nvcc -I/include -L/lib64`. Exporting `/usr/local/cuda` fixed it; the failed log is retained.

## Full GLM-5.2 benchmark

All completed full runs use:

- `zai-org/GLM-5.2-FP8`
- 16 B300 GPUs, TP1/PP2/EP8/CP8/ETP1/DP1
- sequence length 131,072
- four datums, 524,288 tokens per step
- LoRA rank/alpha 32
- `profile_driver_new.py`
- one untraced warmup, one traced step, three untraced controls

| Backend | SMs | tok/s/GPU | Delta vs alltoall | Control FB mean | FB CV |
|---|---:|---:|---:|---:|---:|
| HybridEP | 16 | **837.55** | **+8.86%** | **39.124 s** | 0.607% |
| alltoall | n/a | 769.38 | baseline | 42.590 s | 0.837% |
| DeepEP | 20 | 734.86 | -4.49% | 44.591 s | 1.110% |
| DeepEP | 16 | 708.37 | -7.93% | 46.258 s | 0.140% |

HybridEP improves cluster throughput from 12,310 to 13,401 tok/s. Its additional first-FB warmup cost versus alltoall is repaid after roughly 12 steady steps.

## Trace findings

Rank-0 full traces explain the ordering.

| Backend | Raw dispatcher | Dispatcher overlap | Exposed dispatcher | PP wait |
|---|---:|---:|---:|---:|
| HybridEP 16 | **7.008 s** | 0.000 s | **7.008 s** | **9.692 s** |
| alltoall | 11.556 s | **0.735 s** | 10.822 s | 10.305 s |
| DeepEP 20 | 12.375 s | 0.000 s | 12.375 s | 11.437 s |
| DeepEP 16 | 13.044 s | 0.000 s | 13.044 s | 12.079 s |

- HybridEP payload is only 1.548 s, permutation/metadata is 0.735 s, and device polling is 4.725 s.
- DeepEP 20 reduces payload to 5.087 s but spends 6.090 s in notify/polling and 1.197 s in permutation/layout. None overlaps compute.
- Raising DeepEP from 16 to 20 SMs cuts payload by 18.3%, but notify/polling grows by 18.9%; further SM tuning cannot beat HybridEP's non-payload floor.
- Faster MoE dispatch also reduces PP readiness bubbles. Dispatcher and PP-wait differences account for more than 98% of HybridEP's traced advantage.
- The remaining HybridEP opportunities are its 4.725-second `device_sync_kernel` polling tail and the four exposed PP readiness bubbles. No unrelated optimization was attempted.

## Memory

All 16 rank snapshots were retained for each full run.

| Metric | alltoall | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---|---:|---:|---:|---:|
| Mean live peak | 146.42 GiB | 144.30 GiB | 144.31 GiB | **143.46 GiB** |
| Fleet max live peak | 153.08 GiB | 153.06 GiB | 153.06 GiB | 153.06 GiB |
| Mean reserve | 150.61 GiB | 151.20 GiB | 151.30 GiB | **147.65 GiB** |
| Fleet max reserve | 157.26 GiB | 160.06 GiB | 160.18 GiB | **156.50 GiB** |

- HybridEP lowers mean reserve by 2.96 GiB/rank and fleet max reserve by 0.77 GiB versus alltoall.
- DeepEP raises fleet reserve despite lowering PP0 live allocations.
- PP1 remains the approximately 153 GiB live ceiling under every backend.
- Every one of 64 full-run rank snapshots has zero live-byte growth between steady snapshot markers. No PyTorch-managed leak or OOM occurred.
- Direct CUDA/NVLink allocations owned outside PyTorch's allocator are not visible in these snapshots.

## Correctness and limitations

- All variants complete five optimizer steps with finite loss and gradients.
- Each step reports the same 524,288 tokens and 524,284 loss tokens.
- No completed run logs traceback, OOM, CUDA/NCCL failure, NaN/Inf, or backend-specific token drop.
- Full-run scalar loss spread is at most 0.016%; grad-norm spread reaches about 5.3%. This is operational stability evidence, not bitwise/tensor-level equivalence.
- Each Kineto trace contains one rank-0 step. Multi-rank traces would better identify which rank causes polling tails.
- Three control windows establish a clear backend ordering but are not a long-duration production soak.

## Operational findings

- Flex full-model startup is much longer than alltoall because DeepEP/HybridEP buffers and HybridEP JIT code are initialized lazily during the real startup warmup.
- The inherited devbox trainer and profiler children survived Slurm cancellation during early iterations. Later runs use verified empty process/GPU state and an artifacted stop helper that sweeps trainer, torchrun, and profiler-spawn children inside the allocation before cancellation.
- The devbox's original dirty checkout was not modified. Separate remote worktrees were used for the pushed branches.

## Recommendation

Use **HybridEP with 16 SMs** for GLM-5.2 131K at TP1/PP2/EP8/CP8 on 2x8 B300.

Keep alltoall as the fallback. Do not select DeepEP for this topology unless HybridEP is unavailable; if DeepEP is required, use 20 SMs, but expect about 4.5% lower TPS and higher allocator reserve than alltoall.

## Artifacts

- Original task: `../../ORIGINAL_PROMPT.md`
- Notebook: `../../NOTEBOOK.md`
- Configs and launch controls: `configs/`, `ctl/`, `tools/`
- Raw run artifacts: `artifacts/`
- Full trace analysis: `FULL_FLEX_TRACE_COMPARISON.md`
- Full outcomes/memory: `FULL_FLEX_OUTCOMES.md`
- DeepEP trace detail: `FULL_DEEPEP_TRACE_ANALYSIS.md`
- Debug analyses: `DEBUG_TRACE_ANALYSIS.md`, `DEBUG_MEMORY_ANALYSIS.md`, `DEBUG_CORRECTNESS.md`
- HybridEP debug analyses: `HYBRIDEP_TRACE_ANALYSIS.md`, `HYBRIDEP_DEBUG_ANALYSIS.md`
