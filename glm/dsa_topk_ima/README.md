# GLM-5.2 train-loop restart (incident e3m916q): cuDNN DSA indexer top-k OOB

**Status:** root cause found and fixed. Trainers PR
[#814](https://github.com/basetenlabs/trainers/pull/814) (carried image patch),
upstream PR [NVIDIA/cudnn-frontend#445](https://github.com/NVIDIA/cudnn-frontend/pull/445).

## Incident

- 2026-07-25 00:37:32Z: Loops trainer `baseten-trainer-e3m916q-multinode`
  (GLM-5.2-FP8 LoRA, 4x8 B200, TP1/PP1/EP32/CP32, image
  `baseten/trainers-server:main-1d2de6f`), rank 4 (GPU 4 on
  `b200-worker-striking-tapir`) raised `CUDA error: an illegal memory access
  was encountered` mid-`forward_backward`; NCCL CONTEXT_PARALLEL_GROUP watchdog
  died first; torch elastic SIGTERM'd 8 local ranks; LWS `RecreateGroupOnPodRestart`
  rebuilt the group. DCGM: Xid 43 on that GPU, zero ECC errors (software, not
  hardware). One crash in 5 h; self-healed; later handled 73-86k-token batches.
- Crash batch: max datum length 55,111 tokens (preceding batches ~36-43k).
- Slack: https://basetenlabs.slack.com/archives/C0BE8KE102E/p1784939978170129

## Root cause (mechanism)

All in the cuDNN-frontend 1.26.0 CuTe-DSL radix top-k kernel
(`cudnn/deepseek_sparse_attention/indexer_top_k/`), used by GLM-5.2's fused
DSA indexer -> top-k -> FlashMLA path (`fused_indexer_sparse_attn` in the
vendored Megatron-LM `dsa_cudnn_kernels.py`):

1. Each score row is read in 4096-element vector tiles; the final partial
   tile's out-of-bounds lanes are filled with `-inf` by `_fill_oob` so the
   predicated copy is memory-safe. But the histogram and candidate-collection
   loops in `indexer_top_k_varlen_util.py` then iterate **every** fragment lane
   with no bounds check, counting phantom `-inf` lanes as real elements.
2. `to_coarse_key` derives the coarse radix bin from the **fp16 conversion** of
   the fp32 score, so every score below fp16's -65504 minimum collapses into
   the fp16 `-inf` bin — the same bin the phantom lanes occupy.
3. Harmless while a row's top-k threshold sits above that bin (phantoms sort
   last, dropped). But when the threshold lands **inside** the `-inf` bin —
   fewer than `top_k` (2048) values above ~-65504 — the candidate list becomes
   `real + up-to-4080 phantom`:
   - `large_occupancy` compile (>148 rows; always the case at CP32: 862-row
     calls): per-row candidate buffers (512-entry smem + `num_cols` gmem)
     overflow -> OOB shared/global writes -> `cudaErrorIllegalAddress` (Xid 43).
   - any compile: phantom lanes can be selected as winners -> silently
     out-of-range top-k indices (observed idx 8191 into a 4310-wide row).

The incident row (rank 4 front segment): seq_len 4122 within a 4310-wide
chunk; 2242 of 4122 scores below -65504 (fp16-collapse), only 1880 above, so
the 2048-th largest value sits inside the collapsed bin. Candidate flood =
2242 real + 4072 phantom = 6314 > 4822 buffer capacity.

**Architecture/CUDA-version independence:** the bug is in the Python-level
CuTe-DSL kernel logic (missing bounds check + fp16 collapse), JIT-compiled
identically for every SM90+ arch. Same input crashes B200 (sm100) and B300
(sm103) at the same call/iteration/seed — cross-arch determinism is itself
evidence the fault is algorithmic, not scheduling/hardware. Not specific to a
CUDA toolkit version.

## Why it looked the way it did in prod

- One crash in 5 h, longer sequences fine afterwards: the trigger is the
  per-row score **distribution** (a row with <2048 values above ~-65504 and a
  partial final tile), not the shape. Most batches never produce such a row.
- Watchdog-first signature: the faulting kernel is async; the error surfaced at
  the next NCCL sync, not at the faulting op.

## Evidence

### Deterministic reproduction (incident image, incident shape)

`scripts/repro_glm52_dsa_sparse_backward.py` mirrors one CP rank's exact
tensors (padding to 2*cp, zigzag front/back chunk selection, KV all-gather
order, compact causal top-k metadata) and invokes the production fused block
`fused_indexer_sparse_attn` (indexer -> top-k -> FlashMLA fwd -> cuDNN sparse
bwd). Run on 1xB200 and 1xB300 with the incident image.

- `--doc-lengths 55111 --cp-rank 4 --mode full-indexer` at realistic scale
  (std=0.02): 400+ iterations clean — shape alone is not the trigger.
- Same shape with indexer-activation scale std=8.0 (heavy fp16-collapse):
  crashes at a **deterministic iteration per seed, identical on B200 and B300**
  (seed 1234 -> iteration 20; seed 7 -> iteration 19; seed 99 -> survives 40).
- `CUDA_LAUNCH_BLOCKING=1` attributes the fault to the top-k kernel:

```
File ".../cudnn/deepseek_sparse_attention/indexer_top_k/indexer_top_k_decode_varlen.py", line 710, in cute_dsl_topk_wrapper
    compiled_kernel(
RuntimeError: CUDA Error: cudaErrorIllegalAddress
```

### Input capture + standalone replay

`scripts/capture_topk_crash.py` monkeypatches `_indexer_top_k_one_chunk` to
dump inputs before each call (post-fault CUDA copies are impossible — the
context is poisoned). The crashing call (front segment, iteration 20, seed
1234, std 8.0) is `data/topk_call_41.pt`: scores (862, 4310) fp32, seq_lens
arange(3449, 4311), top_k=2048. `scripts/replay_topk.py` on that dump crashes
in a fresh process; the neighbouring calls' dumps pass.

### compute-sanitizer (pre-patch)

26 x `Invalid __shared__ write of size 2 bytes` (the per-row smem candidate
buffers are 2-byte `Uint16`). The gmem spill overflow is intra-allocation
(invisible to memcheck) and cascades garbage candidate indices into the smem
over-selection. Post-patch: zero invalid accesses.

### Row bisection + perturbation (`replay_slice.py`, `replay_perturb.py`)

- Crash window isolated to a single row: 149-row slices crash iff they contain
  row 673 ([673,822) crashes; [674,823) passes).
- Perturb row 673: tiny noise -> crash; sorted/shuffled -> crash;
  values x0.5 (fewer fp16-collapse) -> **no crash**. Trigger = value
  distribution, not order or exact bit patterns.
- 148-row slices (small-occupancy compile, 8192-entry smem fits the flood):
  never crash — but silently emit out-of-range indices (idx_max 8191).

### Synthetic minimal repro (no incident data)

`scripts/repro_cudnn_dsa_indexer_topk_oob.py`: 149 rows x 4310; trigger row
length 4122 with 1122 small values + 3000 values in [-66000, -200000].
Pre-patch: `cudaErrorIllegalAddress` (distinct values) / out-of-range indices
(identical values). Post-patch: exact parity with `torch.topk`.

### Parity + stress, post-patch (B200 and B300)

- `scripts/parity_topk.py`: 12/12 randomized trials vs `torch.topk` — rows
  {37,149,200,300,862}, cols {4097,4310,4311,8192,51720}, distributions:
  small-normal, large-normal (std=8), 60%-below--65504, 80%-exact-zero ties.
- All previously crashing configs (5 seeds x 40 iters, incident shape,
  std=8.0): 200/200 iterations clean.
- Incident-shape soak at realistic scale: ranks 0/4/31 x 100 iters, two-datum
  and near-max (255,360-token) packs: clean, finite gradients.
- Microbench (`scripts/bench_topk.py`), (862, 51720) k=2048: 0.115 -> 0.129
  ms/call (+12% on a 0.1 ms kernel).

### A/B/A causality proof (fresh env, exact PR patch artifact)

On a clean B300 box with a minimal venv (torch 2.11.0+cu128,
nvidia-cudnn-frontend==1.26.0, nvidia-cutlass-dsl==4.5.2, apache-tvm-ffi==0.1.9
— the trainer image's pins; pristine util file md5-verified byte-identical to
the incident image):

| leg | kernel file | result |
| --- | --- | --- |
| A | pristine 1.26.0 (md5 b74f18f1...) | `cudaErrorIllegalAddress` |
| B | + `server/patches/cudnn-frontend-1.26.0-dsa-indexer-topk-oob.patch` via `patch -p1` (md5 4f86b98c...) | PASS (torch.topk parity) |
| A' | patch reversed via `patch -R` (pristine md5 again) | `cudaErrorIllegalAddress` |
| B' | patch re-applied | PASS |

## The fix

Skip OOB lanes (`col < aligned_size`, the same bound the copy predicate
computes) in the three vectorized histogram/collection loops of
`indexer_top_k_varlen_util.py`. Scalar prologue/leftover loops are already
exact-bounded; selection semantics for real elements are unchanged.

- Trainers: carried build-time patch (server/Dockerfile), regression test
  (`server/tests/unit/dp_worker/test_cudnn_dsa_indexer_topk.py`, fails
  pre-patch / passes post-patch), manual repro
  (`server/scripts/repro_cudnn_dsa_indexer_topk_oob.py`). The Dockerfile step
  fails the build on wheel-version change so the carried patch cannot rot.
- Upstream: NVIDIA/cudnn-frontend#445 (same patch rebased on `develop` + L0
  regression test in their suite).

## Not the cause (ruled out)

- Odd-top-k JIT assert (NVIDIA/cudnn-frontend#406, fixed upstream by #407):
  the incident shape's effective top_k stays 2048 (even) everywhere, and the
  incident image already carried the Megatron-side fallback
  (basetenlabs/Megatron-LM d3932e757).
- Packed-CP causal-offset bug (basetenlabs/Megatron-LM#16): already in the
  incident image.
- Hardware: Xid 43 with zero SBE/DBE ECC; reproduces identically on other
  nodes/architectures.
- int32 scratch overflow (cudnn-frontend#312): 7.4M elements << int32 max.
- Sequence length per se: longer sequences (73-86k) ran fine after the crash;
  400+ iterations of the exact shape pass at realistic score scales.

## Follow-ups

- Audit of the surrounding cuDNN DSA path (indexer forward, compactify /
  local_to_global, sparse-attention backward, Megatron glue) for more latent
  issues is in progress as a separate workstream.
- Sampler/inference is not exposed: `sampler/uv.lock` has no
  nvidia-cudnn-frontend; vLLM uses its own DSA implementation.
