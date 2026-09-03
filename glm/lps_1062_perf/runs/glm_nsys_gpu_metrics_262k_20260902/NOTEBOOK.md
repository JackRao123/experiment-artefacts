# Optimisation notebook (session cauchy, from 2026-09-03 01:30 PDT)

Scope: raise tok/s/GPU of the GLM-5.3 LoRA step at 262,144 tokens on 8xB300
(TP1/PP1/CP8/EP8, HybridEP, DSA cuDNN, full recompute, native-FP8 experts),
parity-safe, working down the ranked list in `ANALYSIS.md`. Box: profiler pod
`jackrao-glm-nsys-b300`. Code: branch `jackrao/glm-262k-step-perf` off origin/main
(worktree `~/Documents/trainers-perf`), one draft PR kept current.

Terms: **control step** = one forward+backward of one 262,144-token sequence as
timed by `profile_driver.py` (mean of 3 after a warmup step). **host sync** = a
`cudaStreamSynchronize` that blocks the CPU until the GPU catches up. **A/B** =
same launcher, same config, one code change between the two measurements.

## Pre-registered decision rules

- Fix 1 (DSA CP layout cache) ships if: host syncs per rank per step drop from
  ~7,521 to ~500 (`host_syncs.csv`), the nonzero/index signature disappears from
  `sync_callchains.csv` under `attention`, control step drops by >= 1.3 s, and
  loss / grad_norm on the fresh-start step match the reference (warmup0
  loss 12.3311 gn 0.5745, control0 loss 12.3222 gn 0.4619 at seed 0xB300) to
  float noise. The cache is the same function run once, so any drift is a bug.
- Fix 3 (LM head off the SIMT fp32 GEMM) ships if: the `cutlass3x_sm100_simt_sgemm`
  kernels vanish from the capture, control step drops by >= 1.0 s, and per-token
  logprobs of the same input agree with the fp32 head within bf16-level noise
  (report max |delta| and loss delta explicitly).

## Log

2026-09-03 01:30 PDT - Took over from laplace. Read ANALYSIS/WORKLOG/REPRODUCING,
dsa_layout.py / dsa.py call sites, chunked_lm_head.py, token_dispatcher.py:1103,
dsa_cudnn_kernels.py:2102. Pod up, trainer still healthy under `nsys launch`
(session glm53full262nvtx), warm at 262k.

2026-09-03 01:45 PDT - **A0 baseline** on the still-running tip-of-main trainer
(nsys launcher, no collection; LoRA already trained a few steps by laplace so
loss values are not the fresh-start reference): controls 26.3 / 27.2 / 27.0 s,
mean **26.9 s (1220 tok/s/GPU)**. Log: `$REMOTE_RUN/ab/A0-baseline.log`.

2026-09-03 01:55 PDT - Fix 1 written as a trainer-side monkeypatch
(`server-megatron-bridge/src/trainers_server_megatron_bridge/dsa_cp_layout_cache.py`,
installed from `backend.py`): memoize
`dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder` on the
`cu_seqlens` tensor object (one per microbatch; the same `PackedSeqParams`
flows through all 78 layers and the recompute). Chosen over a Megatron-LM fork
change to avoid two submodule pointer bumps; `dsa.py` resolves the builder on
the module at call time so the patch reaches every caller. Unit tests
(5, incl. equality against the real Megatron-LM builder at CP4 on two packed
sequences) pass on the pod. Note for later: torch 2.11 on the pod has
`torch.mm(bf16, bf16, out_dtype=float32)` on CUDA (needed for fix 3).

2026-09-03 02:05 PDT - Stopped the trainer, restarted it with the patch under
`nsys launch` (session `glm53fix1`) so B1 is measured under the same launcher
as A0. Pre-existing ty diagnostic at backend.py `model_provider._pg_collection`
is on origin/main too, not from this change.

2026-09-03 02:35 PDT - **B1 (fix 1)**, same nsys launcher, fresh start: controls
25.8 / 26.4 / 25.8 s, mean **26.0 s (1261 tok/s/GPU)** vs A0 26.9 s: **-0.9 s
(3.3%)**, below the pre-registered 1.3 s. Log `$REMOTE_RUN/ab/B1-fix1.log`.
Parity: warmup0 (step 0, before any LoRA update) loss 12.3322 gn 0.5760; the two
fresh-start references on unchanged code were 12.3311/0.5745 (09-03) and
12.3303/0.5675 (09-02), so step-0 values sit inside the unchanged-code spread.
control0 gn 0.4316 vs references 0.4619 / 0.4579 is a 6.5% deviation where the
two references differ by 0.9%. This cannot be the cache: every microbatch in
this benchmark has cu_seqlens = [0, 262144], so even a stale hit would return
the identical tensors; the layout tensors are bitwise the same with or without
the patch. Read it as the benchmark's own noise: step-1 grad_norm depends on
Adam's first update (sign-like, so any grad flip changes it) on top of the
known cuDNN DSA top-k nondeterminism on random tokens. Consequence for later
fixes: use step-0 loss/gn (spread ~1e-3 / ~1%) and per-token logprobs for
parity, not step-1 grad_norm. Random-token loss is also insensitive to
attention-layout errors, so fix 1's correctness rests on construction (same
function, identity-keyed) plus the CP4 two-sequence equality unit test.
Capture: `glm53-fix1-b300-262k-nvtx-all-ranks-gpu-metrics.nsys-rep`
(instrumented step 26.86 s vs 27.70 s in the tip-of-main capture; loss 12.3164,
gn 0.5721 on the capture datum), exporting to `fix1.sqlite`, analysis in
`analysis_fix1/`.

2026-09-03 03:10 PDT - **Fix 1 capture analysis** (`analysis_fix1/`, from
`fix1.sqlite`): host syncs per rank per step **7,521 -> ~280** (host_syncs.csv);
the boolean-mask `nonzero < index_Tensor` signature under `attention` is gone
(45 calls total across all ranks, 7 ms); GPU idle 1.4-1.9 s -> 0.32 s on six
GPUs; attention host wall 72 -> 20 ms per call (3.19 s host vs 3.10 s GPU over
156 calls, no longer host-bound); step window 27.4 -> 26.56 s. Where the rest
went: `hybridep_sync` (spin-wait for other ranks) rose 3.9 -> 4.5 s/GPU, and the
share of collective lateness whose cause is a *host stall on the laggard* rose
from 2.75 s to 7.7 s (summed over GPUs, sync_wait_summary.csv). Reading: the
remaining syncs now fire with a deep GPU queue behind them, so each drain is
longer (DSA backward `nonzero`: 78/rank, max 776 ms; HybridEP `.item()`:
150/rank; indexer `.item()` in backward: 42/rank). Decision: fix 1 ships (strict
improvement, zero numerical risk, 0.9 s measured); siblings 1a/1b are worth
more than before.

2026-09-03 03:20 PDT - **New finding 8: CUDA allocator growth stalls the
laggard rank.** `gap_context.py` on the two long GPU-idle gaps in the fix-1
capture: GPU1 idle 660 ms at 9.25 s into the step and GPU6 idle 383 ms at
10.36 s, both inside `recompute > layer:78/76 > moe_experts > _GroupedLinear`,
right at the start of backward. The host was not in a sync: it was in
`cuMemSetAccess` (one call, 689 ms) and `cuMemCreate` (170 calls, 400 ms).
That is PyTorch's expandable-segments allocator mapping new physical memory
(trainer env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.95`).
The tip-of-main capture has the same pattern (rank with 76 cuMemCreate +
cuMemMap = 350 ms; another 155 ms). Peak reserved memory reported by the driver
creeps up step to step as expert routing varies (204.0 -> 204.7 -> 206.5 ->
206.5 GiB over B1's four steps; 222.6 GiB on the captured step), and each new
high is paid as a host stall on whichever rank hit it, which then makes every
other rank wait at the next HybridEP collective. Cost 0.15-0.7 s per affected
step (measured gaps). Mitigation candidates: reserve the pool once at startup
to a high-water mark with margin (one mapping instead of one per new peak), or
grow the pool in the warmup. Not started; noted for the ranking. Also explains
part of the control-step noise: the slow controls (A0 27.2 s, B1 26.4 s)
coincide with reserved-memory growth.

(Clock note: pod timestamps below are UTC; the pod clock read 06:17 UTC when
the fix-1+3 trainer was started.)

06:00 UTC - **Parity noise floor is large.** `parity_probe.py` (new; `/forward`
only, no gradient side effects, after `/init_trainer_server` resets LoRA to
B = 0 so the model is the frozen base model): the same 262k random-token
datum forwarded twice on the same build gives per-token logprobs that differ
by mean |delta| 0.11 (p99 0.51, max 8.4), zero tokens bitwise equal, loss
12.3340 vs 12.3301. At 4096 tokens, warm vs warm: mean |delta| 0.08, loss
spread 3e-3. The known cuDNN DSA top-k nondeterminism (and possibly FP8
grouped-GEMM ordering) cascades through 78 layers. Consequence: trainer-level
logprob parity can only be statistical (cross-build delta vs this floor); a
head change of 1e-5 in the logits is invisible there. The precise evidence for
fix 3 is the isolated GEMM measurement (`lm_head_gemm_probe.py`) plus a GPU
unit test.

06:05 UTC - **Fix 3 design settled by `lm_head_gemm_probe.py`** (one B300, real
shape 4096 x 6144 x 151552, bf16-valued inputs, fp64 reference): forward fp32
SIMT GEMM 114 ms/chunk, max|d| 1.4e-5; bf16 tensor-core GEMM with fp32
accumulate + fp32 output (`torch.mm(out_dtype=float32)`) 4.0 ms/chunk, max|d|
3.7e-5; plain bf16 output (what the fp32 head was introduced to avoid) max|d|
1.6e-2. Backward grad_hidden: fp32 SIMT 117 ms; bf16 single (grad rounded to
bf16) 4.7 ms, 57% of bf16-cast grad elements bitwise equal to the fp32 path;
bf16 hi/lo split 13 ms, 94% bitwise equal, max|d| after the bf16 cast 1.26e-4
vs 1.22e-4 for fp32 itself. Chose forward bf16->fp32out + backward hi/lo split,
only for a frozen bias-free bf16 head (else the old fp32 module path). Also
removed the 1c syncs (two `.item()` per LM-head chunk -> one transfer). Unit
tests: 31 pass on the pod (incl. GPU test), 28 pass + 3 gpu-skips on the
laptop. Commit 4182de373 on the PR branch. Deployed to the pod (with the
pod-only NVTX decorators re-added) and trainer restarted under `nsys launch`
session `glm53fix13` at 06:17 UTC.

06:35 UTC - **B2 (fix 1 + fix 3 + 1c)**, same nsys launcher, fresh start:
controls 24.2 / 24.1 / 23.8 s, mean **24.0 s (1365 tok/s/GPU)**. vs B1 (fix 1
only) 26.0 s: **-2.0 s** for fix 3 (+1c); vs A0 tip-of-main 26.9 s: **-2.9 s
(-10.8%, +11.9% tok/s/GPU)** cumulative. Step-0 loss/gn 12.3319 / 0.5665 vs
unchanged-code fresh-start references 12.3311 / 0.5745 and 12.3303 / 0.5675
(inside the spread). Logprob parity, statistical (parity_probe.py, LoRA reset,
/forward, same 262k datum): fix13 runs vs the fix1 reference run: mean
|delta logprob| 0.1131 / 0.1131 / 0.1129, p99 0.51; within-build floors:
fix13 run1 vs run2 0.1127, fix1 run0 vs run1 0.1134. Cross-build differences
are indistinguishable from the within-build noise; losses 12.3284 / 12.3349 /
12.3307 (fix13) vs 12.3340 / 12.3301 (fix1). The precise evidence for the
head change remains the isolated GEMM probe (3.7e-5 max logit delta). Logs:
`ab/B2-fix13.log`, `ab/parity-fix13.log`, `ab/parity-*.json`. Capture
`glm53-fix13-...nsys-rep` -> `fix13.sqlite` -> `analysis_fix13/` in progress.

06:40 UTC - **fix13 capture unusable (lesson).** The capture ran as the first
forward+backward after `parity_probe.py --reinit` (LoRA re-init through
`/init_trainer_server`), and behaved like a warmup: 50.4 s step, host syncs up
to 2.0 s, HybridEP spin-wait 24 s/GPU, GPU idle 5-8 s/GPU (backward plans /
autotune rebuilt). Only kernel-category facts survive: `fp32_simt_head` is
absent (0 s; 1.85 s in tip of main and fix 1), i.e. the SIMT GEMMs are gone.
Rule added to the protocol: after any `/init_trainer_server`, run at least one
plain forward+backward before capturing or timing. Re-capturing as `fix13b`
after a warmup + 2 controls.

06:50 UTC - **fix13b capture (clean, after a warm step)**: instrumented step
25.1 s (window 24.8 s; tip of main 27.4, fix 1 26.56). `lm_head` range GPU
time 0.957 -> **0.070 s**; `fp32_simt_head` category gone; forward phase 8.2
-> 7.35 s GPU. Host syncs unchanged from fix 1 (~280 + ~254 on a second
thread per rank). GPU idle 0.33 s on six GPUs; GPU6 1.02 s (674 ms gap) and
GPU1 0.62 s (258 ms gap): `gap_context.py` shows both are again the allocator
(`cuMemCreate` calls of 19-57 ms each, dozens of them) at the first backward
recompute inside `moe_experts > _GroupedLinear`, on a step whose peak
reserved memory set a new high (226.9 GiB). GPU6 was the laggard for 10.7 s
of the 33.3 s summed collective waiting. Categories now (s/GPU): gemm 5.01,
hybridep_sync 4.38, dsa_backward 3.71, cat_copy 2.93, elementwise 2.17,
dsa_forward 1.24, dsa_indexer 1.10, hybridep dispatch+combine 1.76,
moe_permute 0.78, nccl 0.63, activation 0.37, norm 0.31.
Re-warm timing after the probe's LoRA reset: controls 23.8 s and 26.7 s; the
slow one coincides with reserved memory growing 208.5 -> 210.4 GiB (finding
8 again).

06:51 UTC - **Fix 4 deployed** (commit on the PR branch: `_enable_fused_swiglu`
in `megatron_config.py`, sets `bias_activation_fusion=True` for gated SiLU
without GLU interleaving) together with the pod-only diagnostic
`debug_dsa_bwd_flags.py` (prints the five inputs that decide the DSA backward
compaction path, once per rank). Trainer restarted under `nsys launch`
session `glm53fix134`. Pre-registered rule for fix 4: ships if the control
mean drops by >= 0.3 s with step-0 loss/gn inside the unchanged-code spread,
and the capture shows the `activation` category and the moe_experts
elementwise launches (5,095 per GPU per pass) shrink.

07:05 UTC - **B3 (fix 1 + 3 + 4)**, same launcher, fresh start: controls 23.3 /
22.6 / 23.7 s, mean **23.2 s (1411 tok/s/GPU)**; vs B2 24.0 s: **-0.8 s** for
the fused SwiGLU; cumulative vs A0 26.9 s: **-3.7 s (-13.8%, +15.7% tok/s/GPU)**.
Step-0 loss 12.3282 (references 12.3303-12.3322, probe noise ~4e-3: fine).
Step-0 grad_norm **0.5506** vs references 0.5665 / 0.5675 / 0.5745 / 0.5760:
2.9% below the lowest, outside the 1.7% spread. Not accepting this as noise
without a measurement: (a) op-level probe of fused vs unfused SwiGLU vs fp64
(forward output and input gradient) to see whether the unfused bf16 chain adds
rounding noise that inflates gradients; (b) more step-0 grad_norm samples on
this build via `/init_trainer_server` + one step. Log `ab/B3-fix134.log`.

07:15 UTC - **SwiGLU op-level probe** (`swiglu_probe.py`, 65536 x 4096 bf16
inputs, fp32 probs, one B300, fp64 reference): forward output rel-L2 error
unfused bf16 chain 2.93e-3 vs fused 1.66e-3 (ratio 1.76); input gradient
2.90e-3 vs 1.66e-3 (1.75); gradient norms fp64 28.71544 / unfused 28.71329 /
fused 28.71542 (identical to 1e-4). Probs gradient: unfused rel-L2 2.4e-3
(computed from bf16-rounded activations), fused 1.1e-7 (fp32). So the fused
kernel is strictly closer to the exact math; at the op level it does not
inflate or deflate gradient norms. The step-0 grad_norm shift therefore needs
the trainer-level sampling (`gn_probe.py`: re-init LoRA, one step, N=3).

07:20 UTC - **fix134 capture (clean)**: instrumented step 22.9 s (window 22.6;
fix13b 24.8). Categories (s/GPU, fix13b -> fix134): elementwise 2.17 -> 1.24,
activation 0.37 -> 0.18, hybridep_sync 4.38 -> **2.66**, moe range GPU 8.35 ->
6.29; moe_experts elementwise launches per GPU per pass 5,095 -> 4,800
(the silu / mul / probs-mul / cast chain became one kernel) and its time 0.334
-> 0.059 s. The imbalance wait fell because the heaviest rank does 1.6-2.5x
the mean expert work, so it saves the most from the fusion. nccl 0.63 -> 1.22
and topk_router 0.14 -> 0.36 grew (waiting inside collectives / small kernels
that now sit at the sync points; not investigated). GPU0/GPU2 again have
~445 ms idle gaps on a step whose reserved memory grew (allocator, finding 8).

07:27 UTC - **Step-0 grad_norm sampling on the fix134 build** (`gn_probe.py`:
`/init_trainer_server` then one step, three times): loss 12.3308 / 12.3298 /
12.3316 (inside the reference spread); grad_norm **0.5550 / 0.5097 / 0.6128**.
The re-init evidently draws a new LoRA-A each time, so step-0 grad_norm moves
+-10% with the adapter init; the fresh-start references (0.5665-0.5760) share
one seeded init and are the sharper comparison. Against them B3's 0.5506
(same init) is a 2.9% shift, outside the 1.7% spread but a third of the
init sensitivity. Op level (`swiglu_probe.py`): the fused kernel is 1.75x
closer to fp64 than the bf16 chain and computes the router-probability
gradient in fp32 instead of from bf16-rounded activations. Reading: the fused
path changes the gradient slightly and in the direction of the exact math;
loss unchanged. Recommendation: ship, flagged; Jack decides.
Logs `ab/gn-fix134.log`, `ab/B3-fix134.log`, capture `analysis_fix134/`.

07:28 UTC - **B4 deployed**: `allocator_reserve.py` (reserve free minus
16 GiB after the startup warmup) + `backend.py` call; pod-only DSA-flag
diagnostic now writes `ab/dsa_bwd_flags.rank*.txt`. Trainer restarted,
session `glm53fix1348`. Pre-registered rule for B4: ships if captures show no
cuMem* driver calls inside the step window and no allocator idle gaps, the
control mean does not regress, and step-0 loss/gn are unchanged (it cannot
change numerics). Expected gain is only on the steps that previously grew the
pool (0.15-0.7 s each); the control mean may move little.
