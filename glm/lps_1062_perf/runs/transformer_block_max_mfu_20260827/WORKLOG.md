# GLM-5.2 0D1M transformer-block maximum-MFU campaign

## Mission

- Source of truth: `/Users/jackrao/Documents/trainers/prompts/e.md`.
- Optimize the entire single `1M` transformer block: pre-attention norm, DSA
  indexer and sparse attention, attention projections/residual, pre-MoE norm,
  router, token dispatch/combine, routed/shared experts, and MoE residual.
- Work outside that block (embedding, final norm, LM head, loss, and optimizer)
  is excluded from optimization and from useful layer-only MFU.
- Exact shape: `/root/.cache/user_artifacts/glm52-debug-0d1m`, sequence length
  131,072, one datum, LoRA rank/alpha 32/32, TP1 / PP1 / CP8 / EP8 / ETP1 /
  DP1 on one 8xB300 node of devbox `tj-w5y89m3`.
- Driver: experiment-root `tools/profile_driver.py`, SHA-256
  `8b8097eec8e9c1901b3287c10448f77305ba72728c927d1c9d02a912b6c9e51f`.
- Branch: `jackrao/lps-1062-moe-layer-perf`; stacked PR:
  https://github.com/basetenlabs/trainers/pull/1197.
- Never profile unpushed code. Record the exact 40-character trainer SHA for
  every screen and profile.
- Use only devbox-up lifecycle scripts under
  `/root/.cache/user_artifacts/devboxes/w5y89m3/.devbox_up/`.

## Measurement contract

- Screen one variable at a time with one warmup and at least five unprofiled
  controls. Use unprofiled controls for acceptance; use Kineto only for causal
  attribution.
- The final winner receives 50 controls with controls 40-49 declared as its
  steady window.
- Keep model, topology, synthetic-data seed (`0xB300`), routing semantics,
  optimizer, LoRA, and batch shape fixed unless explicitly named as the tested
  variable.
- Training-equivalent candidates must preserve finite loss and gradient norm
  and may not force expert balance, drop tokens, or change DSA top-k/routing.
- `profile_driver.py` raw MFU/HFU assumes 78 production layers and is invalid
  here. Useful block training work is 4.090 GFLOP/token. Compute block MFU from
  the correlated full block wall span using 2.5 PFLOP/s/GPU B300 peak.

Parallel Folding meshes:

```text
Attention: TP1 x CP8 x DP1 x PP1
MoE:       ETP1 x EP8 x EDP1 x PP1
```

## Inherited control

The previous, incorrectly MoE-sublayer-scoped campaign still established a
valid whole-block starting point at pushed SHA
`d471834cd5d96d9e8bff4f4e7a9d906eec951bd2`: HybridEP with integrated
permutation, fused/elided router paths, tensorwise FP8 routed experts, and full
uniform one-layer recompute. Its warmed seven-control confirmation averaged
12,622.3 tok/s/GPU. The best 50-control steady window at pre-review SHA
`56b0b6cad55eae70802ed428e8f65098c269cf12` was 12,662.8 tok/s/GPU.

The supplied baseline trace attributes roughly 65.8 ms of repeated non-EP
work to full-block recomputation, including another DSA indexer, DSA forward,
router, dispatch, and expert forward. Prior 8K 0D1M testing found that
`selective/[]` correctly disables recompute and improved throughput 6.8%, but
that result is not accepted for this 131K contract until remeasured here.

## Candidate ladder

| case | single tested change | class | pushed SHA | status | tok/s/GPU | block metric |
|---|---|---|---|---|---:|---:|
| A0 | inherited full-recompute pushed winner | numerical-equivalent | `d471834cd5d96d9e8bff4f4e7a9d906eec951bd2` | matched control complete | 12,604.0 | 200.218 ms; 13.39% MFU |
| A1 | A0 + no recompute (`selective/[]`) | training-equivalent for exact 0D1M | `d471834cd5d96d9e8bff4f4e7a9d906eec951bd2` | complete; winner | 13,260.0 | 133.760 ms; 20.04% MFU |
| A2 | A1 + `attention_backend=fused` | training-equivalent | `d471834cd5d96d9e8bff4f4e7a9d906eec951bd2` | rejected; no difference | 13,258.0 | same cuDNN DSA path |
| A3 | A1 + provider-default `attention_backend=auto` | training-equivalent | `d471834cd5d96d9e8bff4f4e7a9d906eec951bd2` | rejected; no difference | 13,263.0 | same cuDNN DSA path |
| B1 | A1 + tensorwise FP8 MLA projections and shared experts | numerical-equivalent candidate | `6825bbb52eae3e61d04abf2c76622753de8edcc0` | complete; provisional winner | 13,288.0 | 132.525 ms; 20.23% MFU |
| B2 | B1 with MXFP8 instead of tensorwise on aligned block linears | numerical-equivalent candidate | `e4e67702e3edb82f651553d55da8a1bae9230fb6` | rejected; no win | 13,281.0 | 132.849 ms |
| B3 | B1 with NVFP4 instead of tensorwise on aligned block linears | speed-first numerical candidate | `e4e67702e3edb82f651553d55da8a1bae9230fb6` | rejected | 13,191.0 | 131.784 ms; TPS regression |
| B4 | B1 + EP communication overlap | training-equivalent candidate | `bf0d9c1a31f3ef90d74c430d2b897767464b782e` | rejected; no win | 13,302.0 | 132.915 ms |
| B5 | B1 + Dynamo scalar-output capture for HybridEP preprocessing | training-equivalent candidate | `bf0d9c1a31f3ef90d74c430d2b897767464b782e` | rejected | 13,229.0 | 132.393 ms; TPS regression |
| F0 | cleaned B1 winner, 50-control validation | numerical-equivalent candidate | `c0d37cb0d6ec73d3c1814d63f4344c7feb87377a` | complete; headline | 13,323.1 | 132.578 ms; 20.22% MFU |
| F1 | post-review exact-SHA confirmation | numerical-equivalent candidate | `61438f00c94ee30ed641648682002fd56ccfd4ec` | complete | 13,255.0 | activation confirmed |

## Log

20260827 PDT Corrected the campaign scope after user clarification. `1M` means
the complete attention-plus-MoE transformer block, not only the MoE submodule.
The prior campaign stopped with attention and recompute deliberately frozen;
that result is therefore an inherited starting point, not a whole-block
optimum.

20260827 PDT Read the clarified prompt, supplied CP8/EP8 trace report, prior
campaign worklog, NVIDIA's current MoE optimization workflow, devbox-up rules,
and the earlier 0D1M recompute experiment. Verified the leader has eight idle
B300 (`L20D`) GPUs and its worktree is detached at exact pushed SHA
`d471834cd5d96d9e8bff4f4e7a9d906eec951bd2`.

20260827 PDT A1 no-recompute and matched A0 full-recompute completed at exact
pushed SHA `d471834cd5d96d9e8bff4f4e7a9d906eec951bd2`. Independent warmed
ten-control screens averaged 13,260.0 and 12,604.0 tok/s/GPU respectively, so
removing full block recompute improves end-to-end throughput 5.20%. Losses and
gradient norms remain finite and follow the same trajectory. A1 uses 53.25 GB
on rank 0 after the profile versus A0's 43.69 GB, well within the 275 GB B300.

Flow-correlated GPU spans are 69.761 ms forward plus 130.457 ms
recompute/backward for A0 (200.218 ms total), versus 67.452 ms forward plus
66.309 ms backward for A1 (133.760 ms total). At 4.090 GFLOP/token useful work,
16,384 local tokens/GPU, and 2.5 PFLOP/s B300 peak, useful block MFU rises from
13.39% to 20.04%. A1 removes exactly one extra DSA indexer/top-k pair, sparse
attention forward, HybridEP dispatch/combine pair, router forward, and expert
forward. cuDNN DSA remains active; no attention semantics changed.

A0 artifacts:

- `a0_result.json`: `0f5ffb52105f51e3deb103937a80b58c861b5a122632789dc35d83756c5ff524`
- `a0_steady_result.json`: `28486581a932b742c8ba14ccf186ad3a934de8bacb7f7a9f6d7f7cf5bb7078c0`
- `a0_runtime.pt.trace.json`: `9417137a1abb140b913836a15d76fd1f43798ed7463da0d743272a5c478e7f04`
- `a0_trainer.log`: `88b1e1c07efc205d8a59ac019cadda3be914a7d2e3701fab66ce1447597829b8`

A1 artifacts:

- `a1_result.json`: `a842750361695ad7b0ffa332db70c2ecac49e7ba8bb731297327a2269bad7c1d`
- `a1_steady_result.json`: `a06b4b299f06a7dde3da7d5e24e607994fce05895236d91c2937cbf0934224f5`
- `a1_runtime.pt.trace.json`: `54097be1780190000ea541cde8eccc8fea65354fc92f96e7360ff5fd9a300875`
- `a1_trainer.log`: `2b12cc755fa723f35d51056b34be2fe16df01fd773feec3243bc7677687a2164`

20260827 PDT A2 explicit fused and A3 provider-default auto attention screens
are equivalent to A1 explicit flash: warmed ten-control means are 13,258,
13,263, and 13,260 tok/s/GPU. All three route the GLM DSA path through the
same explicit cuDNN backend, so the selector only changes an irrelevant outer
enum for this model. Keep explicit flash for the inherited configuration; do
not claim a backend gain.

20260827 PDT Pushed B1 at exact SHA
`6825bbb52eae3e61d04abf2c76622753de8edcc0`. It adds a
`tensorwise_block` precision scope that retains tensorwise FP8 routed experts
and also matches MLA Q-down, Q-up, KV-down, output projection, and shared
expert FC1/FC2. The DSA indexer, absorbed raw KV-up weight, router, and DSA
kernels remain unquantized. Repository pre-push `make check` and 173 model
tests passed. B1 is authorized to profile this pushed revision.

20260827 PDT B1 completed at exact pushed SHA
`6825bbb52eae3e61d04abf2c76622753de8edcc0`. The independent warmed
ten-control mean is 13,288 tok/s/GPU, +0.21% versus A1. Its correlated block
span is 67.297 ms forward plus 65.227 ms backward, 132.525 ms total and 20.23%
useful block MFU, a 0.92% wall reduction versus A1. The trace proves the wider
recipe engaged: BF16 NVJet GEMM falls by 6.309 ms; added FP8 GEMM and quant
cost 4.446 ms, leaving 1.863 ms less summed linear-kernel time. DSA kernels and
routing semantics are unchanged. Keep provisionally because the targeted
block metric improves even though fixed out-of-block LM-head work dilutes TPS.

20260827 PDT Pushed B2/B3 recipe support at exact SHA
`e4e67702e3edb82f651553d55da8a1bae9230fb6`. Routed experts remain on
tensorwise FP8; only the aligned, static-shape MLA/shared-expert matchers use
MXFP8 or NVFP4. This avoids the dynamic expert-row shape failures from the
prior campaign. Repository pre-push `make check` and 173 model tests passed.
B2 and B3 are authorized to profile this pushed revision.

20260827 PDT B2 MXFP8 initially failed only in the fixed 64-token startup
warmup: CP8 reduced its local input to `(8, 1, 6144)`, not divisible by the
MXFP8 block size 32. The actual target input is `(16384, 1, 6144)`. Retrying
through the same lifecycle with `BT_WARMUP_SEQ=256` succeeded. Its warmed mean
is 13,281 tok/s/GPU and 132.849 ms block wall, neither better than B1.

B3 NVFP4 also initialized with the aligned warmup and stayed finite, but its
warmed mean is 13,191 tok/s/GPU. Although its profile measured a 131.784 ms
block span, the larger numerical change and clear steady-TPS regression reject
it. B1 tensorwise remains the lower-precision winner.

Indexer projection launch ancestry shows only about 0.23 ms total for the
three pre-top-k projection GEMMs. The 13.8 ms cuDNN indexer and 28 ms cuDNN DSA
backward kernels dominate; concatenating `linear_wk` and
`linear_weights_proj` cannot move the block materially and is not pursued.

20260827 PDT B4 attempt 1 at pushed SHA
`e4e67702e3edb82f651553d55da8a1bae9230fb6` selected MCore's combined 1F1B
overlap schedule but failed in startup warmup before profiling. Trainers' CE
forward-step closure did not accept the schedule's `return_schedule_plan`
keyword. This was an integration defect, not an overlap performance result.

20260827 PDT Pushed the B4 integration fix at exact SHA
`8084a56060dcf51e5e2478c34b59cad8bf956796`. CE, RL, and DPO forward steps now
build MCore schedule plans through the model's output-processor hook while
retaining the trainer's chunked LM-head loss and metrics. Multi-microbatch loss
scaling forwards the plan request. Image inputs and router replay fail clearly
because their overlap semantics are not implemented. Repository pre-push
`make check` passed. B4 retry is authorized on this pushed revision.

20260827 PDT B4 attempt 2 reached chunked CE inside the schedule at pushed SHA
`8084a56060dcf51e5e2478c34b59cad8bf956796`, then MCore's CP normalization
tried to scale the custom chunked-head leaf loss in place. Pushed a
value/gradient-preserving non-leaf loss wrapper at exact SHA
`e31da3c520f66af6b67dfb7f91748b65068a4831`; repository pre-push `make check`
passed. B4 retry 2 is authorized on this pushed revision.

20260827 PDT B4 attempt 3 showed that combined 1F1B materializes its schedule
postprocess output as a leaf after the output-processor callback, so the
trainer-side multiply could not prevent MCore's later in-place CP scaling
failure. Fixed all MCore forward-loss scaling to use autograd-safe out-of-place
operations with leaf-loss value/gradient tests. Pushed MCore SHA
`635cdf44ec4ae0e0fe9fea1fbdfd856a7a1c10d3`, Megatron-Bridge pin SHA
`9d7af6ac32131b38e5001ca4af662546e887252c`, and trainers pin SHA
`26dc36e81012a5d9cb1868ff2a1c1468f831dc4a`. Trainers pre-push `make check`
passed. B4 retry 3 is authorized on this exact pushed trainers revision.

20260827 PDT Corrected the new per-token branch's test expectation: per-token
losses are already normalized and intentionally remain unscaled; the
out-of-place change targets the legacy two-tuple path used here. Direct runtime
checks cover both value and gradient branches. Pushed corrected MCore SHA
`78444f440bf8eb313f1fb5b7a4bcc0ebf10eec4c`, Megatron-Bridge pin SHA
`eb31910f3574d728e38a4db478f0420b642f8b2e`, and trainers pin SHA
`6a284b0dda143f083e8bd0a95d1988e3c1f64009`. Trainers pre-push `make check`
passed. B4 retry 4 is authorized on this exact pushed trainers revision.

20260827 PDT B4 attempt 4 passed forward/loss setup and entered the fine-grained
backward, then failed because `ScheduleNode.default_backward_func` tried to
backpropagate through the frozen embedding preprocessing output. MCore's
ordinary pipeline backward already skips outputs without gradients; added the
same filtering to the fine-grained node, including all-frozen and mixed-output
tests. Pushed MCore SHA `510e82681ac154d055fd82869f5b33f3ffee5cc6`,
Megatron-Bridge pin SHA `cbca8ed13883884407d94af9ca740254e98503ae`, and
trainers pin SHA `bf0d9c1a31f3ef90d74c430d2b897767464b782e`. Trainers
pre-push `make check` passed. B4 retry 5 is authorized on this exact revision.

20260827 PDT B5 scalar-output capture completed at exact pushed SHA
`bf0d9c1a31f3ef90d74c430d2b897767464b782e`. It removed the repeated Dynamo
graph-break warning but extended compilation through much of the first screen.
The independent warmed mean is 13,229 tok/s/GPU and its 132.393 ms block span
is within trace variance of B1. The steady TPS regression rejects it.

20260827 PDT Began final cleanup after all retained-scope screens. Removing
MXFP8/NVFP4 block modes, overlap schedule integration, and nested submodule
pins because none beat B1. Retaining only the B1 `tensorwise_block` scope on
top of the prior validated MoE changes; no-recompute remains an experiment
configuration rather than a global default.

20260827 PDT Final cleanup pushed at exact SHA
`c0d37cb0d6ec73d3c1814d63f4344c7feb87377a`. The selected files are byte-equal
to B1 SHA `6825bbb52eae3e61d04abf2c76622753de8edcc0`: tensorwise routed experts plus
tensorwise MLA/shared linears remain, while rejected Blackwell precision modes,
overlap integration, and nested pins are gone. Repository pre-push `make check`
and 173 model tests passed. F0 is authorized to profile this exact pushed SHA.

20260827 PDT Final review found that `tensorwise_block` would match only a
partial block on non-DSA MoE architectures. Added a fail-fast DSA-provider
guard and test, then pushed exact SHA
`61438f00c94ee30ed641648682002fd56ccfd4ec`. The guard does not change the
GLM-5.2 DSA path. Repository pre-push `make check` passed. F1 is authorized to
profile this exact pushed SHA.

20260827 PDT B4 retry 5 completed at exact pushed SHA
`bf0d9c1a31f3ef90d74c430d2b897767464b782e`. The independent warmed
ten-control mean is 13,302 tok/s/GPU, only +0.11% versus B1 and within run
noise. Its correlated block span is 67.701 ms forward plus 65.213 ms backward,
132.915 ms total, 0.390 ms slower than B1. Loss/grad norm are finite and the
combined schedule is genuinely active, but the target metric rejects overlap
for this one-microbatch shape. The overlap-only trainers and nested-submodule
changes will be removed from the final PR after the remaining screens.

20260827 PDT F0 50-control validation completed at exact pushed SHA
`c0d37cb0d6ec73d3c1814d63f4344c7feb87377a`. The declared controls 40-49
steady window is 1.229744 s mean FB and 13,323.1 tok/s/GPU; median is 1.227924
s, p90 is 1.233665 s, and max is 1.243699 s. Loss stayed
11.950715-11.951034 and grad norm stayed 0.000179903-0.000180164.

Flow-correlated attribution gives 67.346 ms block forward plus 65.232 ms block
backward, 132.578 ms total and 20.22% useful block MFU. The trace proves one
cuDNN DSA indexer/top-k forward, one sparse-attention forward/backward pair,
tensorwise FP8 NVJet kernels across routed experts and matched MLA/shared
linears, fused router, HybridEP dispatch/combine with integrated permutation,
and no duplicate block recomputation. Compared with matched A0, throughput is
+5.70%, block wall is -33.78%, and useful block MFU rises from 13.39% to 20.22%.

F0 artifacts:

- `f0_50controls_result.json`: `1cab35d76a35bd529575930f32679a4fab5a18f35ef42b5282ae5d3821bf1ad7`
- `f0_runtime.pt.trace.json`: `0da215536caa9b63f5ade0df9605c9f7b6dc6bfb6dfeacee8e7cb50ac55a6835`
- `f0_trainer.log`: `af00df6f717664e0076c6bd9272dcf88aef62e0f0f5934f8361626f879c12f9e`
- `config_b1_no_recompute_block_tensorwise_fp8.json`: `5a1b41acdfd514c68039f01ab5a86b80e4be0fcee2732b1bcce54d63c2702401`

20260827 PDT F1 exact-final-SHA confirmation completed at pushed SHA
`61438f00c94ee30ed641648682002fd56ccfd4ec`. The first screen retained normal
process-local HybridEP/JIT transients; the independent warmed ten-control mean
is 13,255 tok/s/GPU. Loss and grad norm remained finite. The runtime trace again
shows cuDNN DSA, block FP8, fused router, and HybridEP integrated permutation.
Focused B300 Megatron config tests passed 28/28. PR #1197 is mergeable and its
final `check` and `sampler-unit-contract-tests` jobs passed.

F1 artifacts:

- `f1_result.json`: `791f83e118dd111010a7375f5b2bfe55b66149c75c5920b22c23be1b4376df7a`
- `f1_steady_result.json`: `e0785d3390cd810cf31b68a0b0f26a8c54a5d4ec031106ef3feaae119a44ecad`
- `f1_runtime.pt.trace.json`: `5bba782f5a364610c654089d4df368f0f7be0e9d123e0e895b14b53baebeabc8`
- `f1_trainer.log`: `5b523438ceae9029347fd13729f4d8e13bce1751fa4ea93fe224e194ecad0f86`

## Final result

The winner is the prior HybridEP stack with integrated permutation, fused and
trivial-group-elided router, tensorwise FP8 routed experts, plus tensorwise FP8
on MLA Q-down/Q-up/KV-down/output projection and shared-expert FC1/FC2. Full
block recompute is disabled with `selective/[]`. DSA indexer/kernels, raw
absorbed KV-up, router precision, routing/top-k semantics, data, optimizer, and
topology remain unchanged.

No remaining candidate is justified under the exact contract:

- cuDNN is already the fused DSA winner; NVIDIA reports it materially ahead of
  TileLang on Blackwell, and the trace proves it is active.
- DSA indexer projections total only about 0.23 ms; fusing them cannot move the
  132 ms block materially.
- MXFP8 and NVFP4 block recipes do not beat tensorwise FP8.
- EP overlap and shared-expert overlap do not beat eager HybridEP at one
  microbatch.
- Dynamo scalar capture regresses steady throughput.
- TE attention graphs do not support this packed-sequence LoRA path's
  non-Tensor `packed_seq_params`; full-iteration graphs would also capture and
  optimize the out-of-scope LM head/loss.
- Changing DSA top-k frequency/selection, forcing balanced routing, dropping
  tokens, or reducing precision inside the DSA indexer/kernel would change the
  model contract and is not accepted as training-equivalent.

20260827 PDT Updated PR #1197 title/body to the full transformer-block scope.
Final head `61438f00c94ee30ed641648682002fd56ccfd4ec` is mergeable; GitHub `check`
and `sampler-unit-contract-tests` both passed. Stopped the trainer through the
devbox-up lifecycle script; all 16 GPUs on `tj-w5y89m3` are idle. The devbox
remains provisioned.
