# GLM-5.2 0D1M MoE-layer maximum-MFU campaign

## Mission

- Source of truth: `/Users/jackrao/Documents/trainers/prompts/e.md`.
- Optimize only the MoE portion of the GLM-5.2 0D1M debug model as aggressively
  as possible for throughput and useful layer-only MFU.
- Devbox: `tj-w5y89m3`, 2 nodes x 8 B300 GPUs. The requested folded mesh uses
  one node: TP1 / PP1 / CP8 / EP8 / ETP1 / DP1.
- Model: `/root/.cache/user_artifacts/glm52-debug-0d1m`.
- Sequence length: 131,072. Datums per step: 1. LoRA rank/alpha: 32/32.
- Driver: experiment-root `tools/profile_driver.py`, SHA-256
  `8b8097eec8e9c1901b3287c10448f77305ba72728c927d1c9d02a912b6c9e51f`.
- Driver helper: experiment-root `tools/mfu.py`, SHA-256
  `4b009fffda3f6100c97ef03acd4864260221748ccda9b4b516ec37ac5383c43b`.
- Code branch: `jackrao/lps-1062-moe-layer-perf`, stacked from approved PR
  #1150 head `ea935be13e258cc032c567c72861ca60bb4e8cdf`.
- Stacked PR: https://github.com/basetenlabs/trainers/pull/1197.
- No code revision may be profiled before it is committed and pushed. Record
  the exact 40-character trainer SHA for every profile below.
- Use only the box-specific devbox-up lifecycle scripts under
  `/root/.cache/user_artifacts/devboxes/w5y89m3/.devbox_up/`; no ad hoc launch,
  polling, or teardown scripts.

## Measurement contract

- Every acceptance run uses one warmup, five unprofiled control windows, and
  one runtime-profiled window unless a case explicitly records otherwise.
- Headline throughput is derived only from unprofiled controls. Kineto is used
  for causal attribution, not acceptance timing.
- Natural routing, optimizer behavior, data RNG seed (`0xB300`), topology,
  recompute, precision, and batch shape remain fixed unless they are the single
  tested variable.
- Training-equivalent candidates must preserve router selections/probabilities,
  loss, and grad norm within the numerical behavior of the selected precision.
- Forced balancing, token dropping, or capacity padding that changes routed
  work is benchmark-only and cannot be the final winner.
- `profile_driver.py`'s raw MFU/HFU fields assume the 78-layer production model
  and are invalid for 0D1M. Report absolute tok/s/GPU and calculate layer-only
  useful-FLOPs MFU from the correlated MoE-layer wall span. Baseline useful
  training work is 4.090 GFLOP/token and B300 peak is 2.5 PFLOP/s/GPU.

Parallel Folding meshes:

```text
Attention: TP1 x CP8 x DP1 x PP1
MoE:       ETP1 x EP8 x EDP1 x PP1
```

## Existing baseline

Source trace:
`../hybridep_b300_benchmarks_20260826/case2_runtime.pt.trace.json` at trainer
SHA `061947aca8bc00540ee7ad9c7c3c8d7f0df68e20`.

| metric | baseline |
|---|---:|
| mean control FB | 1.4659 s |
| control throughput | 11,176.5 tok/s/GPU |
| layer forward | 79.317 ms |
| checkpoint recompute + backward | 156.266 ms |
| total layer training span | 235.583 ms |
| EP all-to-all wall | 33.238 ms (14.1%) |
| adjusted non-EP useful-FLOPs MFU | 13.25% |

The baseline is standard AllToAll, grouped experts, fused permutation, router
fusion disabled, and full one-layer recompute. It is not EP-communication-bound.
The highest-priority MoE candidates are therefore expert-kernel fusion, router
and preprocess fusion, dispatcher A/B, and Blackwell lower precision.

## Candidate ladder

| case | single tested change | class | pushed SHA | status | tok/s/GPU | MoE metric |
|---|---|---|---|---|---:|---:|
| A0 | current stacked-tip AllToAll baseline | equivalent | `ea935be13e258cc032c567c72861ca60bb4e8cdf` | complete | 11,794.3 | 11.82% |
| A1 | HybridEP, GLM provider default 16 SMs | equivalent | `ea935be13e258cc032c567c72861ca60bb4e8cdf` | complete | 12,422.6 | 12.53% |
| A2 | DeepEP, same stack and routing | equivalent | `ea935be13e258cc032c567c72861ca60bb4e8cdf` | rejected | 12,037.2 | 7.25% profiled |
| B1 | HybridEP + TE grouped-MLP op fuser, BF16 | equivalent | `490a950ceadb7e6c193d86c7f185dbcfafa0d92a` | rejected at startup | N/A | N/A |
| B2 | HybridEP + router fusion | equivalent | `4fd5cd8eb6f3f2b21f9b18a5e654eaf47796b292` | complete | 12,468.9 | 12.73% |
| B3 | B2 + HybridEP-integrated permutation | equivalent after parity proof | `43f79a76d8161519e2f678a4b71e7059dcaf5ac4` | complete | 12,509.2 | 13.12% |
| B4 | B3 + HybridEP fused-permute block/SM/chunk sweep | equivalent | `569bfb45045e62c49c5170cba6e65e6bc1a79bcb` | rejected; B3 defaults win | 12,509.2 | 13.12% |
| B5 | eliminate trivial group-limited top-k for 1/1 groups | equivalent after parity proof | `3d8ab9e490297fe0320610cf7518d4276175f7c9` | complete; targeted micro-win | 12,509.0 | 13.06% |
| C0 | B5 + expert-only tensorwise FP8 | numerical-equivalent candidate | `ae5e151d2530993e6ad30dac27cf227dbff4bba6` | complete | 12,638.0 | expert kernels -28.4% |
| C0b | B5 + expert-only blockwise FP8 | numerical-equivalent candidate | `ae5e151d2530993e6ad30dac27cf227dbff4bba6` | rejected at startup | N/A | N/A |
| C1 | B5 + expert-only MXFP8 | numerical-equivalent candidate | `ae5e151d2530993e6ad30dac27cf227dbff4bba6` | rejected at startup | N/A | N/A |
| C2 | B5 + expert-only NVFP4 | speed-first numerical candidate | `ae5e151d2530993e6ad30dac27cf227dbff4bba6` | rejected at startup | N/A | N/A |
| C3 | C0 + shared-expert overlap | numerical-equivalent candidate | `2add46eee79338e84a5dae547d506be371d7b8bd` | rejected | 12,598.9 | 45.3% comm hidden, no TPS win |
| D1 | narrow TE graphs: `moe_router`, `moe_preprocess` | equivalent | N/A | incompatible with full recompute/runtime | N/A | N/A |
| F0 | cleaned C0 winner, 50-control validation | numerical-equivalent candidate | `56b0b6cad55eae70802ed428e8f65098c269cf12` | complete | 12,662.8 steady | expert kernels -28.5% |
| F1 | post-review exact-SHA confirmation | numerical-equivalent candidate | `d471834cd5d96d9e8bff4f4e7a9d906eec951bd2` | complete | 12,622.3 | expert kernels -28.4% |

EP overlap is not an early candidate: full recompute and shared-expert overlap
are incompatible with it, and one microbatch provides little scheduling depth.
It will only be tested if earlier profiles still expose communication and a
matched no-full-recompute configuration remains MoE-scoped and correct.

## Log

20260826 PDT Read the original prompt, supplied Perfetto report, NVIDIA MoE
optimization, dispatcher, and EP-overlap guidance, local code paths, and the
devbox-up skill. Confirmed the provided box is idle, exposes 8 B300 (`L20D`)
GPUs on the leader, has two nodes, and has box-specific lifecycle scripts.

20260826 PDT Created `jackrao/lps-1062-moe-layer-perf` from approved PR #1150
head `ea935be13e258cc032c567c72861ca60bb4e8cdf`. The stacked base is deliberate:
it supplies the already-reviewed dispatcher-backend plumbing without adding
experimental changes to the approved production PR.

20260826 PDT Pushed campaign branch at exact baseline SHA
`ea935be13e258cc032c567c72861ca60bb4e8cdf`. Repository `make check` passed
after initializing the fresh worktree's declared submodules. A0 is authorized
to profile this pushed SHA.

20260827 UTC C1 attempt 1 failed before model construction at exact pushed SHA
`f676c9aeb7080ee589404c05b3ceb4eaa83912e7`; the driver did not run and no
profile exists. Pinned TE 2.16 `GroupedLinear` does not expose the newer
`use_grouped_tensor` argument, so the documented grouped-tensor padding route
cannot run on this image. Preserved `c1_startup_failed.log` with SHA-256
`d76f7a403308b6772290b26186a10b4852252bec9f4053a6a15521dc756a430e`.

20260827 UTC Pushed the installed-TE-compatible per-module autocast path at
exact SHA `ae5e151d2530993e6ad30dac27cf227dbff4bba6`. It leaves grouped-tensor mode
untouched and retains the routed-expert-only recipes. Repository pre-push
`make check` passed. C0 is authorized first; MXFP8/NVFP4 retries depend on
their ordinary grouped-linear shape compatibility.

20260827 UTC C0 expert-only tensorwise FP8 is the current throughput and expert
compute winner. The warmed seven-control screen averaged 12,606.1 tok/s/GPU;
an independent already-warmed validation averaged 12,638.0 tok/s/GPU and
1.29640 s FB, +1.03% versus B3/B5. Expert GroupedLinear ancestry proves real
FP8 kernels (`nvjet_sm103_qqtst`): BF16 expert GEMMs total 20.577 ms; tensorwise
FP8 totals 10.975 ms GEMM + 1.680 ms quant + 2.081 ms other, 14.736 ms total and
a 28.4% expert-path reduction. Loss/grad norm stay finite and follow the same
small optimizer trajectory. Kineto perturbs HybridEP much more under this path,
so whole-layer span is not used for C0 acceptance.

C0 artifacts:

- `c0_result.json`: `7728d935cd38e7aa805800d4ea5339f699e48731c7b71d47fe98897f58717f0e`
- `c0_steady_result.json`: `675d9d48c6c1fd8976633f27d0fb5951c8b1f4db23c1eb631cab0b1e47e71b64`
- `c0_runtime.pt.trace.json`: `0ad1d1a1e469f1c062c91417330b87c1b316fa37cafbd1f103435f17b0776753`
- `c0_trainer.log`: `257865b13d9969d0145bf3ce2b664cb90772a46735a8674d5d882024b8036b1b`
- `c0_validation_result.json`: `74524a6fd810e72e6e7ae3dccab71131988148423d6276ef883b172fea83b099`
- `c0_validation_runtime.pt.trace.json`: `cd9a46575225d30b76317d8ffa74fb34040e4f0b6861e87372177ba1c81981f8`
- `c0_validation_trainer.log`: `63a3266ae484f9be4525057877461a920ecb5abebe9871dad755fd92b7c234b3`

20260827 UTC C1 MXFP8 retry is rejected. It reached expert FC1 in startup
warmup, then TE failed on an expert split shaped `(1, 6144)`: MXFP8 requires
both dimensions divisible by 32. The missing grouped-tensor API prevents
padding these dynamic expert rows on pinned TE 2.16. No driver/profile was
produced. `c1_retry_startup_failed.log` SHA-256:
`d75966b69f52f10f3143bf73d0bc7987280d0f15571bfe6ce1393865db23d48f`.

20260827 UTC C0b blockwise FP8 and C2 NVFP4 are rejected. Both reached routed
expert FC1 during startup warmup, then failed on dynamic expert row counts:
blockwise requires rows divisible by 4; NVFP4 requires rows divisible by 16
and observed `(2, 6144)`. No driver/profile exists for either. Preserved logs:

- `c0b_startup_failed.log`: `876dcacf7a8661eb4dcb180d937197cb7048805a0b29e1480caa66755aa3b296`
- `c2_startup_failed.log`: `4a93f7f8723c833069b8846dbb1e75b2828618f2719def1cd7223eb4cf3f9e0f`

20260827 UTC Pushed the shared-expert overlap override at exact SHA
`2add46eee79338e84a5dae547d506be371d7b8bd`. `None` preserves current
dispatcher behavior; explicit `true` is applied after HybridEP's conservative
disable. Models tests 13/13, focused lint/format, and repository pre-push
`make check` passed. C3 is authorized to profile this pushed SHA.

20260827 UTC C3 shared-expert overlap is rejected for this exact shape. Warmed
throughput is 12,598.9 tok/s/GPU versus C0's 12,638.0 validation. The overlap
is genuinely engaged: `analyze_hybridep_overlap.py` measures 45.29% of
HybridEP kernel time concurrent with compute, versus 0% for C0, but contention
and stream overhead cancel the hidden communication. Expert FP8 kernels remain
unchanged. Artifacts:

- `c3_result.json`: `eaafca820804f8dc8c3f244d25822d81f909fff93882cbcd34d64208e813dd9a`
- `c3_steady_result.json`: `c09130b8f5760715251a1f299c6bbbd9212e0b4c35496598201b81eeb28ba0f7`
- `c3_runtime.pt.trace.json`: `69d7eba9086fe2437256e680424a00a861ef4d9aeddab1e49706dd1c9dacd498`
- `c3_trainer.log`: `d82ebc5f4b0af776414abb39b57fc97dc7a16efe0cf2e648c40230393f9e081e`
- `analyze_hybridep_overlap.py`: `163a39d3d7fd8c6e99e028bd9a188c0d6ef4ecf3219cff27052de15793d4231e`

20260827 UTC D1 narrow TE CUDA graphs are not a valid candidate under the
frozen contract. MCore hard-rejects TE-scoped graphs with full recompute; using
selective recompute changes the non-MoE attention/memory path. Trainers also
does not construct the required `TECudaGraphHelper` lifecycle, so a field-only
change would silently remain eager. D1 is skipped, not measured.

20260827 UTC Cleaned PR #1197 at exact pushed SHA
`56b0b6cad55eae70802ed428e8f65098c269cf12`. Removed every rejected surface:
TE op fuser, HybridEP kernel-budget overrides, shared overlap, and unsupported
blockwise/MXFP8/NVFP4 choices. Retained only HybridEP-integrated permutation,
router fusion, guarded 1/1 group elision, and expert-only tensorwise FP8.
Models tests 13/13, server-main tests 9/9, focused lint/format, and repository
pre-push `make check` passed. F0 is authorized to profile this pushed SHA.

20260827 UTC F0 50-control validation completed at exact pushed SHA
`56b0b6cad55eae70802ed428e8f65098c269cf12`. The all-control mean is 12,545
tok/s/GPU because controls 0-2 include setup. The declared steady window,
controls 40-49, is 1.293866 s mean FB and 12,662.8 tok/s/GPU; median 1.294032
s, p90/max 1.301515 s. Loss remained 11.950714-11.951090 and grad norm
0.000179802-0.000180217. An independent warmed five-control profile run
averaged 12,651.9 tok/s/GPU. The trace proves the final mechanisms: 10.994 ms
expert FP8 GEMM + 1.913 ms quant + 1.807 ms other = 14.714 ms expert path;
14.201 ms fused HybridEP payload; zero standalone permute/unpermute kernels;
and 0.146 ms fused-router forward kernels.

Relative to A0's steady controls 1-4 mean of 12,209.2 tok/s/GPU, F0 improves
end-to-end throughput by 3.72%. Relative to the original 11,176.5 tok/s/GPU
three-control baseline in the supplied report, it is +13.30%, though that older
baseline contained more warmup variance. Using B3's 204.229 ms BF16 layer span
and the measured 5.863 ms expert-path reduction gives an estimated 198.366 ms
final layer span and 13.51% useful-FLOPs layer MFU versus A0's 11.82%; this MFU
is explicitly an estimate because Kineto perturbs HybridEP in FP8 runs.

F0 artifacts:

- `f0_50controls_result.json`: `c19ee538626729d98a788127b119b2a3ab717ad32c86004e21fe0de7653ad50b`
- `f0_profile_result.json`: `324d5cc2d52d72d504031c0ac0c839e4805958b6d03bc13fe3eac7f38cf1dc32`
- `f0_runtime.pt.trace.json`: `e61d06ed6dc4923a5cee1078d1e9d3e413c57cadc77c092ffae7a63a57c19d7f`
- `f0_trainer.log`: `6c3e7a12e537fb3263eda97400841981ee081cca996df5e79e847867f0f5646a`
- `config_c0_hybridep_expert_tensorwise_fp8.json`: `2ed6ef4163e8e28b78f454e3de7a546da671e2a17b26ccd75dcab07e4f059ed1`

20260827 UTC Addressed final review before confirmation. Router fusion now
rejects R3 replay because pinned MCore's fused path bypasses replay, and every
new performance option now requires a provider with real MoE experts instead
of silently no-oping on dense models. Pushed exact SHA
`d471834cd5d96d9e8bff4f4e7a9d906eec951bd2`; repository pre-push `make check`
passed. F1 is authorized to profile this pushed SHA.

20260827 UTC F1 exact-final-SHA confirmation completed at
`d471834cd5d96d9e8bff4f4e7a9d906eec951bd2`. The first process-local profile
screen contained expected HybridEP JIT transients. The second warmed
seven-control screen averaged 12,622.3 tok/s/GPU and 1.29802 s FB; six of seven
controls were 12,647-12,705 tok/s/GPU and one periodic outlier was 12,266.
Loss/grad norm were finite. The trace again proves 11.033 ms expert FP8 GEMM +
1.956 ms quant + 1.759 ms other = 14.748 ms expert path, fused router forward,
fused HybridEP dispatch/combine, and no standalone permute/unpermute kernels.
Focused Megatron config tests passed 26/26 on the box. GitHub `check` and
`sampler-unit-contract-tests` passed; PR #1197 is mergeable. The generated
trainer was stopped through the devbox-up lifecycle script and all GPUs are
idle. Final artifacts:

- `f1_profile_result.json`: `18522499b61266652f3e2585bc750e1a5518d6a1049cbf35bb1c7837d570aa53`
- `f1_steady_result.json`: `ef52d488f5b6e24dba9c12b1abccc1a63321eda5e685428aff3e339b44d104fe`
- `f1_runtime.pt.trace.json`: `facddde358eca545b4cddf1951483d646dfa16c891196505b9787c71a2a262ce`
- `f1_trainer.log`: `d636b8f91610020233e11adeb59d4c2fec166b6f79fed01b2fd6e3a6017ebc89`

## Final result

The final winning configuration is HybridEP with 16 communication SMs,
HybridEP-integrated permutation at its automatic 108 fused blocks and matched
64-token chunks, fused router with redundant 1/1 grouping elided, and
tensorwise FP8 on routed expert FC1/FC2 only. Attention, shared experts, router,
LM head, routing semantics, topology, data, optimizer, and recompute remain at
the frozen contract. Use F0's 50-control controls 40-49 mean, 12,662.8
tok/s/GPU, as the headline; use F1 as exact-final-SHA confirmation.

20260826 PDT A0 AllToAll completed at exact pushed SHA
`ea935be13e258cc032c567c72861ca60bb4e8cdf`. Driver arguments:
`--label a0-ea935be13-alltoall-debug0d1m-131k-d1 --seq-len 131072
--datums 1 --num-gpus 8 --lora-rank 32 --control-repeats 5
--runtime-profile`. Controls were 10,382.7, 12,187.6, 12,168.2, 12,302.9,
and 12,179.3 tok/s/GPU; contractual five-control mean 11,794.3 tok/s/GPU,
mean FB 1.38915 s. Controls 1-4 mean 12,209.2 tok/s/GPU and 1.34193 s FB;
control 0 retained residual warmup. Runtime-profile FB was 1.33341 s. Loss and
grad norm were finite.

Flow-edge attribution gives 74.026 ms layer forward plus 152.890 ms checkpoint
recompute/backward, 226.916 ms total layer wall. Nine EP SendRecv collectives
sum to 32.114 ms. Useful layer MFU is 11.82%; the non-EP compute-part MFU after
removing serial EP time is 13.76%. Summed layer kernel buckets: DSA attention
64.640 ms, NVJet GEMM 42.796 ms, EP AllToAll 32.114 ms, routing/permutation
13.079 ms, other 56.867 ms. The MoE-addressable buckets remain large enough to
justify dispatcher, grouped-expert, and routing fusion work.

A0 artifacts:

- `a0_result.json`: `32bbaa7573098b320921c13c97a6aa4458439dc6e8bdadfa81c9f0b967ca53ab`
- `a0_runtime.pt.trace.json`: `0848d2b5cbeacb747e4556a86bb17cc18681ad6c01a8746495c7bd3180ad8345`
- `a0_trainer.log`: `4db759612496e0cc9aa2b209f4755df34f06540961a60d5ac894596c8131b25a`

20260826 PDT A1 HybridEP completed at exact pushed SHA
`ea935be13e258cc032c567c72861ca60bb4e8cdf`. The current GLM provider supplies
16 communication SMs; no campaign override was applied. The first profile run
captured HybridEP's shape-specific cold path: warmup 34.41 s, then controls
6,982.2, 6,597.3, 12,442.4, 12,320.2, and 12,403.3 tok/s/GPU. Its five-control
mean is intentionally rejected as a steady acceptance metric. The runtime trace
proved real HybridEP dispatch/combine kernels and had 12,324.4 tok/s/GPU.

A second unprofiled screen in the same healthy process produced seven controls
at a 12,422.6 tok/s/GPU mean and 1.31889 s mean FB. This is +1.75% versus A0's
steady controls 1-4 mean. Flow-edge attribution gives 71.613 ms layer forward
plus 142.277 ms recompute/backward, 213.890 ms total layer wall, a 5.74% wall
reduction and useful layer MFU 12.53% (+6.0% relative). HybridEP payload kernels
sum to 10.927 ms and device-sync kernels to 5.260 ms, versus 32.114 ms of A0
SendRecv. Loss and grad norm remain finite with the same values to normal BF16
rounding; HybridEP is the dispatcher winner so far.

A1 artifacts:

- `a1_result.json`: `46cfa03469a3abe6e29466dfd3d13993b856eab3d6da7529b2e95c2e73a3d521`
- `a1_steady_result.json`: `47099614f63912492ea9a9d116ddef5e206f2a621e8ceeb1300961e07edc46ab`
- `a1_runtime.pt.trace.json`: `58e748df446ba4ea5cfceac66d2d8fdaf5e879617f92df0294cd5804ec1b19e7`
- `a1_trainer.log`: `3916d557903151f596bacde80d65f0e20ea4cd28d5ff0a2378fd0cdc303bb6b1`

20260826 PDT A2 DeepEP completed at exact pushed SHA
`ea935be13e258cc032c567c72861ca60bb4e8cdf`. Its first five controls averaged
11,180 tok/s/GPU while setup effects cleared. A second seven-control warmed
screen averaged 12,037.2 tok/s/GPU and 1.36111 s FB, 3.10% below HybridEP.
Loss and grad norm remained finite.

The trace proves the DeepEP backend ran, but it is strongly inferior on this
intra-node EP8 shape: 146.023 ms layer forward plus 223.775 ms
recompute/backward, 369.798 ms profiled layer wall and 7.25% useful layer MFU.
DeepEP kernels sum to 75.877 ms dispatch, 29.209 ms combine, 62.760 ms cached
combine notification, and 2.534 ms dispatch notification. DeepEP is rejected;
HybridEP remains the dispatcher winner.

A2 artifacts:

- `a2_result.json`: `7957da73dda66eabb6ef60668df2066cee28bf5ba28bb5691897f727b76c8a29`
- `a2_steady_result.json`: `f255384fcb8ba9f3a46da53cfdca1779a8f2509cbea5d10d6e4e03fd4daa3c5a`
- `a2_runtime.pt.trace.json`: `0b70c38c5716f2f2859ff608654a940823ef3d2baecfc3c41ef7ac28f25e508f`
- `a2_trainer.log`: `99f6ac4d50778ad858a6c702ae9baa029562a9bd95782b8f32eadcb4af7c6e56`

20260826 PDT Opened stacked trainers PR #1197 and pushed the first op-fuser
canary at exact SHA `490a950ceadb7e6c193d86c7f185dbcfafa0d92a`.
The change is default-off and exposes `use_transformer_engine_op_fuser`, sets
`NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` before lazy Megatron backend import, and
fails if the selected provider cannot apply the setting. Models tests 13/13,
server-main tests 10/10, focused lint, repository lint, formatting, and
typecheck passed. B1 is authorized to profile this pushed SHA.

20260826 PDT B1 failed in the generated startup warmup before health at exact
pushed SHA `490a950ceadb7e6c193d86c7f185dbcfafa0d92a`; the driver did not run and
no profile was produced. The path did not silently fall back: the stack reached
`TEGroupedMLP._fused_forward` and TE's operation fuser, then BF16 SwiGLU failed
in `create_2D_tensor_map` with `CUDA Error: invalid argument` at the 131K
routed-token shape. This establishes that the pinned TE 2.16 BF16 op-fuser path
does not support this shape. GLU interleaving is not applied as a workaround:
the direct HF `GatedMLPMapping` loads contiguous `[gate; up]` weights, so
enabling interleaving without a matched mapping would corrupt numerics. B1 is
rejected rather than weakened into an unproven fallback.

- `b1_startup_failed.log`: `8d59b138626dae354eec116a00ad24a6b92ada1e2fb46b4ac2e2d4ac02d321de`

20260827 UTC Pushed router-fusion canary at exact SHA
`4fd5cd8eb6f3f2b21f9b18a5e654eaf47796b292`. It adds a default-off
`moe_router_fusion` field and fail-fast provider propagation. Models tests
13/13, focused lint/format, and the repository pre-push `make check` passed.
B2 is authorized to profile this pushed SHA.

20260827 UTC B2 router fusion completed at exact pushed SHA
`4fd5cd8eb6f3f2b21f9b18a5e654eaf47796b292`. The first run included HybridEP
JIT transients, then a second seven-control screen averaged 12,468.9 tok/s/GPU
and 1.31399 s FB, +0.37% versus A1. Loss and grad norm remained finite and
matched the baseline BF16 behavior.

The runtime trace proves TE's `FusedTopkScoreFunction` ran. Layer forward fell
from A1's 71.613 ms to 68.196 ms while recompute/backward was flat at 142.334
ms, for 210.530 ms total layer wall (-1.57%) and 12.73% useful layer MFU.
HybridEP payload and sync times were unchanged, so the layer gain is correctly
attributed to routing/pre-dispatch work rather than dispatcher variance. B2 is
the current winner.

B2 artifacts:

- `b2_result.json`: `b5efae9102ecefe816f740b7996c344ab0dae2b583fb872193a5e765c58ad5e7`
- `b2_steady_result.json`: `162e612ab74b58b7b8f9fdd35d07d048c56cafddabd7ba9e69658a225f1caf67`
- `b2_runtime.pt.trace.json`: `0968606e832752b7a983e5be2bf32b2cacf67665d0c48627eeaf275f48d961f3`
- `b2_trainer.log`: `f4403338cdc5ea476ef37cd62e8a87b5b670326a6dc851cba30ca3c732a91bdf`

20260827 UTC Pushed HybridEP-integrated permutation canary at exact SHA
`43f79a76d8161519e2f678a4b71e7059dcaf5ac4`. It adds a default-off field,
requires the HybridEP backend and provider capability, and deliberately leaves
standard `moe_permute_fusion` enabled. Models tests 13/13, focused lint/format,
and repository pre-push `make check` passed. B3 is authorized to profile this
pushed SHA.

20260827 UTC B3 HybridEP-integrated permutation completed at exact pushed SHA
`43f79a76d8161519e2f678a4b71e7059dcaf5ac4`. Its warmed seven-control screen
averaged 12,509.2 tok/s/GPU and 1.30975 s FB, +0.32% versus B2. Loss and grad
norm remained finite and aligned with B2 within normal BF16 reduction-order
variation.

The trace provides direct activation proof: standalone `_permute_kernel` and
`_unpermute_kernel` counts are zero, while HybridEP dispatch/combine templates
carry 108 fused permutation blocks. Layer forward is 65.979 ms and
recompute/backward 138.250 ms, 204.229 ms total (-2.99% versus B2) and 13.12%
useful layer MFU. HybridEP payload/fused-permute kernels total 15.113 ms, but
device sync collapses from 5.260 ms in A1 to 0.350 ms and the separate
routing/permutation bucket falls to 3.355 ms. B3 is the current winner.

B3 artifacts:

- `b3_result.json`: `7c62cccd450dc5cad64c76ad8321b38fca699d0c0d00e3acbe2802a46fd64c67`
- `b3_steady_result.json`: `09612e9fb6836354f3b0499711a6f9ab449fe3f8fd9d67fb0cf27e22564b8b71`
- `b3_runtime.pt.trace.json`: `0ab2656492c18afb4fcc60b021b728eb8050962e8cadaf60ce173fb23e22a7f8`
- `b3_trainer.log`: `bd2b8cb47e12e3fdce8a36fe9c05a2c11afd925e44c49c821792bc049e7e00d7`

20260827 UTC Pushed HybridEP kernel-budget controls at exact SHA
`569bfb45045e62c49c5170cba6e65e6bc1a79bcb`. It exposes optional flex
communication SMs and HybridEP permute/unpermute/preprocessing budgets with
strict backend and capability checks. Defaults remain unchanged. Models tests
13/13, focused lint/format, and repository pre-push `make check` passed. B4 is
authorized to profile this pushed SHA.

20260827 UTC B5 trivial-group elision completed at exact pushed SHA
`3d8ab9e490297fe0320610cf7518d4276175f7c9`. Warmed throughput is effectively
identical to B3 at 12,509.0 tok/s/GPU and 1.30977 s FB. The targeted fused
router forward kernels fall from 0.272 ms to 0.145 ms across original forward
and recompute, proving the redundant group path was removed. Total layer wall
is 205.306 ms versus B3's 204.229 ms, within unrelated DSA/dispatcher trace
variance; useful layer MFU is 13.06%. Loss and grad norm remain aligned. Keep as
a semantics-preserving micro-optimization, not as an end-to-end speed claim.

B5 artifacts:

- `b5_result.json`: `4811b9974c75c2a58caa4ce74b294f60766b75a3d2f0ff0c5dbcb895ae3d3355`
- `b5_steady_result.json`: `7657efe277792791e66198b80242ea93c4e7c8fdf781b897c7b0b52c77079f2b`
- `b5_runtime.pt.trace.json`: `f277ec74d0f80b75e6c23131bcb031bc7e80d762d3c4a3db6e39af517f6665df`
- `b5_trainer.log`: `82d7a03836305fcee109fe71ce76aa617451128b38f11413edb72c36e4493646`

20260827 UTC Pushed expert-only low precision at exact SHA
`f676c9aeb7080ee589404c05b3ceb4eaa83912e7`. The default remains BF16.
`tensorwise`, `blockwise`, and `mxfp8` create FP8 per-module recipes; `nvfp4`
creates an FP4 recipe. Matchers target only routed expert FC1/FC2, force the
required grouped-tensor path, keep router math FP32, reject existing recipes,
and reject the incompatible TE op fuser. Models tests 13/13, focused
lint/format, and repository pre-push `make check` passed. C1/C2 are authorized
to profile this pushed SHA.

20260827 UTC B4 fused-block sweep rejects both overrides. At the same pushed
SHA, 64 blocks produced 12,486.0 tok/s/GPU, 210.210 ms layer wall, and 17.582
ms HybridEP payload/fused-permute time. 32 blocks produced 12,391.5 tok/s/GPU,
216.758 ms layer wall, and 29.731 ms payload/fused-permute time. The B3 default
of 108 blocks remains decisively best at 204.229 ms and 15.113 ms. All arms had
finite, aligned loss/grad norm.

B4a artifacts:

- `b4a_result.json`: `a498c82fde8951f7237aba95dadb668cef063a0d29c1aa98c4619d92ab172d5d`
- `b4a_steady_result.json`: `a468706e6db17116f53b8256c18dd0b482ebcf445baf62739eb41b1d8a959c4d`
- `b4a_runtime.pt.trace.json`: `b011e12841fe0415658e2000b5d4131a7ffb4927d237b2a304f8a5674ff7d88d`
- `b4a_trainer.log`: `8f9c339fce8a771b2c30eb1bd2893ebebafb5789ff8d646a9c66de6e95f6e94a`

B4b artifacts:

- `b4b_result.json`: `bd0c9d1838e0a1273c95b1b596dacc523f4fa4eeb8e5ab8021a9710a1fa756b2`
- `b4b_steady_result.json`: `367b9c3b451f1fa519df597d10726008281e88af34f5b3778dda8a05ee9659c4`
- `b4b_runtime.pt.trace.json`: `2dbe88db4055ce78268995375af5425a92c239b878b98f33a7e16ec67c61259a`
- `b4b_trainer.log`: `ea67ef9884c90d8c1cf0e8fd4c816463e32f3b2b35c00c224adce0a9f79ddf02`

20260827 UTC B4c 20 communication SMs is rejected. It averaged 12,497.3
tok/s/GPU versus B3's 12,509.2, while HybridEP payload/fused-permute kernel
time rose from 15.113 to 15.760 ms. Its 203.083 ms layer span is within
cross-trace DSA variance and is not supported by the directly targeted kernel
metric. The GLM provider's 16-SM setting remains best.

B4c artifacts:

- `b4c_result.json`: `028e74301d8acb764babc795874543759da728d23d4c50a30ab738792d599d54`
- `b4c_steady_result.json`: `32217057f3d5ddede3f55e092d2c99fb1fc1ae021354a64e8ff4337018b7248d`
- `b4c_runtime.pt.trace.json`: `9a1d806517204530dcdda68b74c10a0b97ce1b407409398263d290a88d97d523`
- `b4c_trainer.log`: `5fea23ee508fa45c4e76f3867f17ffa35c067a0dd57b43ebf18a178495e8ee6c`

20260827 UTC B4d matched chunk size 128 is rejected. It averaged 12,479.0
tok/s/GPU and 1.31293 s FB, while HybridEP payload/fused-permute time rose to
16.168 ms versus B3's 15.113 ms at the default matched chunk size 64. Its
203.985 ms layer span is within cross-trace DSA variance and does not override
the directly targeted kernel regression. B4 is complete: 16 communication SMs,
108 fused blocks, and matched 64-token chunks remain optimal.

B4d artifacts:

- `b4d_chunk128.env`: `3ccb1363af6b4e197bd7257a2a8a28631d36d6a74706b7f5cf5071658a8fd7db`
- `b4d_result.json`: `b739323ee71231611ef2462feeb7db9c8d930d19074ee0296ec377abc41c3b3f`
- `b4d_steady_result.json`: `840114bc73eb7ca81e50fbee2691a57c6587d82367c5953a45a11f8044258426`
- `b4d_runtime.pt.trace.json`: `4677c6c39166788673253ec0737e105e7a8e86d6177abc993fddc066624807da`
- `b4d_trainer.log`: `f6984e5b44c86e0225fd3e46ddbaa5a34841ccf8515e56d121b4c365e7443df3`

20260827 UTC Pushed trivial-router-group elision at exact SHA
`3d8ab9e490297fe0320610cf7518d4276175f7c9`. The default-off guard only accepts
providers whose group count and selected-group count are exactly 1/1, then
normalizes both to `None`; all nontrivial grouping is rejected. Models tests
13/13, focused lint/format, and repository pre-push `make check` passed. B5 is
authorized to profile this pushed SHA.
