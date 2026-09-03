# GLM-5.2 FP8 base-training experiment

## Scope

- Base trainer SHA: `f1f34437d2d9e81eaf42110097bb26ff600c7e1e` (`main`).
- Branch: `jackrao/lps-1062-glm52-vpp-balanced`; draft PR #1210 retargeted to
  FP8 base training.
- Local worktree: `/Users/jackrao/Documents/trainers-wt-lps1062-glm52-vpp-balanced`.
- Devbox: `tj-w5y89m3`, 2 nodes x 8 B300 GPUs.
- Target topology: TP1/PP2/CP8/EP8/ETP1/DP1.
- Target workload: GLM-5.2-FP8 LoRA SFT, sequence length 131072, four datums.
- Precision contract: Transformer Engine tensorwise FP8 GEMMs and FP8 eligible
  frozen base parameters; BF16 residual/pipeline activations and BF16 LoRA
  adapters; BF16 custom DSA; existing FP32 output head.

## Log

20260827 20:18 PDT Reused prior same-hardware evidence from the
`transformer_block_max_mfu_20260827` campaign: tensorwise current-scaling FP8
beat MXFP8 on the exact B300 block shape, while MXFP8 required an aligned startup
sequence and did not improve throughput. Selected tensorwise FP8 as the first
full-base candidate.

20260827 20:18 PDT Added runtime fields `fp8_recipe` and `fp8_param`, plus a
BF16 mixed-precision builder that enables TE FP8 while leaving DSA attention and
ordinary activations out of FP8. Pushed exact candidate SHA
`5877ad3a3` pending GPU validation.

20260827 20:26 PDT Debug startup attempt 1 stopped before model construction
because `BT_TRAINER_SERVER_CONFIG_PATH` was omitted from the devbox launch
environment. Relaunched unchanged code/config with the run-local server config.

20260827 20:29 PDT The one-layer PP1/CP8/EP8 HybridEP proxy completed five real
forward/backward and optimizer windows at exact trainer SHA
`5877ad3a36ef1dc058eb5f4cc91f55a6c4fc73cf`. Loss stayed finite around 11.95
and grad norm stayed nonzero around 0.00072. The runtime trace contains E4M3
quantization, amax, and quantized NVJet GEMM kernels while cuDNN DSA remains
BF16. Result: `debug_result.json`; trace: `debug_runtime.pt.trace.json`.

20260827 20:48 PDT Full GLM-5.2 PP2/CP8/EP8 HybridEP FP8-parameter startup
completed on both B300 nodes. Before real 131K steps, the two stages occupied
approximately 56 and 66 GB/GPU. This is direct steady-residency evidence that
eligible base parameters are retained in FP8 rather than merely quantized from
BF16 for each GEMM.

20260827 20:55 PDT The first full-model measurement completed at exact trainer
SHA `5877ad3a36ef1dc058eb5f4cc91f55a6c4fc73cf`. Three-control headline was
929 tok/s/GPU; controls 1-2 were 955 and 957 tok/s/GPU. Runtime-profile FB was
34.3 s. Loss and grad norm remained finite. Rank-0 allocated memory was 54.84
GB before the first driver step and 97.39 GB after five real steps. Result:
`full_result.json`; trace: `full_runtime.pt.trace.json`.

20260827 21:01 PDT Trace attribution against the prior BF16 HybridEP case shows
quantized NVJet GEMMs at 1.63 s, remaining non-FP8 NVJet at 1.38 s, and FP8
quantization/amax overhead at 0.86 s. The prior BF16 trace spent 5.24 s in
NVJet kernels, so the linear path saves about 1.37 s after quantization cost.
cuDNN DSA backward/fwd remain effectively unchanged at 3.25/1.13 s. Exposed
PP SendRecv falls from 11.2 to 9.4 s; HybridEP device synchronization rises
from 2.84 to 3.14 s.

20260827 21:07 PDT A same-code/SHA BF16 control completed five controls at
899.85 tok/s/GPU. Rank-0 allocated memory was 104.35 GB before the first driver
step and 149.10 GB after six real steps: FP8 parameters reduce the corresponding
rank-0 values by 49.51 and 51.71 GB. Result: `bf16_control_result.json`.

20260827 21:15 PDT Independent ten-control FP8 confirmation completed at the
same exact SHA. The all-control headline is 951.09 tok/s/GPU, +5.69% versus
matched BF16; controls 1-9 average 957.42 tok/s/GPU, +6.40%. Loss decreased
smoothly from 12.3141 to 12.2497 over optimizer steps and grad norm remained
finite/nonzero (0.3357-0.4078). Result: `full_steady_result.json`.

20260827 21:16 PDT Added the validated `tensorwise` / `fp8_param=true` fields to
the checked-in GLM-5.2 B300 128K golden config and benchmark projection. Final
PR head `97ebd1af4`; repository pre-push lint, format, and typecheck pass. The
targeted model/controller/registry/golden suites pass 161 tests. The focused
benchmark FP8 projection test passes; its full file has one unrelated current-
main failure because the exact H100 ID allowlist omits four checked-in DeepSeek
V4 cases (`pc_d444a6822f15`). Stopped the trainer through the generated devbox
lifecycle script; Slurm is empty and the leader has no GPU compute processes.

20260828 00:00 PDT Changed the checked-in B300 128K topology to
TP1/PP1/CP8/EP8 on one eight-GPU node and pushed exact trainer SHA
`7fe28a0b27cdb82c7bc4fa25b12c8b1e632053a3` to PR #1210. Model startup
occupied approximately 118.16 GB on rank 0 before driver steps, so the full
FP8 model fits comfortably without pipeline parallelism.

20260828 00:00 PDT PP1 five-control validation completed at the exact pushed
SHA. Headline throughput is 1297 tok/s/GPU with 50.5 s mean FB, versus 951
tok/s/GPU and 34.5 s for FP8 PP2. PP1 improves per-GPU throughput and
GPU-seconds/token by 36.4% while using half the GPUs; aggregate fleet throughput
is 31.8% lower because the step is 46.6% longer. Rank-0 allocation rises from
118.16 GB before driver steps to 179.91 GB after six real steps, still below the
288 GB decimal B300 capacity. Loss/grad norm remain finite. Result:
`pp1_steady_result.json`.

20260828 00:00 PDT PP1 attribution trace confirms zero NCCL SendRecv kernels.
The former 9.4 s exposed PP wait is eliminated. Dominant rank-0 kernels are DSA
backward 6.68 s, HybridEP device synchronization 5.40 s, DSA forward 2.32 s,
and the FP32 vocabulary head 3.73 s. FP8 plus remaining non-FP8 NVJet GEMMs
consume 6.22 s and quantization/amax 1.85 s across all 78 layers. Result:
`pp1_profile_result.json`; trace: `pp1_runtime.pt.trace.json`.

20260828 00:00 PDT Stopped the one-node trainer through the generated lifecycle
script. Slurm is empty and the leader reports no GPU compute processes. PR #1210
head remains exact pushed SHA `7fe28a0b2`.
