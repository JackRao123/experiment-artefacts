# GLM-5.3 routed-expert LoRA

## 20260921 18:55 PDT

- Devbox: `tj-wgpn4vw`, 8 B300 GPUs. Trainer/sampler processes run sequentially on all 8 GPUs.
- PR: https://github.com/basetenlabs/trainers/pull/1457 (draft, target main).
- Local implementation worktree: `/var/folders/1m/bllgmvfs6t7czgc4w3l_h7f00000gn/T/opencode/glm53-expert-lora`.
- Remote worktree: `/root/.cache/user_artifacts/devboxes/wgpn4vw/trainers`.
- Initial environment had empty virtual environments. Built trainer CUDA 13 dependencies and sampler shared-outer/vLLM 0.29.0 stack.
- Initial cached debug smoke used GLM-5.2; all such measurements are labeled `glm52` and are not GLM-5.3 results.
- Downloaded actual GLM-5.3 revision `aca966e4e02791568aa6a4ced368624b3d897f42`; derived 1d1m using layer 0 and first routed MoE layer.
- Both layouts on actual GLM-5.3 1d1m passed forward/backward, optimizer, checkpoint save/load, and export using native FP8 expert storage, TP1/PP1/EP8/CP8/ETP1, rank/alpha 32.
- Both debug adapters loaded into vLLM and generated 16 x 64-token requests. Shared-outer artifact passed vLLM packer/shape checker.
- Fixed GLM fp32 logits patch forwarding for the extra arguments introduced in newer vLLM. 11 focused sampler tests pass.
- FP8 storage verification unwraps the LoRA base module. 20 focused trainer tests pass before final type-safe test cleanup.
- Full-model matched baseline is starting. Both full-model layout runs and full samplers remain pending.

## Measurement definitions

The supplied `profile_driver.py` records training forward/backward wall-clock TPS/GPU; optimizer time is separate. No profiler is enabled in headline runs. Its MFU/HFU values assume the full GLM geometry and are invalid for 1d1m. Sampler records are output-token TPS/GPU for fixed prompt batches and are not comparable to the ~1,500 training TPS/GPU reference.

## 20260921 20:15 PDT

- Full GLM-5.3 baseline, shared_adapter, and shared_outer training completed at 131k tokens; both expert layouts also saved, reloaded, and exported successfully.
- Mean training TPS/GPU: baseline 1339.59, shared_adapter 1131.00, shared_outer 1057.47. Shared outer's first measured window was slow; median TPS/GPU is 1140.87 versus shared_adapter 1133.82. No claim of a meaningful speed difference between layouts.
- Full samplers loaded and generated for both layouts. Batch 16, 64 output tokens, TP8, eager, prefix caching off, launcher-default 5-token MTP: shared_adapter 11.31 output TPS/GPU; shared_outer 11.78. These are small-batch end-to-end rates, not saturation throughput.
- Numerical follow-up remains unresolved: ordinary expert-B zeroing diagnostic max change 0.430 nats, repeated-request variation 0.312 nats. Base model also varied by 0.346 nats. Synthetic 100x/1000x expert-B probes increased the effect but failed the strict 10x-noise threshold; raw failures retained, never used for throughput.
- Batch-invariant mode failed at startup because FP8 Marlin is unsupported in that mode. Restoring ordinary full shared-outer sampler; PR remains draft with this limitation documented.
- Raw training/sampler JSONs and configs are copied into this run directory; summaries and reproducible helpers are in the PR's experiments/glm_expert_lora directory.

## 20260921 20:17 PDT

- Restored full shared-outer sampler under the normal (non-batch-invariant) settings. Healthy on port 8000; reloaded the original unamplified `glm53-shared-outer-full` adapter and received a successful 8-token completion.
- Trainer is stopped. Devbox, worktree, checkpoints, exports, and benchmark artifacts are preserved.
- Latest code/results push: `d184a0690`. Pre-push lint/format/type checks pass; focused trainer tests rerun successfully (20 passed), sampler tests (11 passed).
- PR remains draft due to the explicit numerical-parity limitation; all requested operational smoke tests and training/serving throughput measurements are recorded.
