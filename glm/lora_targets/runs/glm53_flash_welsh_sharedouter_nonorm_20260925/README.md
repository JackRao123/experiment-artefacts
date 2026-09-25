# GLM-5.3-Flash Welsh shared-outer, nonorm (50 steps)

- [W&B run](https://wandb.ai/baseten-training/lora-target-baselines/runs/oy66rhf5): finished 50/50 optimizer steps, exit code 0; final training NLL `0.1380546242`.
- Model: `zai-org/GLM-5.3-Flash`, cached snapshot `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`. Trainer revision `9c395a676bf20866abd79108996667c90ac63d06` plus the devbox's existing local `no_qbproj` overlay. Config has `moe_lora_config="shared_outer"`, `canonical_gu=false`, and `no_qbproj=false`.
- Eight B300 GPUs, TP1/PP1/EP8/CP4/ETP1, LoRA rank/alpha 32. The client used `--no-loss-normalization`, 50 steps, and the original 390-step learning-rate horizon on the [Welsh baseline dataset](../glm53_flash_welsh_baseline_20260924/README.md).
- Ran 2026-09-25 15:58:05–16:11:07 UTC (782 seconds). This is client time, excluding trainer startup. Training throughput was 9,623 input tokens/s over all 50 measured batches, or 10,564 tokens/s after the first three warm-up batches.
- Final resumable training state and sampler adapter are on `tj-wgpn4vw` at `/root/.cache/user_artifacts/devboxes/wgpn4vw/exports/glm53flash-welsh-sharedouter-nonorm-50/{weights,sampler_weights}/final`. The adapter binary stays on the devbox; the trainer was stopped afterward and all eight GPUs were idle.
- The reproducible trainer config and client launcher are archived alongside this note. This run is distinct from the [canonical-GU + nonorm variant](../glm53_flash_welsh_sharedouter_canonicalgu_nonorm_20260925/README.md).
