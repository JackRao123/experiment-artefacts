# GLM-5.3-Flash Welsh shared-outer, canonical-GU, nonorm (50 steps)

- [W&B run](https://wandb.ai/baseten-training/lora-target-baselines/runs/tkcik4s7): finished 50/50 optimizer steps, exit code 0; final training NLL `0.1387756914`.
- Model: `zai-org/GLM-5.3-Flash`, cached snapshot `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`. Trainer revision `9c395a676bf20866abd79108996667c90ac63d06` plus the devbox's existing local `no_qbproj` overlay. Config has `moe_lora_config="shared_outer"`, `canonical_gu=true`, and `no_qbproj=false`.
- Eight B300 GPUs, TP1/PP1/EP8/CP4/ETP1, LoRA rank/alpha 32. The client used `--no-loss-normalization`, 50 steps, and the original 390-step learning-rate horizon on the [Welsh baseline dataset](../glm53_flash_welsh_baseline_20260924/README.md).
- Ran 2026-09-25 00:58:42–01:11:50 UTC (788 seconds). This is client time, excluding trainer startup. The final checkpoint and sampler adapter are on `tj-wgpn4vw` at `/root/.cache/user_artifacts/devboxes/wgpn4vw/exports/glm53flash-welsh-sharedouter-canonicalgu-nonorm-50/{weights,sampler_weights}/final`.
- The reproducible trainer config and client launcher are archived alongside this note. The large checkpoint and adapter binaries stay on the devbox. The trainer was stopped after this run to start the separate non-canonical variant.
