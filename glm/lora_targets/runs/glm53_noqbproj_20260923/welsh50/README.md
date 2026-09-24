# GLM-5.3 Welsh 50-step shared-outer + no-qbproj + nonorm

- [W&B run](https://wandb.ai/baseten-training/reproduce-nvfp4-vs-tinker-loops-baselines/runs/pyiq93in): `GLM-5.3-welsh-validation50-sharedouter-noqbproj-nonorm-baseten`, finished 50/50 steps (history step 49 is zero-indexed).
- Fresh trainer on `tj-wgpn4vw`, base GLM-5.3 snapshot
  `aca966e4e02791568aa6a4ced368624b3d897f42`; LoRA rank/alpha 32,
  `moe_lora_config=shared_outer`, `no_qbproj=true`, `canonical_gu=false`.
  The preceding 131k-token profiling trainer had 13 optimizer steps and was
  stopped before this fresh start; none of its trained weights were reused.
- `skip_token_normalization=true` was sent on each forward/backward request
  via `--no-loss-normalization`. The original 390-step LR horizon was retained.
- Started 2026-09-23 23:19:16 UTC and finished 23:34:19 UTC: **903s (15m03s)**.
- Training mean NLL: step 1 **0.3253377974**, step 50 **0.1354194432**.
  Those are different training batches, not held-out quality measurements.
- Final resumable state on devbox:
  `/root/.cache/user_artifacts/devboxes/wgpn4vw/exports/glm53-welsh-sharedouter-noqbproj-nonorm-50/weights/final`.
- Final sampler adapter on devbox:
  `/root/.cache/user_artifacts/devboxes/wgpn4vw/exports/glm53-welsh-sharedouter-noqbproj-nonorm-50/sampler_weights/final`
  (7,971,678,864-byte adapter file). Neither artifact was copied to this
  local directory; only the run manifest, metrics, and config are here.
- W&B API verified `finished`; the fresh trainer remained healthy at step 50.
