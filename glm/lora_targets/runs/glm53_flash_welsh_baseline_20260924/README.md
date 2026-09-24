# GLM-5.3-Flash Welsh 50-step baseline

- [W&B run](https://wandb.ai/baseten-training/reproduce-nvfp4-vs-tinker-loops-baselines/runs/5no9qmjv): `GLM-5.3-Flash-welsh-validation50-baseline-baseten`, verified `finished` with 50 metric records (last W&B history step 49 is zero-indexed).
- Model: `zai-org/GLM-5.3-Flash`, cached Hugging Face snapshot
  `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (`glm5_next`). Baseten
  trainer on `tj-wgpn4vw`, 8 B300 GPUs, TP1/PP1/EP8/CP4/ETP1,
  native-FP8 expert storage and full recomputation.
- **Default LoRA targets:** rank/alpha 32, `moe_lora_config=null`,
  `no_qbproj=false`, `canonical_gu=false`; no shared-outer, no canonical,
  and normal token normalization. The final adapter has 22 LoRA tensors
  whose names contain `q_b_proj`, confirming that `q_b_proj` remained targeted.
- Same SHA-256-verified Welsh SFT dataset and original 390-step learning-rate
  horizon, capped at 50 optimizer steps. Text renderer:
  `glm5_3_max_reasoning` with Flash's own tokenizer.
- Started 2026-09-24 00:43:12 UTC; finished 01:00:17 UTC:
  **1,025 seconds (17m05s)**. First-batch training NLL: **0.3198957741**;
  last-batch training NLL: **0.1791443378**. These are different training
  batches, not held-out quality measurements.
- Final resumable training state on the devbox:
  `/root/.cache/user_artifacts/devboxes/wgpn4vw/exports/glm53flash-welsh-baseline-50/weights/final`.
- Final sampler adapter export on the devbox:
  `/root/.cache/user_artifacts/devboxes/wgpn4vw/exports/glm53flash-welsh-baseline-50/sampler_weights/final`
  (`adapter_model.safetensors` is 274,132,576 bytes). The adapter is retained
  locally but excluded from Git; its config and manifest are committed.

The model preset is implemented on the
[`jack/glm53-investigation`](https://github.com/basetenlabs/sft-baselines-loops/tree/jack/glm53-investigation)
branch of `sft-baselines-loops` and accepts `--model-preset glm-5.3-flash`
for a repeat run. The trainer source overlaid on the devbox at this time
starts from `b3863e2f0f17c724565049978c9a9329a2037c57`; its opt-in
`no_qbproj` field was explicitly false for this baseline.
