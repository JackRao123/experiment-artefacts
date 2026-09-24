# GLM-5.3 Welsh SFT — full 390-step runs

Both runs use the hash-verified `baseten/wiki-translation-welsh-sft` dataset,
LoRA rank 32, and the original 390-step learning-rate schedule.

| Arm | W&B | Gradient normalization |
| --- | --- | --- |
| Tinker | [4519kmn7](https://wandb.ai/baseten-training/reproduce-nvfp4-vs-tinker-loops-baselines/runs/4519kmn7) | Default Tinker behavior |
| Baseten shared-outer | [4833opuq](https://wandb.ai/baseten-training/reproduce-nvfp4-vs-tinker-loops-baselines/runs/4833opuq) | Skipped (`skip_token_normalization: true`) |

`tinker/` contains the final sampler-weights checkpoint archive plus the
checkpoint manifest, metrics, and config. The Tinker training-state checkpoint
remains stored at the `tinker://` URI in `checkpoints.jsonl`: Tinker's archive
download API only supports `sampler_weights/` checkpoints.

`baseten/` contains only the final sampler adapter export plus the checkpoint
manifest, metrics, and config. The 53 GB resumable training-state checkpoint
stays on the devbox at
`/root/.cache/user_artifacts/devboxes/wgpn4vw/exports/glm53-welsh-sharedouter-nonorm-390/weights/final`.
