# GLM-5.3 canonical gate/up trainer smoke

PR: https://github.com/basetenlabs/trainers/pull/1457

Implementation: `b3863e2f0f17c724565049978c9a9329a2037c57` — `canonicalgu:: split dense and shared-expert gate/up LoRA`.

## Policy

- `--canonicalgu` selects independent gate/up A and B in dense and always-active shared-expert MLPs.
- Routed experts retain fused gate/up (shared A, distinct B slices) with the existing selected MoE layout.
- `--sharedouter` selects `moe_lora_config=shared_outer` and opts GLM routed experts in. The baseline target list without an explicit layout still excludes routed experts.
- MLA/attention targets are unchanged.
- Flags belong to `trainers_server_main.main` / `server-main/scripts/launch.sh`. The independent sft-baselines-loops launcher was not modified.

The devbox generated launcher takes config files, so these runs used the equivalent fields `canonical_gu: true` and `moe_lora_config: null` / `shared_outer` in the adjacent JSON files.

## Workload

Full GLM-5.3, 78 layers, HF revision `aca966e4e02791568aa6a4ced368624b3d897f42`; `tj-wgpn4vw`, 8 B300 (reported as L20D). Rank/alpha 32; TP1/PP1/EP8/CP8/ETP1; native FP8 expert storage; full uniform recomputation; one 131,072-token random datum per step. One warmup and exactly three controls per variant.

```sh
python profile_driver.py --label glm53-full-canonicalgu-131k --seq-len 131072 --datums 1 --num-gpus 8 --lora-rank 32 --control-repeats 3
python profile_driver.py --label glm53-full-canonicalgu-sharedouter-131k --seq-len 131072 --datums 1 --num-gpus 8 --lora-rank 32 --control-repeats 3
```

Driver: `/root/.cache/user_artifacts/devboxes/wgpn4vw/profile_driver.py`; local source in `glm/lps_1062_perf/tools/profile_driver.py`.

## Results

Training throughput is forward/backward tokens/s/GPU, excluding optimizer time. Mean throughput uses total control tokens divided by total control forward/backward time.

| Variant | Controls | Mean TPS/GPU | Median-window TPS/GPU | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Historical fused baseline, routed experts excluded | 5 | 1339.59 | 1352.92 | 158.74 |
| Canonical GU, routed experts excluded | 3 | 1265.13 | 1305.50 | 159.00 |
| Historical fused shared-outer | 5 | 1057.47 | 1140.87 | 175.59 |
| Canonical GU + shared-outer | 3 | 923.44 | 1038.90 | 175.72 |

Canonical GU control times: 13.889, 12.413, 12.550 seconds. Canonical GU + shared-outer: 22.315, 15.771, 15.142 seconds. Shared-outer's first control is slower in both historical and new runs; these short, non-interleaved runs do not establish a precise steady-state regression.

Both variants completed four optimizer steps with finite losses and gradient norms. Checkpoint save and sampler-format export succeeded for both; responses are in the adjacent `*-save.json` files. Reload and serving were not exercised in this run.

Inspected the actual exported safetensors:

- Each variant has 78 dense/shared-expert gate A tensors, and all 78 differ from their corresponding up A tensors (3 dense layers + 75 shared experts).
- Shared-outer retains 75 routed shared gate A tensors, each shaped `[1, 32, 6144]`; all 75 are identical to their paired routed up A tensors. Expert-specific gate/up B tensors remain separate, shaped `[2048, 32]`.

## Checks and final state

- No test files added or modified in the implementation commit.
- Existing `server-main/tests/unit/test_main.py`: 9 passed.
- Full pre-push `make check`: passed after repairing incomplete local venv installations.
- Ordinary push succeeded; PR head is the implementation commit above.
- Trainer stopped after export; GPUs released, devbox/checkpoints/exports preserved.
- Remote worktree contents verified byte-identical to the pushed commit and its detached HEAD aligned to that commit. Pre-existing untracked venv symlinks retained.

## Welsh SFT: 50-step W&B run

Completed: https://wandb.ai/baseten-training/lora-target-baselines/runs/n2p3guk6

- Name: `GLM-5.3-welsh-validation50-sharedouter-canonicalgu-baseten`.
- W&B state verified as `finished`, step 50; client exit code 0.
- Started 2026-09-23 19:39:46 UTC; finished 19:54:54 UTC (908 seconds).
- Fresh trainer at the same implementation commit and HF revision as the smoke above; no smoke-trained weights resumed.
- Launcher: `run_validation_50.sh --backend baseten --trainer-url http://127.0.0.1:8001 --trainer-config /root/.cache/user_artifacts/devboxes/wgpn4vw/welsh50-canonicalgu-sharedouter-trainer-config.json --sharedouter --canonicalgu`.
- Environment: `HF_MODEL_ID=zai-org/GLM-5.3`, `RENDERER=glm5_3_max_reasoning`.
- Frozen Welsh dataset and schedule: rank 32, batch 128, max example length 4096, original 390-step LR horizon, 50 training steps; 6,452,831 elapsed tokens.
- All 50 losses/gradient norms finite; first-batch NLL 0.3253377974, final-batch NLL 0.1574802548. These are training metrics on different batches, not a held-out quality comparison.
- Final training checkpoint: `/root/.cache/user_artifacts/devboxes/wgpn4vw/exports/glm53-welsh-canonicalgu-sharedouter-50/weights/final`.
- Final sampler export: `/root/.cache/user_artifacts/devboxes/wgpn4vw/exports/glm53-welsh-canonicalgu-sharedouter-50/sampler_weights/final`.
- Raw metrics, cookbook config, code diff, logs, completion status and checkpoint manifest are the adjacent `welsh50-*` files / `welsh50.status`.
- Trainer stopped after completion; devbox and final artifacts retained.

Updated the local and devbox copies of the independent `sft-baselines-loops` launcher to accept standalone canonical GU in local mode and validate `canonical_gu` against the trainer config. Local QKV splitting remains unsupported; hosted registry selection still requires the existing combined canonical variant. The new run gets a distinct `sharedouter-canonicalgu` log path. No tests were added or modified. Launcher SHA-256: `aecfbbc8a0ca9be9ebe676de30ace3d6e398896df7aab8a041149ec2570c4e34`. Its changes remain uncommitted in that separate repository, alongside pre-existing edits.
