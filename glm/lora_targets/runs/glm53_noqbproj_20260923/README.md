# GLM-5.3 shared-outer, no `q_b_proj`, skip token normalization

Profiled on `tj-wgpn4vw` (8 B300 GPUs) with full GLM-5.3 revision
`aca966e4e02791568aa6a4ced368624b3d897f42`, LoRA rank/alpha 32,
TP1/PP1/EP8/CP8/ETP1, native FP8 expert storage, and full uniform
recomputation. The trainer uses `moe_lora_config=shared_outer`,
`no_qbproj=true`, `canonical_gu=false`. This removes the LoRA target for MLA
`q_b_proj` (`linear_q_up_proj` in Megatron Bridge), not the pretrained layer.

The driver used one 131,072-token synthetic datum per window, one warmup,
and five unprofiled controls. Every forward/backward response acknowledged
`skip_token_normalization=true`. The profiling trainer was subsequently
restarted from the base checkpoint for the separate [Welsh 50-step run](welsh50/README.md).

| Profile | Controls | Mean TPS/GPU | Median-window TPS/GPU | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| no_qbproj + shared_outer + nonorm, initial pass | 3 | 847.85 | 987.10 | 174.24 |
| **no_qbproj + shared_outer + nonorm, warm five-control pass** | **5** | **1,087.14** | **1,079.85** | **176.77** |

The five forward/backward control times were 14.866, 15.209, 13.963,
16.144, and 15.172 seconds. The headline is aggregate tokens / aggregate
forward/backward wall time / 8 GPUs. Optimizer time (~0.1s per control)
is excluded. The initial pass had a slow first control (26.125s), hence
the separate warm estimate. Raw records are in the adjacent JSON files.
See [the consolidated stability comparison](../GLM53_LORA_THROUGHPUT_STABILITY.md)
for all historical control windows and caveats.

## Earlier full-model LoRA measurements

Same full model, topology, sequence length, rank, and driver; earlier
measurements used default token normalization and different trainer revisions.
The numbers below are context, not a controlled measurement of the isolated
effect of removing `q_b_proj`.

| Layout | Controls | Mean TPS/GPU | Median-window TPS/GPU |
| --- | ---: | ---: | ---: |
| Historical fused baseline (routed experts excluded) | 5 | 1,339.59 | 1,352.92 |
| Canonical gate/up (routed experts excluded) | 3 | 1,265.13 | 1,305.50 |
| Shared-adapter routed experts | 5 | 1,131.00 | 1,133.82 |
| Shared-outer routed experts | 5 | 1,057.47 | 1,140.87 |
| Canonical gate/up + shared-outer | 3 | 923.44 | 1,038.90 |

Sources: [original full-model results](../glm53_expert_lora_20260921/README.md)
and [canonical gate/up results](../glm53_canonicalgu_20260923/README.md).

Trainer implementation branch `jackrao/glm53-noqbproj` at `2e50cf74b`;
SFT launcher/profile wrapper branch `jack/glm53-investigation` at `fe7c0f0`
plus the five-control follow-up commit. Both are local commits; no trainer
image or registry config was published for this experiment.
