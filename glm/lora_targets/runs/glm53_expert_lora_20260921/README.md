# GLM DSA routed-expert LoRA

Select `moe_lora_config: shared_adapter` or `shared_outer` to add routed
expert `linear_fc1` (gate/up) and `linear_fc2` (down) to GLM DSA's existing
LoRA targets. Leaving the field unset retains the existing target list.

`shared_adapter` shares both factors among experts local to an expert-parallel
rank; it does not promise one globally identical adapter across every rank.
`shared_outer` shares the hidden-facing factors globally and retains
expert-specific intermediate-facing factors. Both leave base weights frozen.

## Validation protocol

1. Extract layer 0 and the first routed-expert layer using `prepare_debug.py`.
2. Train with TP1/PP1/EP8/CP8/ETP1, native FP8 expert storage, rank/alpha 32.
3. Run a forward/backward + optimizer smoke, save/load optimizer checkpoint,
   and export the sampler adapter. Check that expert LoRA-B weights changed
   from their zero initialization.
4. Serve with vLLM 0.29.0 and matching expert LoRA targets. Shared outer also
   requires `enable_moe_shared_loras=True` and artifact normalization through
   `sampler.proxy.lora_format.ensure_flat_3d_moe_layout`.
5. Repeat with the full checkpoint, measuring a matched no-routed-expert baseline.

The existing GLM `glm52` sampler preset excludes routed experts. These
experiments explicitly target `fused_qkv_a_proj`, `q_b_proj`, `o_proj`,
`gate_up_proj`, `down_proj`, `experts`, and `lm_head`.

Training TPS/GPU uses the supplied `profile_driver.py`: tokens divided by
forward/backward wall time and GPU count, with optimizer time reported
separately. Its MFU/HFU formulas assume the full model and must be ignored for
the two-layer model. Sampler output TPS/GPU is a separate metric.

## Progress

- Actual GLM-5.3 revision `aca966e4e02791568aa6a4ced368624b3d897f42`, two-layer
  extraction: both shared-adapter and shared-outer training, checkpoint
  save/load, and sampler export passed on 8 B300 GPUs.
- Native FP8 validation now unwraps the LoRA base linear; 20 focused trainer
  unit tests pass.
- GLM logits patch forwards newer vLLM arguments; 11 focused sampler tests pass.
- Preliminary GLM-5.2 two-layer shared-adapter sampler load/generation passed.
- Both actual GLM-5.3 debug samplers loaded and generated successfully.
- Both full-model layouts passed training, checkpoint save/load, and export.
- Both full-model samplers loaded and generated successfully. All 75 shared-outer
  MoE layers passed the artifact-shape check; the vLLM packer accepted the layout.

## Full-model training measurements

Same model revision above, 8 B300 GPUs, one 131,072-token synthetic datum per
step, rank/alpha 32, native FP8, TP1/PP1/EP8/CP8/ETP1, full uniform layer
recomputation. One warmup followed by five unprofiled control steps. Measurements
include HTTP submission/wait time; optimizer time is reported separately.

| Layout | Mean-window TPS/GPU | Median-window TPS/GPU | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Baseline (routed experts excluded) | 1,340 | 1,353 | 158.7 |
| shared_adapter | 1,131 | 1,134 | 167.2 |
| shared_outer | 1,057 | 1,141 | 175.6 |

Shared outer's first control window took 20.4 seconds versus 13.7–14.9 seconds
for its remaining controls. Its lower mean is sensitive to that window; these
runs do not establish a meaningful steady-state speed difference between the
two expert layouts. See `training_results.json` for all timings and gradients.
Synthetic-token loss changes are a smoke test, not a quality comparison.

Full-model shared-adapter export is about 29 GiB because the serving artifact
repeats each EP-local shared pair under individual expert names.

## Full-model sampler measurements

vLLM 0.29.0, TP8, Marlin FP8 weight-only kernels, rank 32, max LoRAs 1,
max sequences 32, max context 8,192, eager execution, prefix caching disabled.
The existing GLM launcher enabled 5-token MTP speculation on the full model.
Each measured request contains 16 identical prompts and generates exactly
64 tokens per prompt. Two warmups and five measured requests per arm.

| Engine layout | Base model (no adapter) output TPS/GPU | Loaded trained adapter output TPS/GPU |
| --- | ---: | ---: |
| shared_adapter | 13.49 | 11.31 |
| shared_outer | 14.96 | 11.78 |

These are end-to-end completion rates at this small batch size, not maximum
serving capacity and not the training TPS/GPU metric. The two independently
started engines also differ in base-model speed; do not interpret the small
adapter-rate difference as proof that one layout is faster.

## Numerical follow-up

Loading and generation are verified, but numerical parity is **not** established.
An additional fixed-token ablation (`expert_effect.py`) zeros only routed-expert
LoRA-B tensors while retaining the other adapters. At the trained scale, the
maximum score change was 0.430 nats, against 0.312 nats of repeated-request
variation, failing the diagnostic's 10x-noise guard. The base model without any
adapter also varied by 0.346 nats between identical requests on this engine.

Diagnostic-only expert-B scaling by 100x and 1000x produced larger changes
(1.918 and 6.888 nats) but still did not pass the deliberately strict guard
(repeat variation 0.467 and 0.789 nats). These synthetic adapters were never
used for throughput numbers or training. This is evidence of a remaining
reproducibility/parity investigation, not a successful parity result.

Trying `VLLM_BATCH_INVARIANT=1` failed during startup: the pinned vLLM reports
`FP8 Marlin not supported for batch invariant execution`. The working ordinary
engine configuration was restored. This flag is not a supported fix on the
tested stack.

## Reproducing

Use `trainer_config.py` with explicit model, checkpoint, and export directories,
then launch with the devbox's generated `start_trainer.sh` and health waiter.
Run the supplied profile driver with `--seq-len 131072 --datums 1 --num-gpus 8
--control-repeats 5`. Save and reload training state before exporting.

For the sampler, use its generated launcher with the expert target list above,
`ENABLE_LORA=1`, `MAX_LORA_RANK=32`, `MAX_LORAS=1`, `MAX_NUM_SEQS=32`,
`LOAD_FORMAT=auto`, `GPU_MEMORY_UTILIZATION=0.8`, and extra arguments
`--max-model-len 8192 --enforce-eager --no-enable-prefix-caching`.
For shared outer, also set `ENABLE_MOE_SHARED_LORAS=1` and normalize the
artifact using `sampler/scripts/check_shared_outer_parity.py --normalize`.
`sampler_bench.py` records completion timing and sample log probabilities.

## Dependencies

This branch includes Lucy's shared-outer implementation and test harness,
including Megatron Bridge PR #86. Those changes remain visible in the diff
against main until the prerequisite work lands.
