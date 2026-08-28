# Artifacts

- Laptop directory: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm52_fp8_base_b300_20260827`
- Devbox: `tj-w5y89m3` (2 x 8 B300)
- PP2 benchmark SHA: `5877ad3a36ef1dc058eb5f4cc91f55a6c4fc73cf`
- FP8 golden-config head: `97ebd1af4`
- PP1 benchmark/final PR head: `7fe28a0b27cdb82c7bc4fa25b12c8b1e632053a3`
- Draft PR: https://github.com/basetenlabs/trainers/pull/1210
- Driver: `profile_driver.py`, copied byte-exactly from the experiment root.

## Configs

- `debug_tensorwise_fp8_param.json`: one-layer, one-node FP8 smoke test.
- `full_tensorwise_fp8_param.json`: target two-node FP8 configuration.
- `full_tensorwise_fp8_param_pp1.json`: winning one-node FP8 configuration.
- `full_bf16_control.json`: matched two-node BF16 control.
- `server_config.json`: trainer server identity for all cases.

## Results

- `debug_result.json`: debug FP8 result, finite loss/grad and 6702.5 tok/s/GPU.
- `debug_runtime.pt.trace.json`: debug Kineto trace, 5,999,023 bytes.
- `debug_trainer.log`: successful debug trainer lifecycle log.
- `full_result.json`: first full FP8 result with runtime profile.
- `full_runtime.pt.trace.json`: full FP8 Kineto trace, 882,350,527 bytes.
- `full_trainer.log`: first full FP8 trainer lifecycle log.
- `bf16_control_result.json`: matched BF16 five-control result.
- `bf16_control_trainer.log`: matched BF16 trainer lifecycle log.
- `full_steady_result.json`: independent ten-control FP8 confirmation.
- `full_steady_trainer.log`: ten-control FP8 trainer lifecycle log.
- `pp1_steady_result.json`: one-node five-control result, 1297 tok/s/GPU.
- `pp1_profile_result.json`: one-node runtime-profile result.
- `pp1_runtime.pt.trace.json`: one-node Kineto trace, 1,839,101,113 bytes.
- `pp1_trainer.log`: one-node trainer lifecycle log.

`profile_driver.py` reports full-model MFU correctly for the full-model cases;
its debug-model MFU is invalid because the helper assumes all 78 layers.
