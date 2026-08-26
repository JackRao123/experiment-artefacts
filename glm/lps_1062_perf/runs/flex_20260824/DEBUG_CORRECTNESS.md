# EP8 backend debug correctness

## Verdict

- The three runs are numerically equivalent over all five windows. Maximum cross-run loss spread is `2.86e-6`; maximum grad-norm spread is `3.29e-9` (`0.00130%`). There is no divergent window or backend-specific drift.
- No dropped-token message/metric, NaN/Inf, traceback, CUDA error, or NCCL warning appears. Every run completed step 5. Each step reports `num_tokens=65536` and `num_loss_tokens=65528`; the fixed eight-token difference is consistent with one masked/non-loss token per datum and is not an emitted MoE drop count.
- `sms20` is fastest and most stable in the three control windows, but the sample is small and the logs contain timing-relevant profiler/autograd warnings.
- Backend and DeepEP SM settings are **not emitted by `trainer_srun.log`**. The directory labels indicate intended `alltoall`, `deepep/sms16`, and `deepep/sms20` settings, but effective selection cannot be independently verified from these logs. The common `trainers-deepep` source path appears even in the all-to-all log and is not proof of the active backend.

## Configuration evidence

All `result.json` files agree on 8 GPUs, sequence length 8192, 65,536 tokens/window, `TP=1`, `PP=1`, `EP=8`, `CP=1`, and `ETP=1`; all logs initialize `TP=1`, `PP=1`, and seed 1234. No backend name or SM count is printed.

| Run | Intended from label | Effective backend/SM evidence in log |
|---|---|---|
| `debug-ep8-alltoall` | all-to-all | Not emitted |
| `debug-ep8-deepep-sms16` | DeepEP, 16 SMs | Not emitted |
| `debug-ep8-deepep-sms20` | DeepEP, 20 SMs | Not emitted |

## Numerical comparison

Values are `alltoall / sms16 / sms20`; spread is max minus min across runs.

| Window | Loss | Loss spread | Grad norm | Grad spread |
|---|---:|---:|---:|---:|
| warmup 0 | `11.950906483 / 11.950904575 / 11.950903621` | `2.86e-6` | `2.542902657e-4 / 2.542887814e-4 / 2.542877919e-4` | `2.47e-9` (`0.00097%`) |
| traced 0 | `11.950767229 / 11.950767229 / 11.950767229` | `0` | `2.539544948e-4 / 2.539546695e-4 / 2.539562993e-4` | `1.80e-9` (`0.00071%`) |
| control 0 | `11.950846394 / 11.950846394 / 11.950847348` | `9.54e-7` | `2.542135480e-4 / 2.542141301e-4 / 2.542165748e-4` | `3.03e-9` (`0.00119%`) |
| control 1 | `11.950973248 / 11.950972294 / 11.950972294` | `9.54e-7` | `2.540935529e-4 / 2.540941641e-4 / 2.540968417e-4` | `3.29e-9` (`0.00129%`) |
| control 2 | `11.950873100 / 11.950872146 / 11.950871192` | `1.91e-6` | `2.541770518e-4 / 2.541800495e-4 / 2.541789145e-4` | `3.00e-9` (`0.00118%`) |

These deltas are consistent with harmless floating-point reduction-order differences.

## Timing variance

Population standard deviation and CV use the three untraced control windows only.

| Run | FB times (s) | Mean +/- SD (s) | CV | Server-step mean / CV | Control TPS/GPU |
|---|---:|---:|---:|---:|---:|
| all-to-all | `1.2566, 1.1606, 0.8346` | `1.0839 +/- 0.1806` | `16.66%` | `0.9913 / 18.96%` | `7,558` |
| DeepEP sms16 | `0.9150, 1.0493, 0.9547` | `0.9730 +/- 0.0564` | `5.79%` | `0.8780 / 5.84%` | `8,419` |
| DeepEP sms20 | `0.8699, 0.8506, 0.8383` | `0.8529 +/- 0.0130` | `1.52%` | `0.7634 / 2.45%` | `9,604` |

- Relative to all-to-all, sms16 reduces mean FB time `10.2%` and raises TPS/GPU `11.4%`; sms20 reduces mean FB time `21.3%` and raises TPS/GPU `27.1%`.
- sms20 is `12.3%` lower mean FB time and `14.1%` higher TPS/GPU than sms16.
- Optimizer timing is also steadier for sms20 (`7.3-10.3 ms`) than all-to-all (`11.9-50.5 ms`) or sms16 (`11.5-45.5 ms`).
- The single traced window is noisy: reported overhead is `+10.0%`, `-5.8%`, and `+28.9%`, respectively. The profiler explicitly warns that it has no warmup, so these values should not drive ranking.

## Log findings

- Each run emits one nonfatal profiler error: `External init callback must run in same thread as registerClient`. Profiling nevertheless starts, stops, and writes one trace file.
- Each run warns that `AccumulateGrad` producer/consumer streams differ, which may add synchronization and break CUDA graph capture. This is performance-relevant and may contribute to variance, but is common to all runs.
- Common startup warnings include missing indexer checkpoint keys, Apex absence with Torch Norm fallback, barrier device inference, deprecations, and experimental APIs. None differs materially by run, and all requests return successfully.
- `final_status.last_loss` and `final_status.grad_norm` are null in all three results despite complete per-window metrics; this is a status-reporting limitation, not a run failure.
