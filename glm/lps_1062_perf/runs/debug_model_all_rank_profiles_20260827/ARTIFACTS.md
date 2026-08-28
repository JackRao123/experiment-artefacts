# Artifact manifest

Laptop directory:

`/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/debug_model_all_rank_profiles_20260827`

## Run provenance

- Devbox: `tj-w5y89m3`, two nodes with eight B300 GPUs each.
- Trainers revision: PR 1157 tip
  `8209793c89ff6be2288b25668041bf27598bb0ce`.
- PR base/current main at run time:
  `2e829d2bc6f162820188b7762c293053cdbc8cad`.
- Model: shared local GLM-5.2 0D1M debug proxy.
- Both cases use sequence length 131072, one datum, LoRA rank 32, one warmup,
  three unprofiled controls, and one all-rank runtime-profile window.
- No memory profiles were requested.

## Outputs

| case | topology | profiled ranks | result | traces | total trace bytes |
|---|---|---|---|---:|---:|
| 1 | TP1/PP1/CP16/EP16/ETP1/DP1, 16 GPUs | 0-15 | `case1_result.json` | 16 | 128,111,715 |
| 2 | TP1/PP1/CP8/EP8/ETP1/DP1, 8 GPUs | 0-7 | `case2_result.json` | 8 | 49,342,799 |

Case 1 ranks 0-7 were copied from the leader and ranks 8-15 from the worker.
Case 2 used only the leader. The local filenames and combined byte counts
exactly match each API stop result.

## Files

- `case1_debug_cp16.json`, `case2_debug_cp8.json`: trainer configs.
- `server_config.json`: trainer server config.
- `profile_driver.py`, `mfu.py`: copied run tools. The driver selects
  `range(num_gpus)` in `POST /runtime_profile/start`.
- `case1_result.json`, `case2_result.json`: complete driver outputs.
- `case1_driver.log`, `case2_driver.log`: driver console logs.
- `case1_trainer.log`, `case2_trainer.log`: generated trainer job logs.
- `case1_traces/`, `case2_traces/`: rank-qualified Kineto traces.
- `WORKLOG.md`: procedure and result summary.
- `SHA256SUMS`: checksums for the preserved artifacts.
