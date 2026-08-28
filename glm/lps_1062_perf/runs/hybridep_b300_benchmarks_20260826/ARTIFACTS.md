# Artifact manifest

Laptop directory:

`/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/hybridep_b300_benchmarks_20260826`

## Benchmark outputs

| case | trainer SHA | driver output | runtime trace | trace size |
|---|---|---|---|---:|
| 1 main debug CP16/EP16 | `061947aca8bc00540ee7ad9c7c3c8d7f0df68e20` | `case1_result.json` | `case1_runtime.pt.trace.json` | 8,075,927 bytes |
| 2 main debug CP8/EP8 | `061947aca8bc00540ee7ad9c7c3c8d7f0df68e20` | `case2_result.json` | `case2_runtime.pt.trace.json` | 6,234,978 bytes |
| 3 main full model | `061947aca8bc00540ee7ad9c7c3c8d7f0df68e20` | `case3_result.json` | `case3_runtime.pt.trace.json` | 735,950,130 bytes |
| 4 PR 1150 HybridEP | `7210af32bd4945e0f7772c9196ca069e35cd1450` | `case4_result.json` | `case4_runtime.pt.trace.json` | 700,203,252 bytes |

## Reproduction inputs

- `case1_debug_cp16.json`
- `case2_debug_cp8.json`
- `case3_full_main.json`
- `case4_full_hybridep.json`
- `server_config.json`
- `profile_driver.py`
- `mfu.py`

## Logs and provenance

- `case1_trainer.log` through `case4_trainer.log`: successful trainer logs.
- `case1_startup.log`: rejected obsolete config-field attempt.
- `case1_startup_attempt2.log`: successful case 1 startup log checkpoint.
- `case4_startup.log`: HybridEP JIT failure before the shared CUDA launcher fix.
- `WORKLOG.md`: chronological procedure, exact arguments, results, caveats, and environment repair.
- `SHA256SUMS`: checksums for every driver result, runtime trace, and copied tool.

All four result/trace pairs were copied from the B300 devbox to this directory.
The local trace byte sizes match the sizes returned by `profile_driver.py`.
No memory profiles were requested or produced.
