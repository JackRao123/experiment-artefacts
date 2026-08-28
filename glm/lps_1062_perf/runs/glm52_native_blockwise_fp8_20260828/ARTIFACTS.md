# Artifacts

- Run root:
  `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm52_native_blockwise_fp8_20260828`
- Devbox: `tj-w5y89m3`, 2 x 8 B300; experiments used one node.
- Candidate profiled trainers SHA:
  `daedf9ad610c09a5b9ec74d945a125fa7b1a518b`.
- Matched tensorwise SHA:
  `fd40df6663df9b0a2e80fbf0dbad7fc7244e1f5e`.
- Final trainers SHA: `4333166b5eed867b3778a28c15b50f0b871199fa`.
- Profiled Megatron-Bridge SHA: `6b5928d4f61c9ffbfcafa46bf64610fb1a046363`.
- Clean PR Megatron-Bridge SHA: `d64ee30fd18e038e6adf0395f068404441c6dcb5`.
- Trainers PR: https://github.com/basetenlabs/trainers/pull/1211
- Megatron-Bridge PR: https://github.com/basetenlabs/Megatron-Bridge/pull/53
- Shared driver: experiment-root `tools/profile_driver.py`, copied to the
  remote run byte-for-byte.

## Reproduction

- `build_native_debug_snapshot.py`: builds the real native-FP8 0d1m snapshot.
- `glm52_native_debug_config.json`: one-layer architecture config.
- `debug_blockwise_candidate.json`, `debug_tensorwise_control.json`: debug arms.
- `full_blockwise_candidate.json`, `full_tensorwise_control.json`: full arms.
- `run_profile_driver.py`: redirects generated JSON to node-local scratch when
  the project cache quota is full.
- `analyze_memory.py`: reconstructs peak allocated bytes from allocator events.
- `kernel_breakdown.sql`, `gpu_overlap.sql`: Perfetto queries used for traces.

## Results

- `debug_blockwise_result.json`: blockwise smoke, memory, and runtime profile.
- `debug_tensorwise_steady_result.json`: matched debug tensorwise controls.
- `full_blockwise_result.json`: full blockwise profile run.
- `full_blockwise_steady_result.json`: full blockwise ten-control run.
- `full_tensorwise_result.json`: exact final-code tensorwise memory run.
- `full_tensorwise_steady_result.json`: exact final-code tensorwise ten-control run.
- `memory_summary.json`: cross-rank peak and reserved rollup.
- `RESULTS.md`: analysis and decision.

Each `*_memory/` directory contains all eight rank snapshots. Runtime traces:

- `debug_blockwise_runtime.pt.trace.json`:
  `46f563c338fbcf52a85ebdd7d21ab11284667664b6a2b48623dc395d671aac7f`
- `debug_tensorwise_runtime.pt.trace.json`:
  `bd828c4bdb2a9df6e8ff7b1b1c96372cf034cb885fa14fbbf94b06b279924310`
- `full_blockwise_runtime.pt.trace.json`:
  `fc4901aa5ec9c4507e0e95411bfbc3643056baea39d0fcf8e1d512293b026b94`

Key result checksums:

- `debug_blockwise_result.json`:
  `9eab17c2dc056ee2448dae782c9a413b2dfdb7325e42ee424cd885d350457a01`
- `debug_tensorwise_steady_result.json`:
  `888423e8ae8fc80f439446815b81777d68597a7199570019c945e99d6659f360`
- `full_blockwise_result.json`:
  `42bab4b72dd751ffd170aac44b7a2cedbc766eab83b7ef36e95ce95101a11aa6`
- `full_blockwise_steady_result.json`:
  `c345adf4be4b3a5f47c429463e36d8ca01dd535dbc8c14e5b24e5bbf06231ad4`
- `full_tensorwise_result.json`:
  `af9dbec5202c5d43457be08e60d678009dbb650913b8b28a0a0586636123662b`
- `full_tensorwise_steady_result.json`:
  `5118504c2dd1f348facde3926330d6eed7af8fe6262d18d635b600c24f8fc82b`
