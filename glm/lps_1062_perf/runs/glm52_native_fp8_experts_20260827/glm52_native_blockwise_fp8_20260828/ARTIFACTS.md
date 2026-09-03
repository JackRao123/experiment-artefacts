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
- `phase1a_benchmark.py`: isolated persistent-BF16 and naive temporary-
  dequantization benchmark over one real routed expert.
- `phase1a_results.json`: complete Phase 1A correctness, CUDA-event timing,
  useful TFLOP/s/MFU, and allocator measurements.
- `PHASE1A.md`: Phase 1A method, summary tables, and conclusion.
- `phase1a_optimize_dequant.py`: matched hot-expert comparison of expanded
  scales, compact-scale broadcasting, and compiled broadcasting.
- `dequant_optimization_m4096.json`: raw iteration 0-2 measurements.
- `phase1a_compiled_cold_sweep.py`: paired benchmark cycling eight real experts.
- `dequant_compiled_cold_m4096_paired.json`: focused paired stability run.
- `dequant_compiled_cold_sweep.json`: final six-point paired measurements.
- `PHASE1A_DEQUANT_OPTIMIZATION.md`: iteration report and final conclusion.
- `PHASE2_TE_GENERIC.md`: final Phase 2 design, pushed SHAs, and validation.
- `phase2_te_generic_final_pushed_steady5.json`: exact pushed six-step 0d1m run.
- `phase2_te_generic_final_pushed_trainer.log`: exact pushed trainer startup/run log.
- `phase2_boundary_comparison.json`: all-rank sampled FC1/FC2/MoE/block/logit parity.
- `PHASE4_FULL_MODEL.md`: full-model throughput, memory, and acceptance report.
- `phase4_full_native_te_generic.json`: exact full-model runtime configuration.
- `phase4_full_native_te_generic_smoke.json`: initial full-model fit smoke.
- `phase4_full_native_te_generic_steady5.json`: five warmed throughput controls.
- `phase4_full_native_te_generic_memory_result.json`: memory-profile window result.
- `phase4_full_native_te_generic_memory.rank0.pickle`: rank-0 allocator snapshot.
- `phase4_full_native_te_generic_trainer.log`: full-model trainer log.
- `phase4_full_native_te_generic_runtime_result.json`: runtime-profile driver result.
- `phase4_full_native_te_generic_runtime.pt.trace.json`: warmed rank-0 Kineto trace.
- `phase4_kernel_categories.sql`: broad GPU kernel-class rollup.
- `phase4_gemm_breakdown.sql`: NVJet versus FP32 output-head GEMMs.
- `phase4_nccl_breakdown.sql`: collective-class rollup.
- `phase4_operator_breakdown.sql`: CPU/operator rollup.
- `phase4_per_layer.sql`: checkpointed forward/backward layer timing.

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
- `phase1a_benchmark.py`:
  `dec1e216f178691a3ab855b821aef2d22fae592bdf2e17bbd3eb49d585a1436d`
- `phase1a_results.json`:
  `c1eef882bcd643c0d37231b3ae75e3102e52344a715990b1c8dbf742140e79c8`
- `PHASE1A.md`:
  `b0bdbb7181bf47acf5e8b1e354382ae7241144f11d95e38bd643630ad2a1eeb2`
- `phase1a_optimize_dequant.py`:
  `78be3a890a6995d5f60ce4e391f8c2b6fc9d9df2f7bfe03947dfb8f32a050d30`
- `dequant_optimization_m4096.json`:
  `d5f2af193b40e2d51fb105dac01783298f6f76f96c5fbbf09e9c9f16c46e15e3`
- `phase1a_compiled_cold_sweep.py`:
  `7c955d51502f6417f8e7bf866cc26159ceab7959ce54aed6cfe5ccf965d2073f`
- `dequant_compiled_cold_m4096_paired.json`:
  `7aacc26faddf289b81909dab0491171bf5f60491792fe51e03267378ca6716d6`
- `dequant_compiled_cold_sweep.json`:
  `6fa02bc08e8160644887bf9171eaea30cb15a6d2740554533a1c80a4cd5099d7`
- `PHASE1A_DEQUANT_OPTIMIZATION.md`:
  `8d13f57d5fcb7bf89e4739d00e93dad929287b963742e33b5c14d4468a310e68`
- `PHASE2_TE_GENERIC.md`:
  `c2bc1fb406063784b6d8f6d548c729f758be80e6ee04676ed456bea3abbc97b6`
- `phase2_te_generic_final_pushed_steady5.json`:
  `14950f024eeb5c2b4beecdf5b62de83a3ece78a8665477632ee66fccb4910601`
- `phase2_te_generic_final_pushed_trainer.log`:
  `3809230c701c1cf635a85e18e530ee22c0f41c1f8d6f2f13b7b9d5a3062bac20`
- `phase2_boundary_comparison.json`:
  `56b56d5e425485e115a947451c305029162e3993e05ca8075d839d1282e5e444`
- `PHASE4_FULL_MODEL.md`:
  `d0f29e31f4940913b72978958185099a9bc5eed0dd6d6a3d913df1475e593e47`
- `phase4_full_native_te_generic.json`:
  `9b2c8fe1afd6a29f5bae683d54896f37498b076b9d103ba2b6597627bfa9700a`
- `phase4_full_native_te_generic_smoke.json`:
  `78cae0165ed55d318db1bf2aa5f9657825e4262265b281f66b65974df3221086`
- `phase4_full_native_te_generic_steady5.json`:
  `3010ff8b9bd1e4b8f75404ea01c5bf19241dc28ff4d8ed41ce96b25a86f7d132`
- `phase4_full_native_te_generic_memory_result.json`:
  `d2455f8674a7fe620ac2d29a9f5382fb0d8d727b1a88553e5e919e3637096d89`
- `phase4_full_native_te_generic_memory.rank0.pickle`:
  `527b17e795e37c46995d06428f5cba9f2a99f517f97fe3c476c51c2f5068d043`
- `phase4_full_native_te_generic_trainer.log`:
  `c87f352690a790642765ae94f8cb0f25179c18a67f5d8cb5de6b81e10633a34a`
- `phase4_full_native_te_generic_runtime_result.json`:
  `b32f29f128ced073c3d1f7781ca753638d84067ac271ddf65d99c16bab73563b`
- `phase4_full_native_te_generic_runtime.pt.trace.json`:
  `2d3d37947b2b17246905d04e03b3f78c55041209204b2d5323c10377ebf7073b`
- `phase4_kernel_categories.sql`:
  `80d0ed0be344f516a7ce3d83967c3104bfbe55bf30919a2450f330fad3b67b24`
- `phase4_gemm_breakdown.sql`:
  `80e45701950007e80fd28bb018960f752eef2f4b27d511b52ab553232f7e599d`
- `phase4_nccl_breakdown.sql`:
  `cf38ba7c0be31d7d3fb0e288073269d77adf5c24a5d28cfd82ae655f168579b7`
- `phase4_operator_breakdown.sql`:
  `3b7d14330ca9e3ddb1fbab75b14d4b9c56027e0aee97c17ba5f1922f93657562`
- `phase4_per_layer.sql`:
  `c375b9fb3d9279b34b8b0d53a871462318650dddf260d4ed5c7d641f3d017e49`
