# GLM-5.3 B300 CP8/EP8/PP1 memory profiles

20260831 13:27 PDT - Started on `tj-qjke943`, one 8xB300 node. Target is the complete 141-shard base `zai-org/GLM-5.3` FP8 checkpoint with native FP8 routed-expert storage, HybridEP, CP8, EP8, PP1, and full uniform one-layer recompute. Planned memory-profiled steps at 131072 and 262144 tokens; runtime profiling is intentionally disabled. The trainer must remain running after profiling.

20260831 13:45 PDT - Completed both full-model profiles without OOM. At 131K, the driver control mean was 1175 tok/s/GPU and the last two controls stabilized at 1200 tok/s/GPU; hottest-rank peak allocation was 171.06 GiB. At 262K, the control mean was 1176 tok/s/GPU and the last two controls stabilized at 1187 tok/s/GPU; hottest-rank peak allocation was 213.32 GiB. The 262K run retained 54.37 GiB of allocated-memory headroom. Runtime profiling remained disabled.

20260831 13:45 PDT - Copied all 16 memory snapshots, driver JSON, trainer config/log, and machine-readable memory analysis into this run directory. Verified `/health` and `/status` after profiling. Left the GLM-5.3 trainer running at step 10 as requested.
