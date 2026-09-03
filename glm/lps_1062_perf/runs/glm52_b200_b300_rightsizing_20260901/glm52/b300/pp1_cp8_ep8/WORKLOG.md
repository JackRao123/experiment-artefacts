# GLM-5.2 B300 CP8/EP8/PP1 memory profiles

20260831 12:03 PDT - Started on `tj-qjke943`, one 8xB300 node. Target is the full 78-layer `zai-org/GLM-5.2-FP8` checkpoint with native FP8 routed-expert storage, HybridEP, CP8, EP8, PP1, and full uniform one-layer recompute. Planned memory-profiled steps at 131072 and 262144 tokens; runtime profiling is intentionally disabled.

20260831 12:03 PDT - Base GLM-5.3 cache entry existed but was metadata-only (4 files, 21 MB). Delegated a full FP8 snapshot download into the shared team Hugging Face cache with Xet high-performance mode and 64 concurrent range gets.

20260831 12:34 PDT - Completed both full-model profiles without OOM. At 131K, the driver control mean was 1147 tok/s/GPU and the last two controls stabilized at 1194 tok/s/GPU; hottest-rank peak allocation was 171.06 GiB. At 262K, the control mean was 1186 tok/s/GPU and the last two controls stabilized at 1195 tok/s/GPU; hottest-rank peak allocation was 213.32 GiB. The 262K run retained 54.37 GiB of allocated-memory headroom. Runtime profiling remained disabled.

20260831 12:34 PDT - Copied all 16 memory snapshots, driver JSON, trainer config/log, and the machine-readable memory analysis into this run directory. GLM-5.3 FP8 download completed at snapshot `187fb9fff6319062325ff825627ef6db084d9bc6`: 705 GiB, 141 safetensors shards, no missing or broken files. Stopped the trainer after preserving artifacts.
