# GLM-5.3 B300 sequence-length sweep

20260831 13:54 PDT - Preparing a full-model GLM-5.3 sweep on `tj-qjke943` at exact token counts 131072, 196608, 262144, 327680, and 393216. The trainer is configured with `max_seq_len=524288`, TP1/PP1/CP8/EP8/ETP1, HybridEP, native FP8 routed-expert storage, LoRA r32, and full uniform one-layer recompute. Each shape gets one warmup and three control steps. The trainer must remain running afterward.

20260831 14:24 PDT - Completed all five lengths without OOM. Stable TPS/GPU was 1196, 1192, 1187, 1077, and 879. Peak allocated memory was 171.06, 192.19, 213.32, 234.45, and 255.58 GiB. Peak reserved memory was 174.38, 196.82, 218.55, 240.94, and 257.54 GiB.

20260831 14:24 PDT - The initial 393216 controls were noisy, so ran five additional controls. Follow-up TPS/GPU averaged 659, median 647, range 606-775. The shape uses 95.5% allocated and 96.2% reserved memory, crossing the configured allocator GC threshold of 0.95; allocator pressure is the leading explanation for the instability, though no runtime trace was collected. Preserved JSON and trainer log locally. Left the 524288-configured trainer running as requested.
