RUNNING: CP8 startup. Full GLM-5.3, BF16 expert storage, MFSDP parameter sharding, TP1/PP1/EP1/ETP1. Start CP8 on eight GPUs; if validated, relax CP. Sequence length remains 131072 per datum. Use one datum per data-parallel replica: CP8 D1, CP4 D2, CP2 D4, so no rank benchmarks an empty DP shard. Separate latency and tokens/GPU. Three warmups, five controls, memory and all-rank runtime profiles. Source and validation must be recorded before interpreting results.

Standard TE grouped GEMMs are retained. The experimental zero-copy grouped-MM
alternative passed a debug loss/norm check but did not show a robust end-to-end
gain. Normal LoRA initialization is used here, not the special parity values.

Source: trainer c9a723bf431621ca05580726f4acd01ec618326a with Core changes
equivalent to 27f9fdc9d57c20c84ac298279242dae6c54a99e4, plus the run-local
MFSDP enablement shim. Exact patches and source fingerprints will accompany
completed results. The original profiler driver and mfu.py are unchanged.
