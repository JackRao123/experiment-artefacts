# Benchmarking Results — Last Successful Run Per Model

Source: `tests/benchmarking/` GitHub Actions workflow (`benchmarking-tests.yml`).
TPS = main forward/backward tokens per second. TPS/GPU = per-GPU throughput.
Parallelism: TP = tensor, PP = pipeline, EP = expert, CP = context parallel size.

| Model | GPU | #GPU | Seq Len | Parallelism (TP/PP/EP/CP) | TPS | TPS/GPU | Last Run |
|---|---|---:|---:|---|---:|---:|---|
| Kimi-K2.6 | B200 | 16 | 131,072 | TP16/PP1/EP16/CP1 | 6,601.0 | 413.0 | 2026-07-19 |
| Kimi-K2.6 | B200 | 32 | 262,144 | TP8/PP1/EP32/CP1 | 9,906.6 | 309.6 | 2026-07-20 |
| Kimi-K2.7-Code | B200 | 32 | 262,144 | TP8/PP1/EP32/CP1 | 10,229.0 | 319.7 | 2026-07-21 |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | B200 | 8 | 262,144 | TP8/PP1/EP8/CP1 | 17,590.0 | 2,199.0 | 2026-07-19 |
| Qwen3-0.6B | B200 | 1 | 8,192 | TP1/PP1/EP1/CP1 | 25,905.0 | 25,905.0 | 2026-07-19 |
| Qwen3-0.6B | H100 | 1 | 8,192 | TP1/PP1/EP1/CP1 | 21,258.7 | 21,258.7 | 2026-07-15 |
| Qwen3-30B-A3B-Instruct-2507 | B200 | 8 | 131,072 | TP1/PP1/EP8/CP1 | 19,969.0 | 2,496.0 | 2026-07-19 |
| Qwen3-30B-A3B-Instruct-2507 | H200 | 8 | 131,072 | TP1/PP1/EP8/CP1 | 8,005.0 | 1,001.0 | 2026-06-14 |
| Qwen3-4B-Instruct-2507 | B200 | 1 | 40,960 | TP1/PP1/EP1/CP1 | 10,446.0 | 10,446.0 | 2026-07-19 |
| Qwen3-4B-Instruct-2507 | H200 | 1 | 40,960 | TP1/PP1/EP1/CP1 | 6,746.0 | 6,746.0 | 2026-06-10 |
| Qwen3-8B | B200 | 1 | 40,960 | TP1/PP1/EP1/CP1 | 8,867.0 | 8,867.0 | 2026-07-19 |
| Qwen3.5-0.8B | B200 | 1 | 131,072 | TP1/PP1/EP1/CP1 | 30,880.0 | 30,880.0 | 2026-07-19 |
| Qwen3.5-0.8B | H200 | 2 | 131,072 | TP1/PP1/EP1/CP1 | 35,001.0 | 17,501.0 | 2026-06-14 |
| Qwen3.5-122B-A10B | B200 | 8 | 131,072 | TP2/PP1/EP8/CP1 | 21,172.0 | 2,646.0 | 2026-07-19 |
| Qwen3.5-122B-A10B | H100 | 8 | 131,072 | TP8/PP1/EP8/CP1 | 8,961.0 | 1,120.0 | 2026-08-04 |
| Qwen3.5-122B-A10B | H200 | 8 | 131,072 | TP4/PP1/EP8/CP1 | 10,068.0 | 1,259.0 | 2026-06-14 |
| Qwen3.5-27B | B200 | 2 | 131,072 | TP2/PP1/EP1/CP1 | 5,682.0 | 2,841.0 | 2026-07-19 |
| Qwen3.5-27B | H200 | 4 | 131,072 | TP2/PP1/EP1/CP1 | 5,941.0 | 1,485.0 | 2026-06-14 |
| Qwen3.5-2B | B200 | 1 | 131,072 | TP1/PP1/EP1/CP1 | 26,139.0 | 26,139.0 | 2026-07-19 |
| Qwen3.5-2B | H200 | 2 | 131,072 | TP1/PP1/EP1/CP1 | 27,855.0 | 13,928.0 | 2026-06-14 |
| Qwen3.5-35B-A3B | B200 | 8 | 131,072 | TP1/PP1/EP8/CP1 | 42,260.0 | 5,283.0 | 2026-07-19 |
| Qwen3.5-35B-A3B | H200 | 4 | 131,072 | TP1/PP1/EP8/CP1 | 10,970.0 | 2,743.0 | 2026-06-14 |
| Qwen3.5-397B-A17B | B200 | 16 | 262,144 | TP8/PP1/EP16/CP1 | 10,845.9 | 677.9 | 2026-07-29 |
| Qwen3.5-4B | B200 | 1 | 131,072 | TP1/PP1/EP1/CP1 | 11,591.0 | 11,591.0 | 2026-07-19 |
| Qwen3.5-4B | H200 | 4 | 131,072 | TP1/PP1/EP1/CP1 | 21,665.0 | 5,416.0 | 2026-06-14 |
| Qwen3.5-9B | B200 | 1 | 131,072 | TP1/PP1/EP1/CP1 | 9,661.0 | 9,661.0 | 2026-07-19 |
| Qwen3.5-9B | H200 | 4 | 131,072 | TP1/PP1/EP1/CP1 | 16,927.0 | 4,232.0 | 2026-06-14 |
| Qwen3.6-27B | B200 | 2 | 131,072 | TP2/PP1/EP1/CP1 | 5,676.0 | 2,838.0 | 2026-07-19 |
| Qwen3.6-27B | H200 | 4 | 131,072 | TP2/PP1/EP1/CP1 | 5,964.0 | 1,491.0 | 2026-06-14 |
| Qwen3.6-35B-A3B | B200 | 8 | 262,144 | TP2/PP1/EP8/CP1 | 30,737.2 | 3,842.2 | 2026-07-30 |
| Qwen3.6-35B-A3B | H200 | 4 | 131,072 | TP1/PP1/EP8/CP1 | 11,234.0 | 2,808.0 | 2026-06-14 |

## Notes

- Same model name with different GPU type/count = separate configs (e.g. Kimi-K2.6 on 16 vs 32 B200 GPUs).
- Parallelism config is from the golden trainer configs (`loops_models/model_configs/trainer_configs.py`) for report-only rows, and from the actual benchmark `junit.xml` perf data for artifact rows.
- B200 data from 2026-07-19 (scheduled run) carried throughput-regression warnings vs baseline, but the benchmarks themselves completed successfully.
- H200 data is from 2026-06-14 (last successful scheduled H200 run); H100 data is sparse (only Qwen3-0.6B on 2026-07-15 and Qwen3.5-122B-A10B on 2026-08-04).
- Models that have never had a successful run: Qwen3.5-397B-A17B (on B200 8-node), GLM-5.2, NVIDIA-Nemotron-3-Ultra-550B-A55B, Kimi-K2.7-Code (on B200 16 GPUs).
