# GLM-5.3 B300 sequence-length sweep

## Configuration

- Devbox: `tj-qjke943`, one 8xB300 node with 267.69 GiB usable CUDA memory per GPU.
- Model: full base `zai-org/GLM-5.3` FP8 snapshot `187fb9fff6319062325ff825627ef6db084d9bc6`.
- Trainer maximum: 524288 tokens.
- Parallelism: TP1, PP1, CP8, EP8, ETP1, DP1.
- Training: LoRA r32/alpha32, HybridEP, native FP8 routed-expert storage, BF16 expert compute, full uniform one-layer recompute.
- Protocol: increasing sequence lengths, one shape warmup and three control steps each. No memory or runtime profiler was enabled; peak memory comes from the optimizer response's all-rank CUDA maximum.
- Allocator: `expandable_segments:True,garbage_collection_threshold:0.95`.

## Results

| Exact tokens | Approx. label | Control TPS/GPU | Stable last-two TPS/GPU | Peak allocated | Peak reserved | Allocated headroom | Reserved headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 131072 | 131K | 1179 | 1196 | 171.06 GiB | 174.38 GiB | 96.63 GiB | 93.31 GiB |
| 196608 | 197K | 1197 | 1192 | 192.19 GiB | 196.82 GiB | 75.50 GiB | 70.87 GiB |
| 262144 | 262K | 1189 | 1187 | 213.32 GiB | 218.55 GiB | 54.37 GiB | 49.13 GiB |
| 327680 | 328K | 1081 | 1077 | 234.45 GiB | 240.94 GiB | 33.24 GiB | 26.75 GiB |
| 393216 | 393K | 913 | 879 | 255.58 GiB | 257.54 GiB | 12.11 GiB | 10.15 GiB |

All sweep steps completed with finite loss and gradient norm. There were no OOMs.

## Interpretation

- Peak allocation increases by approximately 21.13 GiB for each additional 65536 tokens. The memory curve remains linear through 393216.
- Throughput is flat around 1190 TPS/GPU through 262144.
- At 327680, stabilized throughput falls to 1077 TPS/GPU, about 9.3% below 262144.
- At 393216, peak allocated reaches 95.5% of device memory and peak reserved reaches 96.2%.
- The first three 393216 controls averaged 913 TPS/GPU and ranged from 849 to 990. A five-control follow-up after continued operation averaged 659 TPS/GPU, with median 647 and range 606-775.
- The 393216 degradation correlates with crossing `garbage_collection_threshold=0.95`: active allocations alone exceed 95% and the allocator has only about 10 GiB reserved headroom. Allocator reclamation/thrashing is the leading explanation, but this sweep did not collect a runtime trace, so it does not prove causality.
- Configuring `max_seq_len=524288` succeeds, but this run did not execute a 524288-token step and does not claim that 512K fits.

## Live trainer

The trainer remains running on `tj-qjke943`, port 8001, with `max_seq_len=524288`.

## Artifacts

- `sweep_result.json`: full five-length sweep windows and aggregate results.
- `393k_repeat_result.json`: five-control follow-up at 393216.
- `sweep_driver.py`: exact sweep protocol.
- `trainer-config.json`, `trainer-server-config.json`, `trainer_srun.log`: configuration and runtime evidence.
