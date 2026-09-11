# GLM-5.3 262K on two-node GB300 NVL72

## Configuration

- Model: `zai-org/GLM-5.3`
- Model revision: `187fb9fff6319062325ff825627ef6db084d9bc6`
- Trainers revision: `e8eef2d4f5a4b53ff89b9a98867b920104ce41db`
- Hardware: 8 GB300 GPUs, two physical nodes with four GPUs each
- Parallelism: TP1, PP1, CP8, EP8, ETP1, DP1
- Sequence: one 262,144-token synthetic datum per step
- LoRA: rank 32, alpha 32
- Dispatcher: HybridEP
- Attention: FlashAttention
- Expert storage: native FP8
- Recompute: full, uniform, one layer
- Fabric: Grove ComputeDomain with MNNVL enabled

The model and parallel configuration match the B300 S256K golden row. The
physical placement differs: the golden row assumes one eight-GPU B300 node,
while this run uses two four-GPU GB300 nodes in one NVL72 ComputeDomain.

## Throughput

The supplied `tools/profile_driver.py` ran one warmup followed by three
unprofiled control windows.

| Window | Forward/backward | TPS/GPU | Optimizer | Loss |
|---|---:|---:|---:|---:|
| Warmup | 181.38s | 180.66 | 1.28s | 12.331361 |
| Control 0 | 23.42s | 1399.05 | 0.21s | 12.326888 |
| Control 1 | 23.26s | 1408.61 | 0.21s | 12.324952 |
| Control 2 | 23.19s | 1412.96 | 0.20s | 12.320674 |

Headline control mean:

- **1,406.85 tokens/s/GPU**
- **11,254.78 aggregate tokens/s** across eight GPUs
- **23.2918s** forward/backward
- **0.2052s** optimizer step
- Peak allocated: **206.10 GB** (191.95 GiB) on rank zero
- Peak reserved: **209.04 GB** (194.68 GiB) on rank zero

The driver reports 16.34% MFU and 23.47% analytic HFU using its default 2.25
PFLOP/s B300 peak. Using the tool's documented 2.5 PFLOP/s GB300 peak gives
approximately 14.71% MFU and 21.13% HFU. TPS is unaffected by that peak-FLOPS
normalization.

The preserved B300 reference says 26.3-26.7s for the clean control. This GB300
NVL72 run reduces forward/backward wall time by 11.4-12.8%, but it is not an
apples-to-apples hardware comparison because the physical topology and trainers
revision differ.

## Artifacts

- `benchmark.json`: raw driver output and per-window measurements
- `trainer-config.json`: exact runtime configuration
- `leader-trainer.log`: ranks 0-3 startup and execution log
- `worker-trainer.log`: ranks 4-7 startup and execution log

No memory or runtime profiler was enabled; the throughput headline is an
unprofiled control measurement.
