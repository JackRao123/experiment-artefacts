# Four-group offload layer scaling

Run timestamps: 2026-08-23 UTC

## Objective

Test whether the selected activation-offload policy (`core_attn`, `moe_act`,
`qkv_linear`, and `attn_proj`) preserves its memory and throughput relationship
as the 0D2M proxy expands to four and six cloned MoE layers.

## Provenance and protocol

```text
trainers:        c226338ab13dfa709e4ac7c126ded016b513ea1b
Megatron-Bridge: cad4f085cde65ba5d884c32545c5e7a70fee1b97
Megatron-LM:     d7a72ec1fead20e0c32048836fb3374737cb952e
devbox worktree: /root/.cache/user_artifacts/devboxes/qkpox9w/trainers
```

- One B300 exposed as `cuda:0`; TP1/PP1/CP1/EP1.
- Snapshots: 0D2M, 0D4M, and 0D6M, all at 8,192 tokens.
- The four- and six-layer snapshots clone the same 0D1M MoE layer and give
  every layer a full DSA indexer. Snapshot sizes are approximately 78 GiB and
  114 GiB on disk.
- Each depth used a fresh no-recompute/no-offload process and a fresh
  four-group-offload process, one driver warmup, and three measured windows
  with deterministic inputs.
- MCore retains the final layer's same-name groups for immediate backward, so
  a model with N layers steadily offloads N-1 layers.

The exact 2-layer result predicts 2.821594715 GiB saved per steadily offloaded
layer. Perfect linear scaling therefore predicts 2.821594715, 8.464784145, and
14.107973576 GiB at depths 2, 4, and 6.

## Results

| MoE layers | layers actually offloaded | no-offload median TPS/GPU | four-group median TPS/GPU | TPS change | measured allocation saved | linear prediction | prediction error | saved per offloaded layer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 13,203 | 12,065 | -8.6% | 2.821595 GiB | 2.821595 GiB | 0 bytes | 2.821595 GiB |
| 4 | 3 | 11,722 | 7,212 | -38.5% | 8.464784 GiB | 8.464784 GiB | 0 bytes | 2.821595 GiB |
| 6 | 5 | 10,034 | 6,888 | -31.4% | 14.107974 GiB | 14.107974 GiB | 0 bytes | 2.821595 GiB |

<!--
Allocator peaks, reservations, aggregate TPS, and individual measured windows
are retained here but hidden from the rendered decision table:

| layers | baseline peak allocated | offload peak allocated | baseline peak reserved | offload peak reserved | baseline aggregate TPS | offload aggregate TPS | baseline measured TPS windows | offload measured TPS windows |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2 | 65.154605 GiB | 62.333010 GiB | 68.222656 GiB | 67.345703 GiB | 12,803 | 11,900 | 11,989; 13,303; 13,203 | 11,262; 12,433; 12,065 |
| 4 | 108.517540 GiB | 100.052756 GiB | 111.511719 GiB | 108.089844 GiB | 11,554 | 5,165 | 11,183; 11,722; 11,776 | 7,212; 3,225; 7,569 |
| 6 | 151.963394 GiB | 137.855420 GiB | 154.941406 GiB | 145.687500 GiB | 9,922 | 6,783 | 9,592; 10,034; 10,156 | 6,134; 6,888; 7,459 |
-->

Exact byte counters, unrounded timings, losses, and gradient norms are in
`results.csv` and `raw/`. Full trainer output is in `logs/`.

## Correctness

All six arms completed without OOM or trainer failure. At each depth, offloaded
and non-offloaded runs used matching inputs. Maximum absolute differences were:

| layers | max loss delta | max gradient-norm delta |
|---:|---:|---:|
| 2 | 1.91e-6 | 3.96e-9 |
| 4 | 1.91e-6 | 7.63e-8 |
| 6 | 2.86e-6 | 2.55e-7 |

These are numerical-noise-scale differences, not correctness drift.

## Conclusion

Memory behavior scales perfectly in this controlled proxy. The allocation
delta is byte-exactly linear with the number of layers actually offloaded:

```text
saved allocation = 3,029,664,256 bytes * (MoE layers - 1)
```

The mechanism also remains functionally correct through six layers. There is
no evidence of a memory-scaling or correctness breakdown.

The throughput relationship does break down. The median penalty grows from
8.6% at two layers to 38.5% at four and 31.4% at six. The four-layer offload run
contains one slow 2.54-second window, but its other two windows are still only
7.2-7.6k TPS versus 11.2-11.8k without offload, so the deeper-model slowdown is
not explained by that outlier. Six-layer offload is similarly consistent at
6.1-7.5k TPS versus 9.6-10.2k without offload.

Therefore the four-group selection is validated for linear memory savings and
correctness, but not yet for acceptable full-depth throughput. Extrapolating
the 2-layer 9% penalty to 78 layers would be wrong. The next performance step
should profile the six-layer run's H2D reload schedule and demand waits, then
tune prefetch depth/ordering before testing the production topology.
