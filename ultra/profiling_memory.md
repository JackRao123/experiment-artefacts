# Nemotron-3-Ultra 550B memory profiling (PP=4, 4 nodes)

Profiling of `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` LoRA SFT on a
**4-node 8xB200** box (`w57o7m3`, 32 GPUs, 183 GiB usable/GPU). Ultra is the
same NemotronH hybrid arch as Super but ~4.6x larger: **108 layers** (48 Mamba,
48 MoE, 12 attention), **512 routed experts / top-22**, 1 shared expert, hidden
8192, Mamba `n_groups=8` (so TP ≤ 8), vocab 131072, ~1.1 TB bf16 weights.

Layout: **TP=8, PP=4, EP=8, ETP=1, CP=1** (world=32, DP=1), bf16 LoRA rank 16,
micro-batch 1, **full** activation recompute (the hybrid layer-recompute
monkeypatch, same as Super Exp 6/8). PP=4 is required: at PP=1 the weights alone
(~4.6x Super's ~30 GiB intercept ≈ 140 GiB/GPU) don't leave room for activations,
so PP shards the 108 layers across 4 stages (27 layers/stage). EP is capped at 8
at PP=4 (each stage has TP·CP·DP = 8 GPUs, so EP·ETP·EDP = 8; EP=32 would need
PP=1/DP=4, which can't hold the weights).

Driven with `sft_driver.py --source synthetic --reset-peak --steps 2 --microbatch-size 1`.
Two memory metrics are reported:
- **`peak_alloc` (cuda:0)** = `torch.cuda.max_memory_allocated` on **rank 0 =
  pipeline stage 0**, read from `/status` (what the driver prints).
- **hottest-GPU `used`** = max `nvidia-smi memory.used` across all 32 GPUs right
  after a step. This is the OOM-relevant number: it includes reserved/cached +
  NCCL/cuBLAS/TE/Mamba workspaces + the **per-GPU MoE-expert imbalance**, and is
  much higher than rank-0 `peak_alloc` (e.g. 83 GiB alloc vs 137 GiB used at
  131K). Stage 0 (rank 0) is NOT the hottest stage.

**PP>1 pads every forward to `max_seq_len`** (see
`megatron_controller._forward_batch_seq_length`: `if pp>1: return max_seq_len`).
So unlike the Super PP=1 sweeps above, each datapoint here is a **separate
trainer relaunch** with a different `max_seq_len` (feeding `--seq-len == max_seq_len`),
not a `--seq-len` sweep within one launch. Confirmed empirically: at
`max_seq_len`=16384, feeding `--seq-len` 4096 still peaks ~39 GiB.

## Experiment U1 — dropless seqlen sweep (no expert-capacity cap)

Default MoE routing (dropless). `peak_alloc` is rank-0/stage-0 allocated.

| max_seq_len | peak_alloc cuda:0 (GiB) | reserved cuda:0 (GiB) | hottest GPU used (GiB) | step1→step2 loss | status |
| --- | --- | --- | --- | --- | --- |
| 16 384  | 38.82 | – | – | 11.917 → 11.895 | ok |
| 32 768  | 44.72 | 47.70 | – | 11.303 → 11.283 | ok |
| 65 536  | 57.63 | 63.44 | – | 0.832 → 0.807 | ok |
| 98 304  | 70.50 | 78.88 | – | 1.078 → 1.036 | ok |
| 131 072 | 83.28 | 95.27 | **~137** | 0.969 → 0.914 | ok ← largest that fits dropless |
| 196 608 | — | — | **182.6 → OOM** | — | **OOM** |
| 262 144 | — | — | — | — | not attempted |

- Stage-0 `peak_alloc` fit (32K–131K): **y ≈ 0.40·(S/1024) + 31.9 GiB** (~0.40
  MiB/tok/GPU) — smooth and linear, ~1.3x Super's 0.30 slope (bigger model).
- **But the dropless run dies at 196 608**, far below where the stage-0 line
  predicts (~109 GiB). One rank shot to **182.6 GiB and OOM'd** inside
  `execute_optim_step → get_grad_norm → NCCL all_reduce` (`Cuda failure 2 'out of
  memory'`). The blow-up is **not** the steady activation curve — it's a
  **per-GPU MoE expert-dispatch imbalance**: with 512 experts / top-22 dropless
  routing, hot experts on some GPUs get disproportionate tokens, so one GPU's
  dispatch buffer balloons non-linearly with context. (Stage-0 cuda:0 stays low;
  the imbalance is across GPUs *within* a stage — 95→137 GiB spread at 131K.)
- **Not** caused by batch size: micro-batch=1 and DP=1, so exactly one sequence
  is in flight; and the rank that OOM'd was not rank 0.

## Experiment U2 — bounded MoE expert capacity fixes it → 256K fits

Add `"moe_expert_capacity": {"capacity_factor": 1.0, "pad_to_capacity": true}`
(the same config the K2.6 512-expert regime uses). This equalizes per-expert
buffers (capacity = ceil(S·top_k/num_experts); overflow tokens on hot experts are
dropped), removing the imbalanced-rank spike. Same TP8/PP4/EP8 layout, full
recompute.

| max_seq_len | peak_alloc cuda:0 (GiB) | hottest GPU used (GiB) | step1→step2 loss | status |
| --- | --- | --- | --- | --- |
| 196 608 | 91.45 | **~121** | 0.898 → 0.880 | ok (was OOM dropless) |
| 262 144 | 110.92 | **~146.6** | 0.918 → 0.899 | **ok — 256K fits** |

- The cap takes **196K from OOM (182.6) → 121 GiB hottest** — *lower* than 131K
  dropless (137 GiB) — and **256K fits at ~146.6 / 183 GiB hottest** (~36 GiB
  headroom). Stage-0 `peak_alloc` slope drops to **~0.30 GiB/1024-tok** (196K→256K),
  vs 0.40 dropless: capping bounds the expert growth.
- **256K trains, loss decreases monotonically** over a 6-step run:
  0.829 → 0.712 → 0.651 → 0.614 → 0.586 → 0.561, ~32.5 s/step, peak 110.92 GiB.

### Verdict
**256K Ultra LoRA SFT works without OOM at TP=8/PP=4/EP=8/CP=1 on 4 nodes with
full recompute + the MoE expert-capacity cap.** The cap is the load-bearing
lever, not the parallelism layout: dropless 512-expert routing OOMs one rank past
~131K regardless of activation recompute; bounding expert capacity is what fits
256K. Trade-off: capacity dropping discards overflow tokens on hot experts (a
quality consideration; the same config is already used for K2.6, and loss
descends cleanly here). This is the committed golden config
(`examples/trainer-configs/nemotron3-ultra-b200-256k-pp4-lora.json`,
`trainer_configs.py` `NEMOTRON_3_ULTRA` B200 S256K).

### Notes / follow-ups
- Driver `peak_alloc` (cuda:0 = stage 0) **under-reports** true peak; the hottest
  GPU (MoE imbalance + workspaces) is the binding constraint. A per-rank max-peak
  readout in `/status` would make sweeps less reliant on side-channel nvidia-smi.
- Synthetic absolute losses are not comparable across seqlens (different random
  content per run); the within-run decrease is the validity check and held
  everywhere.
- To run 256K **dropless** (no token dropping) you'd need more pipeline stages
  (PP=8 / 8 nodes) to dilute the per-GPU expert load, or a correctness-complete
  CP path (CP shards the sequence; still experimental per the Super CP section).
- PP>1 padding to `max_seq_len` is a codebase choice
  (`_forward_batch_seq_length`), not a hard PP requirement; padding to the
  per-batch max (as the PP=1 path does) would save memory/compute on
  shorter-than-max real data, but does not change the worst-case 256K footprint.

## Experiment U1-rerun — dropless sweep with per-GPU (all-32) memory capture

Re-run of U1 (dropless, **no** `moe_expert_capacity` cap) on the same 4×8 B200
box (`w57o7m3`, 32 GPUs, 178.35 GiB usable/GPU), same layout **TP=8 / PP=4 /
EP=8 / ETP=1 / CP=1**, full recompute, LoRA r16, micro-batch 1, 2 steps,
synthetic data. Purpose: record the **hottest GPU `memory.used`** (the
OOM-binding number) and the **memory of all 32 GPUs** at every sequence length —
including 3 new points evenly between 131K and 196K — so the dropless MoE
per-GPU imbalance is visible directly rather than inferred from stage-0.

**Capture method.** `peak_alloc cuda:0` is rank-0/stage-0 `max_memory_allocated`
from `/status` (matches the original U1 column — and it does, to ≤0.01 GiB).
The per-GPU numbers are `nvidia-smi memory.used` sampled at 1 Hz on all 4 nodes
(`srun --overlap`) spanning the whole 2-step run; each GPU's value is its
max-over-the-run, and **hottest = max over all 32**. For the OOM rows the value
is the last reading captured **just before the crash** (best-effort), so it is a
lower bound on the true instantaneous peak.

| max_seq_len | peak_alloc cuda:0 (GiB) | **hottest GPU used (GiB)** | 32-GPU min / median (GiB) | spread max−min (GiB) | status |
| --- | --- | --- | --- | --- | --- |
| 16 384  | 38.84 | 56.73 | 47.5 / 49.2 | 9.2 | ok |
| 32 768  | 44.72 | 68.33 | 53.7 / 57.0 | 14.6 | ok |
| 65 536  | 57.63 | 87.31 | 67.0 / 71.5 | 20.3 | ok |
| 98 304  | 70.50 | 111.00 | 79.9 / 86.7 | 31.1 | ok |
| 131 072 | 83.28 | 133.95 | 93.1 / 101.4 | 40.9 | ok ← largest U1 reported as fitting |
| 147 456 | 89.53 | 144.96 | 99.9 / 108.0 | 45.1 | ok (new point) |
| 163 840 | 95.92 | 156.98 | 106.7 / 116.6 | 50.3 | ok (new point) |
| 180 224 | 102.05 | 167.56 | 112.7 / 123.0 | 54.9 | ok (new point) ← **largest dropless fit** |
| 196 608 | — (OOM) | **178.34** (pre-crash) | 118.3 / 130.2 | 60.0 | **OOM** (`Cuda failure 2 'out of memory'` in optim step) |
| 262 144 | — (OOM) | **174.58** (pre-crash) | 119.7 / 159.0 | 54.9 | **OOM** (`tried to allocate 31.65 GiB`, GPU cap 178.35) |

### All 32 GPUs `memory.used` (GiB), order = node0 g0–g7, node1 g0–g7, node2 g0–g7, node3 g0–g7

```
16 384 : 48.6 48.8 48.7 52.9 48.3 48.6 48.8 50.1 | 49.2 50.8 49.2 48.8 50.1 49.5 48.6 49.7 | 48.5 49.0 48.7 49.6 47.5 48.6 49.1 48.2 | 53.2 53.2 54.1 54.9 56.7 52.9 53.0 54.4
32 768 : 55.8 56.0 54.2 64.2 55.3 55.7 55.8 58.4 | 57.4 60.5 57.2 57.4 59.2 56.9 56.1 58.1 | 55.4 56.3 55.3 56.8 53.7 56.1 56.4 55.2 | 60.5 60.4 61.9 64.3 68.3 60.6 61.1 63.3
65 536 : 71.8 70.6 67.9 87.3 69.2 70.2 70.9 75.8 | 71.2 76.4 76.3 73.9 74.0 70.9 70.0 72.4 | 69.9 67.9 70.8 68.3 67.0 70.3 70.2 69.8 | 73.0 73.6 76.0 75.3 85.1 74.2 78.7 79.5
98 304 : 87.7 85.8 82.4 111.0 83.7 85.6 85.7 93.1 | 86.2 94.1 93.2 90.2 91.9 87.1 86.4 87.0 | 85.1 81.2 88.2 81.9 79.9 84.8 84.1 85.3 | 85.6 88.5 89.9 86.5 101.2 86.8 96.9 95.4
131 072: 104.0 100.4 96.2 134.0 97.7 101.0 107.7 109.8 | 101.0 109.4 107.5 104.2 108.4 100.0 101.2 101.6 | 100.0 94.2 104.7 95.2 93.1 100.0 97.7 101.6 | 98.4 103.5 105.0 97.7 117.2 99.0 115.0 111.1
147 456: 111.0 107.4 102.2 145.0 104.6 107.2 108.0 117.5 | 108.1 117.2 116.7 111.4 116.8 107.4 108.9 108.7 | 107.1 100.6 113.4 101.8 99.9 105.8 104.5 107.4 | 104.4 110.8 112.6 103.0 125.0 105.5 124.4 119.0
163 840: 119.6 114.9 110.1 157.0 111.7 115.9 116.4 126.5 | 116.8 126.3 123.5 119.7 125.9 115.3 118.1 117.7 | 114.4 107.2 121.2 108.5 106.7 113.2 111.2 115.7 | 110.9 118.6 120.5 109.2 133.1 111.8 133.5 127.2
180 224: 126.1 122.2 115.7 167.6 118.8 122.0 122.7 134.1 | 122.9 132.8 130.7 125.3 132.9 121.4 124.5 123.8 | 122.2 113.7 129.9 115.4 112.7 120.0 118.5 123.0 | 116.9 126.2 128.2 116.3 141.2 118.2 143.4 134.9
196 608: 134.1 130.0 122.9 178.3 126.2 129.3 130.4 141.6 | 130.0 140.6 136.2 131.8 140.9 127.5 132.2 130.7 | 129.3 119.8 137.8 122.0 118.3 126.5 124.7 130.1 | 123.4 133.0 135.6 122.9 148.9 124.7 152.1 142.5   (OOM)
262 144: 162.5 133.4 132.2 158.9 119.7 145.3 145.9 171.3 | 159.4 170.6 166.0 158.4 170.8 154.8 163.0 160.6 | 159.4 145.1 169.2 148.2 143.6 153.1 152.5 159.1 | 149.1 162.4 165.0 149.5 167.3 149.7 163.2 174.6   (OOM)
```

### Findings
- **Stage-0 `peak_alloc` matches the original U1 exactly** (38.84/44.72/57.63/
  70.50/83.28 vs U1's 38.82/44.72/57.63/70.50/83.28) — the dropless config is
  faithfully reproduced. So everything new here is the per-GPU view U1 only
  estimated ("~137 hottest at 131K"; measured here 133.95).
- **The hottest GPU is ~1.5–1.7× the stage-0 `peak_alloc`** and is the real
  OOM constraint. The gap is the MoE expert-dispatch imbalance + workspaces +
  `expandable_segments` reserved fragments that stage-0/cuda:0 never sees.
- **The imbalance grows with context**: 32-GPU spread (max−min) goes 9 → 15 →
  20 → 31 → 41 → 45 → 50 → 55 → 60 GiB from 16K to 196K. At 180K the hottest GPU
  (167.6) is **45 GiB above the median** (123.0) — i.e. the box is bottlenecked
  by one hot GPU while most sit ~45 GiB cooler. This is exactly why bounding
  per-expert capacity (U2) fixes the OOM without touching parallelism.
- **The dropless wall is between 180K and 196K.** 180 224 is the largest that
  fits (hottest 167.56, ~11 GiB headroom under 178.35); 196 608 OOMs hitting
  178.3 (matches U1's ~182 at 196K), and 262 144 OOMs needing a further 31.65
  GiB block. The per-stage hot GPU varies (node0-g3 dominates the mid-range; at
  16K node3-g4 was hottest), consistent with imbalance being a routing artifact,
  not a fixed rank.
- **Methodology caveat:** the OOM-row hottest values (178.34, 174.58) are the
  last 1 Hz sample before the crash, so they understate the instantaneous peak
  that tripped the allocator (the 262K error alone asked for +31.65 GiB beyond a
  GPU already at ~147 reserved). Treat them as "≈178 then OOM", not exact peaks.

## Experiment U1-hist — is the per-GPU imbalance data-driven or intrinsic?

The U1-rerun per-GPU memory plot has a "hockey stick": one GPU (stage-0 local
g3 = global rank 3) runs much hotter than the rest, growing with context. Open
question: is that a real MoE imbalance, or an artifact of the **degenerate
synthetic input** (a fixed deterministic ramp), or a routing bug? Test:
instrument the router and compare the per-expert routing histogram across
**different data distributions** at a fixed `max_seq_len=163840`.

**Setup.** Same dropless layout (TP8/PP4/EP8/ETP1/CP1), single boot, 6 forwards:
3 ramp variants (`100+i`, `500+i`, `100+7i`, all mod 30000) and 3 uniform-random
token streams (seeds 0/1/2 over the full 131072 vocab). The router
(`TopKRouter.routing`) was patched (env-gated, reverted after) to dump
`routing_map.sum(0)` per layer per rank. Conservation check passed exactly: each
rank/layer = 450560 = (163840/8 SP-local tokens) × 22 top-k; summing the 8 TP
ranks of a pipeline stage = S×top-k. Experts bin to GPUs as expert→local-GPU
`e//64`, physical GPU `stage*8 + e//64`.

**Per-physical-GPU routing load (max/median over the 32 GPUs):**

| dataset | loss | max/median | routing-hottest GPU |
| --- | --- | --- | --- |
| syn1 ramp (current)   | 1.11 | 1.18 | r28 / r3 (tied) |
| syn2 ramp +offset     | 0.98 | 1.20 | r3 |
| syn3 ramp +stride7    | 1.19 | 1.21 | r3 |
| rand seed0            | 13.72 | 1.30 | r28 |
| rand seed1            | 13.58 | 1.28 | r28 |
| rand seed2            | 13.47 | 1.31 | r28 |

### Findings
- **The routing imbalance is mild — ~1.2–1.3× max/median, not a collapse.** At
  the GPU granularity no GPU's experts get anywhere near the multiples implied by
  the memory "hockey stick".
- **It is intrinsic, not data-driven.** The same small set of GPUs is hottest
  (ranks 3, 28, 30, 31, 9) whether the input is the structured ramp or uniformly
  random tokens, and random is if anything *slightly more* imbalanced (1.30 vs
  1.19), not less. The losses (ramp ≈1.0 vs random ≈13.5) confirm the inputs are
  genuinely different distributions, yet the hot-GPU set and magnitude don't
  move. This is the pretrained router's **learned expert popularity**, not a
  synthetic-data artifact and not a routing bug.
- **Correction to an earlier speculation:** I previously guessed real text would
  spread routing and push the dropless wall higher than the synthetic sweep
  suggested. That is wrong — the imbalance is input-independent, so the
  **~180K–196K dropless wall is representative of real data**, not pessimistic.
- **The memory hockey-stick is not purely MoE routing.** The memory-hottest GPU
  (rank 3, stage 0) is *not* the routing-hottest (rank 28, stage 3). The visual
  spike is mild intrinsic routing imbalance (~1.25×) **plus** stage-specific
  memory (last stage carries the LM-head/loss, stage 0 the embeddings) **plus**
  long-context dropless-buffer amplification **plus** the monotonic
  `expandable_segments` reserved-pool ratchet. So bounding expert capacity (U2)
  helps the routing component, but the last-stage/embedding asymmetry is separate.
- **Measurement caveat:** the per-GPU `memory.used` across these 6 same-process
  forwards is confounded (reserved pool only grows), so the conclusions rest on
  the per-forward routing histograms (conservation-checked), not used-memory
  deltas. Plot: `examples/ptt-nemotron3-super-sft/moe_routing_vs_memory_163840.png`.

## Experiment U3 — expert tensor parallelism (ETP) shrinks the MoE spike → 256K dropless fits

U1-hist showed the hottest GPU's peak is dominated by the transient MoE
expert-compute buffers (`experts.py bias_act_func` ~20 GiB, `grouped_linear.py`
~10 GiB). ETP shards each expert's FFN (weight + that activation) across ETP
ranks; with `EP×ETP=8` fixed per stage, dropping EP also regroups the 512
experts into fewer-but-larger expert-parallel groups, which flattens the
dropless per-GPU imbalance tail. Tested EP4/ETP2 and EP2/ETP4 vs the EP8/ETP1
baseline, same dropless layout (TP8/PP4/CP1), synthetic ramp, `max_seq_len`
163840, full recompute, LoRA r16, micro-batch 1. Hottest GPU = max
`nvidia-smi memory.used` across all 32 GPUs over a 2-step run; warm step = 2nd
step (1st includes per-process autotune).

| config (per stage) | peak_alloc cuda:0 (GiB) | **hottest GPU used (GiB)** | 32-GPU spread (GiB) | warm step (s) | status |
| --- | --- | --- | --- | --- | --- |
| EP8 / ETP1 (baseline) | 95.92 | 156.98 | 50.3 | 21.5 | ok |
| **EP4 / ETP2** | 85.64 | **122.29** | 26.6 | 25.5 | ok (best) |
| EP2 / ETP4 | 115.87 | 128.92 | 16.3 | ~34 | ok (worse) |

### All 32 GPUs `memory.used` (GiB) at 163840, order node0 g0-7 | node1 | node2 | node3
```
EP8/ETP1 : 119.6 114.9 110.1 157.0 111.7 115.9 116.4 126.5 | 116.8 126.3 123.5 119.7 125.9 115.3 118.1 117.7 | 114.4 107.2 121.2 108.5 106.7 113.2 111.2 115.7 | 110.9 118.6 120.5 109.2 133.1 111.8 133.5 127.2
EP4/ETP2 : 101.1 101.2 115.6 115.8  95.7  96.0 104.7 104.9 | 108.1 108.3 106.5 106.8 108.5 108.7 100.2 100.4 | 101.0 107.2 115.4 103.5 112.3 100.5 116.6 109.2 | 110.4 100.9 102.1 103.6 111.2 111.3 107.6 122.3
EP2/ETP4 : 128.8 128.8 128.9 128.9 114.9 115.0 115.0 115.0 | 118.1 118.2 118.3 118.2 113.8 114.1 114.5 114.5 | 114.4 114.4 114.3 114.1 112.7 112.8 112.7 112.6 | 116.7 116.8 116.9 116.9 119.7 119.6 119.4 119.4
```

### Per-category peak composition (snapshot decomposition, hottest rank, 163840)
Reconstructed allocated peak and the tallest single allocation per category (GiB),
from `examples/ptt-nemotron3-super-sft/moe_mem_breakdown.py`:

| | EP8/ETP1 | EP4/ETP2 | EP2/ETP4 |
| --- | --- | --- | --- |
| reconstructed peak | 93.4 | 70.6 | 82.4 |
| tallest expert-act (`experts.py`) | 19.79 | 14.14 | 11.83 |
| tallest expert-GEMM (`grouped_linear.py`) | 9.90 | 7.07 | 9.47 |
| tallest MoE-dispatch (`mappings.py all_to_all`) | 3.96 | 11.31 | **18.93** |

### Findings
- **EP4/ETP2 is the winner: hottest GPU 156.98 → 122.29 GiB (−22%)** at 163840,
  for a small throughput cost (warm step 21.5 → 25.5 s, +19%). The expert-act/GEMM
  spikes shrink (19.79 → 14.14; 9.90 → 7.07).
- **EP2/ETP4 is worse**, not better: the all-to-all **dispatch buffer is NOT
  ETP-sharded**, and halving EP again doubles tokens-per-EP-group, so dispatch
  balloons (3.96 → 18.93 GiB) and more than eats the expert savings — hottest
  rises to 128.92 and peak_alloc to 115.87, with a slower step (~34 s).
- **The win is ~22%, not the naive 50%.** With `EP×ETP=8` fixed the per-GPU
  expert *activation* is ~constant; the real gains are (a) EP4 regrouping experts
  (512/4 vs 512/8) which flattens the imbalance tail (spread 50.3 → 26.6 GiB) and
  (b) partial GEMM-workspace sharding. ETP is a rebalancing+sharding lever, not a
  clean 1/ETP divide.

### 256K dropless now fits (the payoff)
EP8 dropless OOMs at ~196K (U1: hottest ~182 GiB). **EP4/ETP2 fits 262144
dropless** — no OOM, hottest GPU **168.25 GiB** (~10 GiB headroom under 178),
loss 1.026 → 0.937:

| config | max_seq_len | peak_alloc cuda:0 | hottest used | warm step (s) | status |
| --- | --- | --- | --- | --- | --- |
| EP8/ETP1 | 262144 | — | — | — | OOM (U1) |
| **EP4/ETP2** | 262144 | 123.92 | **168.25** | 44.2 | **ok — 256K dropless fits** |

All 32 GPUs `memory.used` (GiB) at 262144, EP4/ETP2:
```
137.1 137.2 160.2 160.4 127.3 127.6 138.4 138.7 | 144.5 144.6 142.8 143.3 140.8 141.0 134.3 134.4 | 133.6 133.9 137.8 138.2 134.0 135.0 137.2 137.6 | 150.9 134.9 138.6 136.2 148.5 148.4 144.5 168.3
```

### Weight-sync verified under ETP>1
`/save_weights_for_sampler` (weight_sync `local`, LoRA r16) completes cleanly
under EP4/ETP2 — `adapter_model.safetensors` (172 MB, 480 tensors incl. Mamba
`mixer.*` and `shared_experts.*` LoRA) + `adapter_config.json`. So the PP-safe
export and LoRA'd shared experts work with ETP-sharded experts; no deadlock.

### Verdict
Committed golden config `NEMOTRON_3_ULTRA` B200 S256K changed to **EP4/ETP2**
(from EP8/ETP1). This is what makes the *dropless* 256K config actually fit (the
prior EP8/ETP1 dropless leaf would OOM past ~196K). EP2/ETP4 is documented as
strictly worse. Snapshots per config are on the laptop under
`ultra_TP8_EP8_PP4/`, `ultra_TP8_EP4_ETP2_PP4/`, `ultra_TP8_EP2_ETP4_PP4/`,
`ultra_TP8_EP4_ETP2_PP4_256k/` (open in memory_viz; remapped to device 0).

## Benchmark harness rerun — PP4 ETP2 vs PP1 EP16/ETP2 (2026-07-01)

Ran the repo benchmark harness logic on 4x8 B200 with:
`--warmup-datums-per-dp=1 --main-datums-per-dp=2 --main-repeats=2`,
LoRA r16, `max_seq_len=262144`, external `nvidia-smi` polling for per-GPU
`memory.used`, and harness memory profiling disabled to avoid allocator-snapshot
overhead. Dataset prep was online; trainer boot/load was forced offline against
the local Ultra BF16 snapshot.

| config | status | warmup FB (s) | main FB mean (s) | main FB TPS/GPU | main optim mean (s) | hottest `memory.used` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TP8/PP4/EP4/ETP2 | ok | 447.77 | 52.47 | 312.25 | 0.022 | 165.5 GiB |
| TP8/PP1/EP16/ETP2 | OOM during warmup FB | — | — | — | — | 177.2 GiB |

PP4 measured main windows:
- window 0: 524288 tokens, FB 53.09 s, 308.63 TPS/GPU
- window 1: 524288 tokens, FB 51.86 s, 315.95 TPS/GPU

PP1/EP16/ETP2 loaded and began the first warmup forward/backward, then OOMed in
the MoE reduce-scatter path:
`torch.OutOfMemoryError: Tried to allocate 7.30 GiB. GPU 0 has ... 5.19 GiB free.
... process has 173.15 GiB memory in use.` No main-window TPS is available for
this config.

All 32 GPUs peak `memory.used` (GiB), order node0 g0-7 | node1 | node2 | node3:
```
PP4/EP4/ETP2 : 144.5 160.8 164.8 149.4 148.8 148.9 154.6 153.4 | 159.9 154.3 157.0 157.5 165.1 152.5 165.2 165.5 | 150.3 150.4 145.1 148.5 151.2 146.2 156.7 154.6 | 129.5 138.8 135.0 135.7 143.5 138.2 141.9 131.5
PP1/EP16/ETP2: 173.2 176.5 158.6 159.0 169.4 170.3 157.6 157.8 | 170.6 171.2 176.7 177.2 167.9 167.6 169.1 169.3 | 172.9 173.2 174.3 174.5 172.5 172.7 166.2 167.2 | 157.3 157.3 160.9 161.3 174.6 175.0 157.8 158.3
```

Takeaway: the benchmark-harness steady-state number for the committed PP4
EP4/ETP2 256K config is ~312 TPS/GPU, matching the earlier inbuilt harness
observation and showing the ad-hoc 583 s first-call timing was warmup/autotune,
not steady state. Removing PP and trying EP16/ETP2 does not make the 256K Ultra
case viable: it OOMs during warmup despite using nearly the full 178 GiB B200
memory budget.
