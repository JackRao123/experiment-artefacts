# PP_REBALANCE_ARM — layer-rebalance scoping (kepler, 2026-08-13)

**Task source:** fermi assignment 3 (Mac-side scoping). **Question:** does
moving 1–2 PP layers from stage 1 to stage 0 attack the ~7.4 s/step boundary
skew measured in the stage-asymmetry check (IDLE_RESIDUAL_MAP.md §7)?
**Verdict: KILLED — the current 38/40 split is already the compute-optimal
LEGAL split; the rebalance quantum is 4 layers, and the next legal cell is
measurably worse. No box window spent.** Reopen condition in §5.

## 1. Q1 — where 38/40 comes from

Not an mcore default and not a mission-config knob: the mission config
(`pp2cp8ep8/configs/trainer_pp2cp8ep8_131k.json`) sets only
`pipeline_parallel_size: 2`. The split is a **trainers-side explicit layout
override**: `server/src/trainers_server/dp_worker/backends/megatron_bridge/glm52_dsa.py`,
`_GLM52_DSA_PIPELINE_LAYOUTS[(78, 2, 1)] = [38+embedding, 40+loss]`, installed
via `provider.pipeline_model_parallel_layout`.

**GLM-5.2 = 78 hidden layers** (HF config `num_hidden_layers=78`, confirmed
also by fermi via lebesgue's VPP work; my trace call-counts — 38×16=608 CF on
rank0, 40×16=640 on rank8 — match). HF also carries `num_nextn_predict_layers=1`
(MTP), but the layout table carries no `mtp` entry — MTP is not trained in
this LoRA config.

**Why not 39/39:** GLM-5.2's DSA shares top-k indices across layer groups
(`index_topk_freq=4`); a PP stage must START on a top-k-computing layer or it
would need indices produced by the other stage (PP passes only hidden states).
Per the in-code constraint (`(start−3) % 4 == 0`; the VPP2 comment documents
chunk starts 19/39/59), legal stage-1 starts are 1-indexed layers ≡ 3 (mod 4):
…, 35, **39 (current)**, 43, … → legal PP2 splits are **{34/44, 38/40, 42/36}**.
The even 39/39 is illegal, and **the rebalance quantum is 4 layers** — "move
1–2 layers" is not expressible under the constraint.

## 2. Q2 — per-stage compute and the optimality math

Measured (non-NCCL compute wall per step, union of non-NCCL kernel/memcpy/
memset intervals; l3 traces are OLD-wheel — class structure valid, absolute
times suspect; fe127 rank0 is the fixed-wheel number-of-record trace):

| stage | trace | layers + extras | compute wall/step | per-mb |
|---|---|---|---:|---:|
| 0 | fe127 r0 (new wheel) | 38 + emb | 70.52 s | 4.41 s |
| 0 | l3 r0 (old wheel) | 38 + emb | 68.40 s | 4.28 s |
| 1 | l3 r8 (old wheel) | 40 + head + loss | 74.05 s | 4.63 s |

Per-layer cost L ≈ 68.40/38 = **1.80 s/layer/step** (fwd+bwd, full recompute).
Stage-1 overhead H (chunked LM head + CE loss, vocab 154,880, untied) ≈
74.05 − 40×1.80 ≈ **2.0 s/step** (consistent with the FLOP bound: the head
GEMM ≈ 31 TFLOP/mb fwd at 16,384 tokens/rank → ~1 s/step fwd+bwd; measured CE
kernels 0.05 s; H is wheel-insensitive — GEMM/CE, not the cudnn-frontend path).
Cross-check on l3 rank8: no large-GEMM head class exists (largest nvjet 8.3 ms;
the campaign's chunked_lm_head chops the logits GEMM below 3 ms granularity),
so the residual estimate H ≈ 2.0 s is the bound — 42/36 needs H > 2L = 3.6 s
to even tie; margin ~1.6 s.

Max-stage time over the legal cells:

| split | stage 0 | stage 1 | max |
|---|---:|---:|---:|
| 34/44 | 61.2 | 81.2 | 81.2 |
| **38/40 (current)** | 68.4 | 74.05 | **74.05** |
| 42/36 | 75.6 | 66.85 | 75.6 |

The continuous optimum is moving ~1.6 layers to stage 0 (equalizes at
T1−T0 = 5.65 = 2xL → x = 1.57) — **below the legal quantum**. The next legal
cell (42/36) overshoots: max-stage **+1.55 s WORSE**. 38/40 is the
compute-optimal legal split; the designer already picked the best cell.

**Boundary-skew accounting:** the ~7.4 s skew (5.0–5.2 s stage-0 start wait +
2.4–2.7 s drain wait) decomposes into (a) stage-1 compute excess ~5.65 s —
already at its legal minimum, not attackable by layer movement — and (b)
stage-1's seam/optimizer tail — attackable, and it routes to **lever R2 (seam
python)** from the residual map, not to this arm.

**block+K21 (W1b) interaction (fermi's input, pre-registered against both
baselines):** partial recompute shrinks L (kept layers skip recompute; bwd
recompute ≈ 65 ms of the 134.5 ms per-layer bwd on the fixed wheel). At
−25% L (≈1.35 s) the 42/36 margin narrows from 2L−H = 1.6 s to ≈0.7 s but
**does not flip**; it would take L ≤ ~1.0 s (−45%) for 42/36 to tie. Conclusion
is unchanged under the W1b baseline, with the margin noted as narrower.

## 3. Q3 — memory (mission-config answer; W1b section HELD OPEN per fermi)

Per moved layer landing on stage 0 (EP8-sharded FP8 params: 32 experts/rank ×
3 × 6144 × 2048 ≈ 1.21 GB + attention ≈ 0.1 GB; activations at full recompute:
~201 MB/layer/mb at 16,384 tok/rank × ~2 in-flight mb sets ≈ 0.4–0.5 GiB/layer):

- **42/36 moves 4 layers: ≈ +5.2 GB params + ~2 GB activations ≈ +7 GB on
  stage 0.** Mission-config peak 162 GiB vs 267.7 GiB torch capacity → fits
  with ~98 GiB headroom. Memory is NOT the blocker under the mission config.
- **W1b (block+K21) baseline: HELD OPEN.** fermi's instruction: use W1b's
  measured peaks (predicted 221/240 GiB), not estimates. With ~28 GiB of
  predicted headroom at the tight rank, +5.2 GB params plus K21-kept
  activations scaling with stage-0 layer count (38→42 = +10.5%) needs the
  measured numbers. Moot unless §5's reopen condition fires.

## 4. Q4 — disposition

**Arm KILLED pre-box** (saves the window). Reasons, in order:
1. Legality: rebalance quantum is 4 layers (top-k sharing), not 1–2.
2. Compute: 38/40 is the optimal legal cell; 42/36 is +1.55 s max-stage worse
   (mission config), margin narrowing but not flipping under block+K21.
3. Memory would have fit under the mission config; moot.

## 5. Reopen condition (pre-registered, zero box cost)

The W1b window already includes my fixed-wheel rank0+rank8 trace captures
(`BT_PROFILE_RANKS=0,8`, mechanism-verification lane). Pre-registered reading
on those traces: **reopen the arm iff H > 2L′ on the fixed wheel post-K21**
(equivalently: stage-1 overhead alone exceeds two layer-times), i.e. the
42/36 cell's max-stage would beat 38/40's. Current reading: 2.0 vs 3.6 — arm
stays dead. If it ever fires, the memory section §3-W1b must be completed
from measured W1b peaks before any boot.
