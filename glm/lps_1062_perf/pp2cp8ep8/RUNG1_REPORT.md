# Rung 1 — the 131k memory census: results

conway, 2026-08-20. Box **qed7z1w** (2 nodes x 8 B300, ali). Booted commit
**7eec3054** on `jackrao/lps-1062-actplace`. Config
`trainer_pp2cp8ep8_131k.json` (full recompute, no offload), d2, 12 steps,
run twice.

Framing rules that apply to every number here: the whole ladder runs one tree
with no TF32 — that is the environment, not a caveat. Absolute tokens/sec
figures are context only, on our own tree, and are **not** comparable to any
historical band. Selective recompute and the offload are ONE package.

---

## 0. The headline: gate #1 cannot be answered by a full-recompute boot

**The census ran cleanly and measured the wrong thing, for a reason that is
structural rather than operational.**

Gate #1 exists to replace the plan's inferred glue row — the norms, residuals
and router bookkeeping that the plan keeps resident on the GPU, estimated by
subtraction at 0.64-1.14 GiB per MoE layer per microbatch. The plan's
instruction was to measure it on a baseline boot.

But under **full recompute** those tensors are not resident at all. That is
what full recompute means: each layer's eager set is produced, freed, and
regenerated during backward. At any instant only the layer currently being
recomputed holds an eager set. So a full-recompute boot can only measure
today's residency — the checkpoint-stored layer inputs — never the quantity
the plan wants.

The measured numbers say exactly this, unambiguously:

| bucket | measured, GiB per MoE set (rank 0) | the plan's eager figure |
|---|---|---|
| `moe_act` | **0.042** | 0.75-1.15 |
| `core_attn` internals | **0.040** | 0.70 |
| `attn_proj` | **0.000** | 0.20 |
| "glue" | **0.250** | 0.64-1.14 |

`moe_act` comes out ~20x under the plan's figure and `core_attn` ~18x under,
which is not a measurement error — it is the recompute working. And the
"glue" column is not glue: at 0.250 x 70 = 17.5 GiB it is essentially the
checkpoint-stored layer inputs (S_ckpt = 0.19 GiB x 76 sets = 14.4 GiB) plus
the single layer in flight, i.e. precisely the quantity the plan says is
ALREADY inside the base peak.

**Consequence.** The glue row cannot be closed from this boot, and it cannot
be closed from any full-recompute boot. It can only be measured on a
configuration where those tensors are actually resident — a selective-recompute
run. Rung 2a would have been one at 32k; it is deferred. **Arm 3b is one at
131k**, and its peak yields the glue row directly:
`glue = (peak_3b - base - prefetch) / sets`.

So gate #1's question is answered by the first arm rather than before it. The
projection that decides whether that arm is safe to boot is in §5, and it is
built from the plan's inherited band, which is honest about being 78% wide.

## 1. The second finding: the last stage's loss head is 5x the plan's estimate

The plan sizes the last stage's loss and output logits at **~5.1 GiB** in
bf16, or ~10.1 GiB if the loss upcasts to fp32. Measured at the peak instant
on rank 8:

**`loss_head` = 26.04 GiB.**

That is 5.1x the bf16 estimate and 2.6x the fp32 upper bound, and it is the
single largest bucket anywhere in the census — larger on rank 8 than every
activation bucket on rank 0 combined. It is also stable to two decimal places
across all eight ranks of the last stage (26.00-26.04 GiB), so it is not a
sampling artifact.

This explains the stage asymmetry the campaign kept seeing and the runbook
flagged as a possible headline: the last stage peaks **higher** than the
first despite holding 40 activation sets to the first stage's 70.

It matters for the plan beyond bookkeeping. The plan's fit arithmetic treats
rank 0 as the binding stage. That is still true at 131k — rank 0 binds in
every row of the projection in §5 — but the margin on rank 8 is 21 GiB
smaller than the plan assumed, and any future change that adds resident bytes
to the last stage (a larger vocabulary, an fp32 loss upcast, more in-flight
microbatches) eats a budget nobody has been costing correctly.

## 2. Stage peaks — all three metrics, never mixed

| metric | first stage (ranks 0-7) | last stage (ranks 8-15) | difference |
|---|---|---|---|
| nvidia-smi reserved, plateau | **152.5 GiB** | **162.6 GiB** | last stage +10.1 |
| torch peak (driver `/status`) | 141 GiB | 141 GiB | — |
| live at peak instant (allocator replay, above the persistent baseline) | 38.3 GiB | 41.3 GiB | last stage +3.0 |
| per-GPU spread within the last window | 6.7 GiB | 0.8 GiB | — |

nvidia-smi reserved is the OOM-relevant metric and the one the projections
use. It is never mixed with the torch figure; they differ by NCCL and driver
overhead.

**Plateau declared.** The last two control windows agree exactly on both
nodes (152.5/152.5 and 162.6/162.6), and the whole-run max equals the last
window max, so nothing earlier spiked above it.

The first stage's 6.7 GiB per-GPU spread against the last stage's 0.8 GiB is
worth a note: the first stage holds the dense layers and the embedding, and
its expert routing is jittered per rank, so its ranks are genuinely
heterogeneous. The projection uses the max, not the mean.

## 3. The d2 baseline

| run | mean tok/s/GPU (controls) | min-max | within-run spread | step | torch peak |
|---|---|---|---|---|---|
| **A (clean)** | **645.1** | 641-650 | **1.4%** | 25.4 s | 141 GiB |
| B | 629.7 | 607-644 | 5.9% | 26.0 s | 141 GiB |

**Run A is the baseline of record.** Run B is 2.4% slower with four times the
spread, and the cause is mine: I started the census analysis (16 allocator
snapshots, ~175 MB each) on the leader node while run B was stepping, so its
host-side work competed with the trainer's. Run B's memory read is unaffected
and agrees with A's exactly; its throughput is not a clean sample and is not
used as one.

The **throughput noise floor** for judging the arms is therefore run A's
within-run spread, **1.4%**, with 2.4% as a loose upper bound from the
contaminated pair. Against an expected arm effect of tens of percent, either
figure is comfortably small.

Kineto's cost on the traced step measured within noise of zero (-1.7% and
-3.1%, i.e. the traced step came out marginally *faster* than the control
mean both times), so tracing both stage leaders is effectively free here.

## 4. The numerical noise floor (forward-only, fresh weights)

Two identical forward-only legs on the same boot, same weights, same fixed
9-datum set (262,032 real tokens). This is a floor MEASUREMENT, not a gate;
the driver's built-in 1e-6/1e-3 tolerances are absolute bars from a
deterministic-path era and are not judged against.

| statistic | value |
|---|---|
| loss | 12.303311598498 vs 12.303669235329 |
| loss relative difference | **2.907e-05** |
| per-token abs diff — median | **0.0592** |
| p95 / p99 / p99.9 | 0.235 / 0.411 / 0.871 |
| max | **4.338** (datum 1, position 17097) |
| share differing > 1e-3 | **95.67%** |
| share differing > 1e-2 | 87.89% |
| share differing > 1.0 | 0.065% |
| bitwise identical | 3.43% of tokens |

Plausibility check only, per the runbook: the campaign's per-token floor was
3.7-5.5 and our max lands at 4.338, inside it. So this is the known behaviour
of this path, not a new finding.

What it means for the ladder is worth stating plainly: **at this noise level
the per-token logprob vector can only detect gross breakage.** An arm that
shifted the maths subtly would hide inside a distribution where 95.67% of
tokens already move by more than 1e-3 between identical runs. The aggregate
loss, stable to 2.9e-05, is the sharper instrument, and the LPS-1063 failure
shape (silent mis-attention) would move the distribution wholesale rather
than by a hair, so the gate still has teeth for the failure it is aimed at.

`tools/parity_floor_stats.py` computes this and can judge a later pair
against the stored floor as ratios, with no absolute bar anywhere.

## 5. The projection, and the arm-order decision it forces

Effective ceiling 247.7 GiB = the 267.7 GiB card (confirmed by CUDA on this
box) less the ~20 GiB cold-allocator burst. Gate-#1 FAIL line = 227.7 GiB.
Glue-alone is the primary projection: the base already contains the 0.19
GiB/set inputs, so glue+input double-counts.

Because the glue row is unmeasured (§0), the band below uses the plan's own
range against the measured base:

| glue GiB/set | 3a rank 0 | 3a rank 8 | 3b rank 0 | 3b rank 8 |
|---|---|---|---|---|
| 0.64 low | 216.3 clears | 201.2 clears | 202.3 clears | 193.2 clears |
| 0.89 mid | **233.8 FAIL** | 211.2 clears | 219.8 clears | 203.2 clears |
| 1.14 high | **251.3 — over the ceiling, predicted OOM** | 221.2 clears | **237.3 FAIL** | 213.2 clears |

**Rank 0 binds at every point in the band.** Rank 8 clears throughout despite
its higher base and its 26 GiB loss head, because it holds 40 sets to rank
0's 70 — the two stages bind on different things exactly as the plan says,
and averaging them would have hidden this.

**This fires the pre-registered arm-order rule: 3b boots first.** 3a carries
`attn_proj` resident (~0.20 GiB/set, 14 GiB on rank 0) and is at or past the
fail line across most of the band, with an outright OOM predicted at the top
of it. 3b offloads that and clears at low and mid glue. Running the arm most
likely to OOM first would have confounded bring-up failures with memory
failures on the most expensive configuration available.

## 6. The DSA topk stash row — reported, not trusted

| rank | GiB at peak | count | expected | top alloc sites |
|---|---|---|---|---|
| 0 | 3.00 | 24 | ~22 | `graph.py:869`, `rope_utils.py:136`, `gemm.py:211` |
| 8 | 2.50 | 20 | ~10 | `graph.py:869`, `rope_utils.py:136`, `gemm.py:211` |

Rank 0's count is close to the expected 22, but **rank 8's is double its
expected 10, and on both ranks the printed allocation sites are
autograd/RoPE/GEMM call sites rather than `dsa.py` or the indexer.** That is
the size-alias hazard the plan warned about, confirmed live: the tool
isolates these blocks by exact byte size, and at this geometry other tensors
share that size. **Treat this row as an upper bound of unknown composition,
not as the DSA stash.**

It does not change any conclusion here: whatever these blocks are, they are
paid TODAY under full recompute, so they are already inside the measured base
and are not a cost of the plan.

## 7. Config dump — the geometry matches the plan exactly

Read from the checkpoint config pre-boot (`ba978f7d`, 78 layers):

- `indexer_types`: 21 full / 57 shared. **Stage 0 (layers 0-37) = 11 full + 27
  shared; stage 1 (38-77) = 10 full + 30 shared.** Matches the freq-4 sharing
  rule (11 leaders / 10 leaders) exactly.
- `mlp_layer_types`: 3 dense / 75 sparse, **all 3 dense on stage 0, none on
  stage 1.** Matches.

So the census denominators stand as the plan states them: rank 0 = 70 MoE
sets + 6 dense sets, rank 8 = 40 MoE sets.

## 8. Canary

| window | run A loss | run A grad norm | run B loss | run B grad norm |
|---|---|---|---|---|
| warmup | 12.310 | 0.404 | 12.191 | 0.266 |
| traced | 12.307 | 0.415 | 12.181 | 0.257 |
| control 0 | 12.305 | 0.466 | 12.174 | 0.255 |
| control 9 | 12.199 | 0.272 | 12.104 | 0.212 |

Loss declines monotonically across the twelve windows and continues into run
B, which is not drift: the driver steps the optimizer every window, so run A
performs twelve real updates and **run B starts from run A's weights**. By
control 9 of run A the loss reaches 12.199, marginally under the canary
band's 12.2 lower edge, for that reason.

This also corrects a piece of the runbook's method: it treats the matched
per-window `|loss A - B|` as the stepping-side noise floor. It cannot be —
the weights moved twelve steps between them, so that difference measures
training progress, not noise. The forward-only parity legs in §4 are the
numerical floor; run A's within-run window spread in §3 is the throughput
floor.

## 9. Instruments

Both were exercised before the arms, and both earned it.

- `memory_census.py` parses real snapshots and produced the tables above. Two
  corrections to how it is read: the driver now requests **1,000,000**
  allocator events (the server default of 100,000 covered 10.2 s, under half
  a d2 step, and produced a window-local composition); and its "ring wrapped"
  warning is a **false positive** at this setting — the actual window spans
  288-293 s, about eleven steps. The warning fires because the replay starts
  from zero and therefore cannot see the ~100 GiB persistent baseline
  allocated before profiling began, so dump-time live legitimately exceeds
  the replay peak. Worth fixing in the tool's heuristic.
- `copy_exposure.py` reproduces the reference trace exactly — a2a 28.55 s,
  p2p 8.55 s over a 132.5 s window, matching the plan's constants of record —
  with 66,126 background pinned device-to-host copies fully hidden (0.23 s
  resident, none over 100 us). That background is the baseline the offload
  arms read their copy numbers as a delta against.

## 10. Deviations from the runbook, and why

1. **Boot 2 (the cross-boot floor cell, §6b) was not run.** Pre-registered
   rule in `RUNG3_PREREG.md`: it costs a full weight load, and cross-boot
   noise is by construction at least within-boot noise, so an arm parity leg
   that lands inside the within-boot floor is inside the cross-boot floor too.
   The boot is spent only if an arm lands outside it and the result is
   genuinely ambiguous.
2. **The arms run out of numbered order** (3b before 3a), per §5.
3. **Run B's throughput is not used**, per §3.
4. The runbook's venv verification line imports `cudnn_frontend`; the import
   name is `cudnn`, and the `nvidia-cudnn-frontend==1.27.0` install it
   prescribes is a repair that clobbers the tree's vendored pin on a healthy
   box. Both corrected in the runbook.

## 11. Evidence

On the box under `/root/.cache/user_artifacts/lps1062_pp2/`:
`artifacts/runA/<node>/` and `artifacts/runB/<node>/` — 8 allocator snapshots
per node plus the stage-leader kineto trace (1.4 GB per node per run);
`census_runA.json`; `floor_withinboot.json`; `mem/mem.<node>.csv` (pollers);
`logs/` (boot env, trainer, census, parity).
Bench JSONs under `/root/.cache/user_artifacts/lps1062_bench/`, labelled with
job id and UTC stamp `20260820T222014Z` (census) and `20260820T220915Z`
(parity).
