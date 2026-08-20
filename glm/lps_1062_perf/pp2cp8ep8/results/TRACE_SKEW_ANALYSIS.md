# TRACE_SKEW_ANALYSIS — Run B (d4) rank 0, PP2/CP8/EP8 @131k

**Trace:** `/Users/jackrao/perf_profiles/lps-1062/pp2cp8ep8/runB_d4_rank0.pt.trace.json` (667 MB)
**Analyzed:** 2026-08-12, via Perfetto `trace_processor` SQL extraction (SendRecv slices + kineto collective-metadata args) → Python. Scripts and CSVs in `/private/tmp/claude-501/-Users-jackrao/ba38046c-82be-4a98-adea-4669695b6fb5/scratchpad/lps1062/`.

---

## PREMISE CORRECTION (read first)

**This trace contains ONE GPU timeline, not 8.** It is rank 0's per-process kineto trace: a single
process (pid 0, "python") with ~30 CUDA streams on one device. There are no GPU-1..7 timelines in
the file, and no sibling rank traces exist in `/Users/jackrao/perf_profiles/lps-1062/pp2cp8ep8/`.
Consequences:

- **Q1, Q4: fully answered** (better than planned — kineto recorded per-kernel collective metadata).
- **Q2 (cross-GPU arrival skew, laggard-GPU histogram): not directly measurable.** Substituted two
  single-rank proxies that answer the underlying question (see Q2 section): (a) in-kernel wait of
  the collective that immediately follows expert compute = direct lower bound on cross-rank arrival
  skew; (b) routing hot-rank identity from the recorded per-rank a2a split sizes.
- **Q3 (last-arriver duration): not directly measurable.** Substituted the duration *floor*
  (min / p5 / the wait-free bwd large a2a), which equals the transfer-only time whenever rank 0 is
  itself the last arriver. The floor is tight and consistent, so the substitution is sound.
- To do Q2/Q3 properly: re-profile with all 8 local ranks recording
  (`BT_PROFILE_RANKS=0,1,...,7` or `--runtime-profile` on all ranks) and index-match kernels.

A second structural surprise, load-bearing for the whole investigation, is in Q4: the prior
"38.5 s SendRecv busy per GPU" is **84% pipeline-parallel p2p wait, not expert a2a**.

---

## Classification method (Q1 methodology)

Kineto recorded full collective metadata on every NCCL kernel slice (`Collective name`,
`In/Out msg nelems`, `dtype`, `Group size`, `Process Group Ranks`, `In/Out split size`), so
classification used **metadata, not positional inference** — the preferred path in the brief.

- Track/stream census: 1,276 `ncclDevKernel_SendRecv` kernels total on this GPU.
  - **Stream 99 (track 8): 1,260 kernels, all `all_to_allv` on PG "107" = ranks [0–7]** → the EP
    a2a. Note: EP group is **intra-node** (ranks 0–7 = node 0 = stage 0), i.e. all EP traffic is NVLink.
  - **Stream 35 (track 16): 16 kernels, no collective metadata (batched isend/irecv)** → the PP
    p2p. Count matches the "~16 p2p" prior exactly; cleanly separable by stream, no outlier
    heuristic needed.
- Class assignment within the 1,260:
  - `probs`: dtype=Float, In nelems = 131,072 (= 16,384 local tokens × top-k 8), ~0.5 MB. n=420.
  - `dispatch`-shaped (local→expert): BFloat16, In nelems = 805,306,368 = 131,072 rows × hidden
    6,144 → **1.61 GB input buffer, every call**. n=420.
  - `combine`-shaped (expert→local): BFloat16, mirror of dispatch (in/out nelems swapped). n=420.
- Validation: all 420 consecutive triplets on stream 99 are exactly
  [large-dispatch, probs, large-combine] (420/420 pattern-clean, 420/420 in/out mirror-clean).
  420 triplets = 35 MoE layers × 3 passes × 4 microbatches ✓.
- Pass labeling: triplets sharing identical (out_nelems, in_split) were grouped — each group of 3
  is the same (microbatch, layer) seen in forward / recompute / backward, ordered by timestamp.
  138/140 clean groups (2 boundary groups ragged; excluded from per-pass tables only).
- Naming caveat: in the backward pass, position-0 is combine-backward (dispatch-*shaped*) and
  position-2 is dispatch-backward (combine-*shaped*). Tables below use shape-position naming.

Units: trace_processor durations are ns; all tables report ms.

---

## Q1 — Duration stats per class: sync wait dominates, with a pass-dependent twist

### Overall (n=420 each)

| class | min | p5 | median | p90 | p99 | max | mean | total |
|---|---|---|---|---|---|---|---|---|
| dispatch-shaped (1.61 GB) | 2.61 | 3.37 | 4.48 | 5.19 | 6.15 | 6.31 | 4.44 | 1.86 s |
| probs (~0.5 MB) | 0.013 | 0.016 | 0.42 | 10.22 | 16.15 | 17.39 | 2.69 | 1.13 s |
| combine-shaped (1.61 GB) | 2.50 | 3.33 | 6.82 | 15.77 | 19.82 | **451.8** | 9.75 | 4.09 s |

### Split by pass (median / p90 / max, ms) — the real structure

| class | forward | recompute | backward |
|---|---|---|---|
| dispatch-shaped | 4.33 / 4.99 / 5.5 | 4.51 / 5.25 / 6.1 | 4.60 / 5.39 / 6.3 |
| probs | 0.16 / 0.67 / 1.4 | 0.33 / 0.84 / 1.3 | **7.31 / 14.06 / 17.4** |
| combine-shaped | **9.91 / 16.29 / 19.8** | **10.17 / 17.00 / 451.8** | 3.82 / 4.66 / 4.9 |

### Verdict

**Yes — synchronization wait, not bytes, dominates; but it lands on a different collective per
pass.** The naive overall-median comparison (probs 0.42 ms vs dispatch 4.48 ms, ratio 0.09) hides
the mechanism. Segmented by pass:

- **In backward, the 0.5 MB probs a2a costs MORE than the 1.6 GB dispatch: 7.31 ms vs 4.60 ms
  median — 1.6× — and ~1,000× its own bandwidth cost (0.5 MB @ NVLink ≪ 10 µs; observed floor
  13–16 µs).** That is pure in-kernel peer wait.
- In forward/recompute the probs a2a is nearly free (0.16–0.33 ms) and the wait lands on the
  combine instead: 9.9–10.2 ms median vs a 2.7 ms bandwidth bound → ~7 ms of wait.
- Pattern: **per layer-pass, exactly one a2a — whichever immediately follows the expert-MLP compute
  segment — absorbs a ~7–10 ms (median) straggler wait**; the other two run near the wire floor.
  (Fwd/recompute: combine follows expert compute. Bwd: the position-1 collective follows expert
  backward.)
- Aggregate: of the 7.09 s of EP-a2a kernel busy, **4.59 s (65%) is above the 600 GB/s bandwidth
  bound** = in-kernel waiting. The probs class alone contributes 1.12 s of wait carrying ~0.03% of
  the bytes.
- 88/420 probs calls exceed 5 ms; 45/420 exceed 10 ms. They occur essentially only in the
  backward-phase time buckets.

---

## Q2 (adapted) — Skew fingerprint from one rank

Cross-GPU arrival skew cannot be computed from this file (single timeline). Two proxies:

### (a) Skew magnitude — in-kernel wait of the post-compute collective

The collective right after expert compute waits inside the kernel for the slowest peer, so its
duration minus its transfer floor is a lower bound on cross-rank arrival spread at that point:

| proxy | median | p90 | max |
|---|---|---|---|
| bwd probs duration (transfer ≈ 0) | 7.3 ms | 14.1 ms | 17.4 ms |
| fwd/recomp combine wait (dur − 2.7 ms floor) | ~7.2 ms | ~13–14 ms | 449 ms (one outlier) |

Consistent across both proxies: **typical cross-rank skew ~7 ms, p90 ~14 ms, ceiling ~17–20 ms**
(excluding one 451.8 ms combine at t=61.2 s in recompute; next largest 100.6 / 88.1 / 81.4 ms).
Per-collective compute between a2a's is only ~7.4 ms (median probs→combine stream gap), so the
skew is the same order as the per-layer expert compute itself.

### (b) Laggard identity — hot expert-rank from recorded split sizes

Every dispatch carries rank 0's a2a row: `In split` = rows sent to each of the 8 EP ranks,
`Out split` = rows received from each. Since the straggler at combine is the most-loaded expert
rank, argmax(In split) identifies the likely laggard:

| EP rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| hot-rank share (fwd, n=138) | 11% | 4% | 12% | 16% | 10% | 12% | **26%** | 8% |

- **ROTATING, not consistent**: no rank exceeds 50%; max is rank 6 at 26%. By the brief's
  criterion this is **content/layer-dependent skew, not a single expert-routing hot spot**.
- **Within a layer the hot rank is sticky**: identical across microbatches 2–4 in 29/35 layers
  (only 2/35 across all four — µb1 routes differently; likely data content of µb1). And trivially
  identical across fwd/recompute/bwd of the same microbatch (same routing). So: **same laggard
  within a layer, rotating across layers** — the layer-intrinsic-hot-expert picture.
- Imbalance magnitude: per-dispatch send-split max/mean = **1.61× median, 2.09× p90, 2.27× max**
  (per-layer fwd averages range 1.21–2.22×). Receive-side splits are near-uniform (max/mean median
  1.03): all source ranks route similarly (CP shards of the same sequences), so the imbalance axis
  is *which expert rank* gets the tokens, not which source sends them. Rank 0's own per-layer
  expert load spans 72k–271k rows (balanced = 131k), i.e. 0.55×–2.07× mean — matching the ~1.6–2×
  hot-rank overload that the ~7–10 ms combine/probs waits imply.

---

## Q3 (adapted) — True transfer cost: the collective itself is FINE

Last-arriver duration is unavailable (single rank); the duration floor stands in for it (when
rank 0 arrives last, its kernel ≈ pure transfer, and the floor is tight):

| estimate of transfer-only time (1.61 GB buffer) | value |
|---|---|
| 600 GB/s bandwidth bound (correct payload) | 2.68 ms |
| dispatch min / p5 | 2.61 / 3.37 ms |
| combine min / p5 | 2.50 / 3.33 ms |
| bwd dispatch-shaped median (wait-free class) | 4.60 ms |
| bwd combine-shaped median (wait-free class) | 3.82 ms |
| best-case effective BW observed | 790 GB/s (dispatch), 701 GB/s (combine) |

- **Transfer-only time is ~2.5–3.8 ms — NOT tens of ms.** Nothing inside the collective is broken;
  effective bandwidth at the floor meets/exceeds the 600 GB/s microbenchmark. The peer-wait theory
  survives; there is no need to suspect the collective internals.
- Calibration note: the brief's "0.6–0.9 ms bandwidth-only" prior was too optimistic — the real
  per-call buffer is 1.61 GB (131,072 rows × 6,144 hidden × bf16), not ~0.5 GB, so the honest
  floor at 600 GB/s is ~2.7 ms. Observed floors match that almost exactly.
- Median-vs-floor: dispatch median exceeds its bound by only 1.5 ms (p90 2.4 ms) — mostly healthy;
  combine median exceeds it by 3.8 ms (p90 13.0 ms) — that's the straggler wait of Q1/Q2, not
  transfer.

---

## Q4 — Sanity totals: aggregate matches priors, decomposition overturns them

| metric | Run B rank 0 | prior (sibling clean-tip) |
|---|---|---|
| SendRecv call count | 1,276 (1,260 a2a + 16 p2p) | ~1,276 |
| SendRecv total busy | 43.31 s | ≈38.5 s |
| SendRecv avg duration | 33.9 ms | 30.5 ms |
| compute busy (union of all non-NCCL kernels) | 17.4 s | ≈18.25 s |
| a2a↔compute overlap | 0.0% vs main compute stream; 10.5% vs any kernel | ≈1% |
| kernel-active window | 70.3 s | ≈60 s |

Run B mirrors the priors in aggregate (+12% busy, same count) — **but the decomposition matters:**

| bucket | calls | busy | avg |
|---|---|---|---|
| EP a2a (stream 99) | 1,260 | **7.09 s** | 5.63 ms |
| PP p2p (stream 35) | 16 | **36.23 s** | 2,264 ms |

**The "38.5 s of SendRecv" prior is ~84% pipeline p2p wait, not expert a2a.** The 16 p2p kernels
split as: 12 short (0–4.3 ms — the actual activation/grad transfers) + 4 giant blocking waits:
**11.81 s @ t=2.75 s, 8.11 s @ t=21.6 s, 8.14 s @ t=36.5 s, 8.13 s @ t=51.4 s.** During those
36.2 s (**52% of the 70.3 s window**) this GPU runs **zero** other kernels (0.0% overlap measured
against every stream) — stage 0 is fully starved waiting on stage 1, once per ~15 s microbatch
cycle. Other NCCL on this rank is minor: AllGather 0.85 s, ReduceScatter 0.27 s, AllReduce 0.65 s.

---

## Implications (priority-ordered)

1. **The dominant cost on this node is not a2a at all — it's 36.2 s of stage-0 pipeline starvation
   (52% of the step).** Whether that is stage-imbalance (stage 1 slower: MTP/loss/more work) or
   schedule bubble needs the node-1 trace; either way it dwarfs every a2a lever by ~5×.
2. Of the 7.09 s EP a2a, ~4.6 s is straggler wait (65%) driven by ~1.6–2.1× per-layer expert-rank
   load imbalance with a rotating hot rank; ~2.5 s is irreducible transfer at current payload.
   Ceiling from perfect balance: ~4.6 s/step. Ceiling from overlapping a2a with compute
   (currently 0% vs main stream): up to ~7 s, but only ~17 s of compute exists to hide it in.
3. The probs a2a is a free skew probe: its bwd-pass duration distribution *is* the arrival-skew
   distribution (med 7.3 ms / p90 14 ms / max 17 ms).

## Caveats

- Single-rank view: all skew/laggard statements are inferred from rank 0's kernel waits and
  rank 0's row of the a2a split matrix, not from cross-timeline timing. The hot-rank histogram
  assumes the most-loaded expert rank is the last arriver at combine — sound for compute-bound
  expert MLPs, but not a direct measurement.
- "queued" arg is 0 throughout (not populated), so host-enqueue vs kernel-start delay was not
  separable; stream-gap analysis was used instead (dispatch→probs gap median 4 µs = back-to-back;
  probs→combine gap median 7.4 ms = local expert compute).
- The 2 ragged routing-groups (4 triplets) are excluded from per-pass tables; included everywhere else.
- p2p kernels carry no collective metadata (batched isend/irecv), identified by stream + count.
- GPU reports as "NVIDIA L20D" (platform masquerade for B300) — ignored per brief.
- trace_processor `export sqlite` in this build (perfetto ~v50) writes a 0-byte file (exit 0);
  worked around via repeated `-q` CSV extraction. Papercut filed.

## Reproduction

```
~/bin/trace_processor -q q_extract.sql runB_d4_rank0.pt.trace.json > sendrecv_full.csv
# q_extract.sql pivots args (Collective name, In/Out msg nelems, dtype, In/Out split size,
# Process Group Ranks) per SendRecv slice; analyze*.py in the scratchpad dir do the rest.
```
