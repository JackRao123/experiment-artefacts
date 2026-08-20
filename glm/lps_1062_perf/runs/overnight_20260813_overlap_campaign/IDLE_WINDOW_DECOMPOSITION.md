# IDLE WINDOW DECOMPOSITION — fe127 d16 rank0 (LPS-1062 overlap campaign)

**Author:** kepler · **Date:** 2026-08-13 · **Task source:** fermi assignment (idle-window attack, trace-only)
**Trace:** `~/perf_profiles/lps-1062/pp2cp8ep8/fe127_d16_rank0.pt.trace.json` (2.83 GB, 971,927 kernels, span 133.350 s, rank 0 = PP stage 0)
**Served by:** trace_processor pid 47309 on http://localhost:9001 (queried via perfetto python client, read-only)

---

## 1. Pre-registration (written BEFORE any decomposition query)

Written after identity recon only (slice count, category histogram, trace span).
No gap/idle/attribution query has been run yet at this point. Chronology is
honest: recon → this pre-registration → analysis.

### 1.0 Recon facts known at pre-registration time

- Trace span 133.350 s ≈ exactly one 133.2 s step. So the trace captures ~1
  step; a step-boundary seam, if present, will appear inside or at the edges of
  the window.
- Categories present: cpu_op 3.57M, cuda_runtime 2.11M, kernel 971,927,
  cuda_driver 174k, gpu_memcpy 82k, gpu_memset 14k, gpu_user_annotation 8,391,
  user_annotation 8,385.
- erdos's headline (CAMPAIGN_ORDERS): 20.8 s/step PURE IDLE (no kernel on any
  stream) = 15.7% of 133.4 s; ideal PP2 bubble at d16 ≈ 6% ⇒ ~10 pts attackable.

### 1.1 Idle definition

- **GPU-busy** = union of intervals of all slices with category ∈
  {`kernel`, `gpu_memcpy`, `gpu_memset`} on any stream. (memcpy/memset are real
  GPU work; a D2H stash or memset still occupies the copy engine.)
- **IDLE** = complement of GPU-busy within [first GPU work, last GPU work].
- **Secondary measure (reconciliation):** kernel-only complement, to reproduce
  erdos's 20.8 s ("no kernel on any stream"). If the two measures differ
  materially, the difference = memcpy/memset-covered windows and is reported.
- Micro-gaps < 1 ms are launch-overhead noise: counted as one aggregate line,
  not bucketed as attackable slack.

### 1.2 Bucket definitions (pre-registered)

- **Bucket A — structural PP2 pipeline idle.** Idle windows strictly INSIDE the
  step's compute region (between the first forward kernel of the step and the
  last backward/grad-comm kernel of the step) that do NOT overlap a named
  host-side non-pipeline activity (optimizer, save, logging, data). On rank 0
  (PP stage 0, 1F1B with M=16) this is expected to be drain-dominated (waiting
  on stage-1 backward grads) plus any mid-step convoy/imbalance stalls.
  Sub-split by position: early / mid / late third of the in-step region.
- **Bucket B — step-boundary seam.** Idle window(s) between the last
  backward/grad-comm kernel of step N and the first forward kernel of step N+1
  (with a 1-step trace: the idle region adjacent to the optimizer phase,
  including trace-edge adjacency). Each seam gap is attributed to the
  host-side slices (cpu_op / cuda_runtime / user_annotation) overlapping it,
  sub-bucketed by named activity:
  - **B1 optimizer step** (Adam / clip / grad-norm / param update host work)
  - **B2 data loading / next-batch host prep** (dataloader, collate, H2D prep)
  - **B3 telemetry / logging** (wandb, tensorboard, timers, printing, metrics)
  - **B4 checkpoint / save cadence**
  - **B5 other host** (anything else; named per-slice in results)
- **Bucket C — other.** Idle neither A nor B: profiler start/end edge effects,
  plus the aggregate of sub-1 ms micro-gaps (reported as one line).

### 1.3 Predictions (pre-registered guesses; the task exists to test them)

- A ≈ 6–9 s (near the ~6% ideal = 8.0 s, plus some convoy slack)
- B ≈ 10–13 s, expected dominated by B1 (optimizer host time) and B3
  (telemetry) per the orders' suspect list
- C ≈ 1–3 s
- Reconciliation: kernel-only idle should reproduce 20.8 s ± 0.5 s.

### 1.4 Method — exact queries (to be run against :9001)

**Q1 gap extraction** (all GPU work; the `category='kernel'` variant gives the
secondary measure):

```sql
WITH gpu AS (
  SELECT ts, dur, ts+dur AS end_ts FROM slice
  WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur > 0
),
g AS (
  SELECT ts, end_ts,
    MAX(end_ts) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
      AS prev_end
  FROM gpu
)
SELECT prev_end/1e9 AS gap_start_s, ts/1e9 AS gap_end_s, (ts-prev_end)/1e6 AS gap_ms
FROM g
WHERE prev_end IS NOT NULL AND ts > prev_end
ORDER BY gap_ms DESC;
```

**Q2 reconciliation totals:** sum of Q1 gaps (all gaps; and gaps ≥ 1 ms) vs
20.8 s.

**Q3 host attribution per gap** (for each gap > 50 ms): overlapping host slices
ranked by overlap time:

```sql
SELECT s.name, s.category, s.ts/1e9, s.dur/1e6,
       (MIN(s.ts+s.dur, :gap_end) - MAX(s.ts, :gap_start))/1e6 AS overlap_ms
FROM slice s
WHERE s.ts < :gap_end AND s.ts+s.dur > :gap_start
  AND s.category IN ('cpu_op','cuda_runtime','user_annotation','cuda_driver')
ORDER BY overlap_ms DESC LIMIT 25;
```

plus the enclosing user_annotation stack at gap midpoint for naming.

**Q4 step structure:** user_annotation name census (top by total dur);
CheckpointFunction / CheckpointFunctionBackward counts (expect 16 microbatches
× layers-per-stage); first-forward / last-backward kernel timestamps per step;
optimizer-region markers (names matching %optim%, %clip%, %norm%, %save%,
%log%, %wandb%, %data%).

**Q5 bucket assignment:** each gap ≥ 1 ms assigned to A / B1–B5 / C by the
rules in §1.2; sums per bucket; A sub-split by step-phase thirds.

**Q6 rank-8 cross-check (caveated):** no fe127 d16 rank8 trace exists Mac-side
(hausdorff sweep pulled fe127 rank0 only; rank8 exists only for the l3 config
— different code path). Cross-check is therefore LIMITED: if used at all,
l3_d16_rank8 gives only a qualitative "does stage 1 compute while stage 0
idles at the seam" reference under a different config. Stated as a caveat, not
evidence.

---

## 2. Results (filled AFTER pre-registration)

All numbers measured on :9001 (and, after the incident in §4, my own instance
:9002 serving the identical file). Reconciliation: kernel-only idle =
**20.797 s** = erdos's 20.8 s headline ✓ (method validated). Primary all-GPU
measure (kernel ∪ memcpy ∪ memset): **20.226 s** across 898,031 gaps. The
0.57 s delta = memcpy/memset-covered windows.

### 2.1 Headline decomposition

| bucket | idle (s/step) | % of 133.4 s step | what it is |
|---|---:|---:|---|
| **A — structural PP2 fill/drain bubble** | **≈ 0** | ~0% | **Not present in the idle budget.** On stage 0, PP waits are posted as NCCL SendRecv kernels = GPU-busy → they live in erdos's 34.3 s *exposed SendRecv* line, not in pure idle. Zero of the 495 ≥1 ms gaps are unattributed pipeline waits. |
| **B — step-boundary seam** | **1.95** (≥1 ms: 1.93; micro: 0.02) | 1.5% | Seam wall = 2.035 s (last bwd end 3670531.670 → trace end), 95% idle. Nearly all **untraced host python** (gaps of 1294 / 414 / 116 ms). Optimizer proper is tiny: `Optimizer.step#FusedAdam.step` annotation 28.8 ms; adam GPU kernels 1.0 ms (13 `multi_tensor_adam`) + 456+456 bf16 copy-back storm ≈ 3 ms. |
| **C — diffuse micro-gaps (<1 ms)** | **10.41** | 7.8% | 897,536 gaps, mean 11.6 µs, p50 8.7 µs, p99 66 µs; spread uniformly across the whole step (~0.3–0.5 s per 5 s bin). Per-kernel host-dispatch tax at 972k kernels + 96k memcpy/memset per step. |
| **In-step ≥1 ms named host stalls** | **7.88** | 5.9% | See §2.2 — dominated by the DSA indexer `nonzero` sync. |

(A+C+B+in-step-named accounting: the ≥1 ms in-step named stalls 7.88 s and the
micro-gap aggregate 10.41 s are disjoint by construction; seam 1.95 s disjoint
by region. 7.88 + 10.41 + 1.95 = 20.23 s ✓.)

### 2.2 In-step ≥1 ms gaps by named host activity (n=495, 9.81 s)

| class | s/step | n | signature (host stack) |
|---|---:|---:|---|
| **DSA indexer `nonzero` D2H sync** | **4.06** | 234 | `CheckpointFunction(Backward) → aten::index → aten::nonzero → cudaStreamSynchronize`; GPU bookends: cub reduce/compare → (gap) → cub DeviceSelectSweep/index kernels. Data-dependent top-k count readback in the sparse-attention indexer, firing in fwd (1.52 s) and bwd recompute (2.55 s). Split by step third: 1.16 / 1.82 / 1.09 s. |
| **MoE a2a dispatch host block** | **1.20** | 44 | `CheckpointFunction(Backward) → _AllToAll → aten::new_empty → cudaEventQuery` (spinning 100–160 ms!). Between MoE permute kernels and the dispatch SendRecv. Allocator/event stall sizing the a2a recv buffer. fwd 0.40 / bwd 0.80 s. |
| **bwd `aten::cat` host block** | **0.71** | 19 | `CheckpointFunctionBackward → aten::cat → cudaEventQuery` spin; bookended by DtoH pinned and CatArrayBatchedCopy. |
| misc small host syncs | 1.91 | 188 | `aten::to/_to_copy/item/empty` + small DtoH reads inside fwd/bwd (per-layer scalar reads; indexer-adjacent). |

### 2.3 Time-weighted overlap (union trick — covers micro-gaps too)

Idle time overlapping host slices of a given class (exact interval union vs the
full 20.226 s, not just ≥1 ms gaps):

| host-signature set | idle covered |
|---|---:|
| `aten::nonzero` alone | **6.97 s** |
| nonzero + index | 7.81 s |
| `_AllToAll` | 1.09 s |
| `aten::cat` | 1.15 s |
| `cudaStreamSynchronize` | 5.11 s |
| `cudaEventQuery`/`cudaEventSynchronize` | 2.76 s |
| **ALL indexer/a2a/cat host ops (dedup'd union)** | **11.95 s** |

`aten::nonzero` census: **55,328 calls, 33.97 s host time, avg 614 µs** —
matches erdos's "nonzero 34 s/55k, absorbed" line. The decomposition shows
**6.97 s of it is NOT absorbed** — it leaks as pure GPU idle (4.06 s in ≥1 ms
gaps + ~2.9 s of the micro-gap aggregate). Of the 10.41 s micro-gap aggregate:
~6.0 s overlaps indexer/a2a/cat host ops; **~4.4 s is generic per-kernel
dispatch tax** (cudaLaunchKernel/python dispatch between the step's ~1.07 M GPU
ops).

### 2.4 Seam narration (GPU-side bookends, all timestamps trace-absolute s)

1. last DSA bwd kernels (2× ~22 ms `dsa_bwd`) end ~3670531.694
2. `ReduceScatter_Sum_bf16` 3.7 ms (grad RS over CP8) + `AllReduce_bf16` 0.74 ms + 3× `Broadcast` (grad-norm/loss comm)
3. **DtoH pageable 18.2 ms** (metrics readback) + DtoH pinned
4. **[413.7 ms gap — untraced python]**
5. HtoD pageable + `AllReduce_Sum_u64` 0.07 ms (flag/count) + DtoH + HtoD
6. **[115.9 ms gap — untraced python]**
7. HtoD + DtoD + 2× `Broadcast`
8. **[1294.3 ms gap — untraced python]** (only 0.54 ms of aten ops inside; no dataloader-worker aten activity on any thread)
9. HtoD (KB-scale scalars) → `FusedAdam.step` annotation 28.8 ms containing the 1.0 ms adam kernels → 456× bf16 copy-back + 456× direct_copy → `AllReduce_Sum_u64` **15.0 ms** (post-step flag; size = cross-rank skew absorber) + `AllReduce_f32` 1.6 ms → DtoH/HtoD → 2× `Broadcast` → trace end 3670533.705

The 1294 ms hole sits between grad-norm broadcast and the optimizer step. The
trace carries no python stacks (HTTP-driven path is slimmed), so its content is
**unresolvable from this artifact alone**: candidates are Megatron
optimizer-step python glue, logging/timers/wandb, and next-batch host fetch
(pure-python/IO wouldn't show aten ops). A py-spy dump at the seam (or a
`with_stack=True` env-gated trace) resolves it in one shot.

### 2.5 Predictions vs actuals (honest chronology)

| pre-registered guess | actual | verdict |
|---|---|---|
| A ≈ 6–9 s structural bubble | ≈ 0 s in idle (it lives in exposed-SendRecv) | **wrong** |
| B ≈ 10–13 s, optimizer+telemetry dominated | 1.95 s; optimizer 0.03 s; dominated by untraced python | **wrong on size**; right that it's host-side |
| C ≈ 1–3 s | 10.41 s micro-gap aggregate | **wrong** — diffuse is the biggest single line |
| (not suspected) | DSA indexer nonzero sync = 6.97 s time-weighted | the dominant named class was **not on my suspect list** (erdos's map had it as "absorbed") |

### 2.6 Ranked attackable slack

| # | target | est. win (s/step @133.4 s) | % step | mechanism / fix direction |
|---|---|---:|---:|---|
| 1 | **DSA indexer sync-free rework** | 4–7 | 3–5% | 55k `nonzero` D2H syncs/step (34 s host, 6.97 s leaked idle). Fixed-shape top-k (pad-to-max + mask, no nonzero), or device-side count consumption by the (already custom cutlass) indexer kernels. Also relieves the micro-gap tax (~2.9 s of it). |
| 2 | **generic dispatch tax** (1.07 M GPU ops/step) | 2–4 | 1.5–3% | kernel-count reduction / fusion (trace is full of µs-scale FillFunctor/add/copy), CUDA-graph shape-static regions. Harder, no single window. |
| 3 | **seam untraced python** | 1–1.5 | ~1% | py-spy at the seam to name it (one cheap live probe), then async logging / prefetch / glue slimming. Seam measured 1.95 s is a **lower bound** — anything outside the ProfilerStep window is invisible to this artifact. |
| 4 | **MoE a2a host block + bwd cat** | 1–2 | ~1.5% | allocator/event spin in `new_empty` under `_AllToAll` (persistent pre-sized a2a buffers); `aten::cat` event spin in bwd. **Disjoint from the 34.3 s exposed-SendRecv prize** (that is GPU-busy time; this is pure idle) but the fix path overlaps lever-1 contract-shim territory. |

Sum of realistic wins ≈ 8–14 s/step (6–10%), consistent with the orders'
"~10 points attackable" estimate — but the composition is entirely different
from the pre-task framing: **~0 s is pipeline bubble; the hole is host-side
syncs (indexer), diffuse dispatch tax, and seam python.**

### 2.7 Caveats

- **Rank-0 only.** fe127 d16 rank8 was never pulled Mac-side (hausdorff sweep:
  fe127 rank0 only; rank8 exists only for the l3 config — different code path,
  not used). The named classes (indexer/a2a/cat syncs, dispatch tax) are
  per-rank-local host stalls — every rank runs the indexer — so classes should
  transfer, magnitudes may differ. The "≈0 structural bubble in idle" finding
  is a stage-0 statement; stage 1's fill would likewise surface as
  SendRecv-exposed, not idle.
- Seam is partially captured: the trace ends at the ProfilerStep#0 boundary;
  post-optimizer logging/save and next-step data load (if outside the
  annotation) are invisible. 1.95 s is a floor.
- Micro-gap "idle" includes irreducible launch latency (~2–5 µs); the
  attackable share of the 4.4 s generic tax is the conservative 2–4 s quoted.
- Overlap measures in §2.3 are per-class; only the "ALL" row is deduplicated.

## 3. Method additions made during analysis (honest chronology)

- Gap classification loop (per-gap top-6 overlapping host slices → signature
  rules) — written after seeing the top-14 attributions; rules frozen before
  the full 495-gap run.
- **Union-trick overlap measure** (§2.3): `idle(union(GPU, X))` vs baseline
  `idle(GPU)`; difference = idle covered by X. Developed because the naive
  gap×slice interval join was O(N×M) and **wedged the shared :9001 server**
  (see §4). Same window-function cost as the base gap query — seconds.
- Sub-1ms host-overlap sample (300 random micro-gaps): 32% indexer-adjacent —
  consistent with the exact union-trick number; superseded by §2.3.

## 4. Incident report — shared :9001 server state

At ~19:2x CDT I issued a naive 898k×55k interval LEFT JOIN against the shared
trace_processor (pid 47309, :9001). trace_processor executes queries serially;
the join is still churning server-side (98% CPU, observed) and the server does
not answer new queries until it drains. **Per orders I did NOT kill pid
47309** — it remains alive and will recover when the query finishes (or fermi
may restart it at the cost of a ~2 min reload; the trace file is local and
unchanged). All results above were completed on my own instance:
`trace_processor_shell --httpd --http-port 9002` (pid 29344, same trace file),
which I leave running for fleet use. My mistake — the join should never have
been sent to the shared server; the union-trick formulation is strictly better
anyway.
