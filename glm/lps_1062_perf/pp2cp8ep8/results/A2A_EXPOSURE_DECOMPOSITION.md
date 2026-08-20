# A2A EXPOSURE DECOMPOSITION — d16 L3 traces (gauss, 2026-08-13 ~02:5x CDT)

**Traces:** `traces/l3_d16_rank{0,8}.pt.trace.json` (one full d16 step, driver
step-2 steady state, +4.0% kineto tax). Method: perfetto on box; every
`ncclDevKernel_SendRecv` (EP all-to-all) kernel linked via its
`args.correlation` → `cuLaunchKernelEx` → parent chain to the enclosing
`CheckpointFunction` (FWD) / `CheckpointFunctionBackward` (REPLAY — the
full-recompute re-run of forward) / `_AllToAllBackward` (BWD). Call roles by
order+dtype within each layer-pass (BF16=tokens, Float=probs). Exposure =
kernel time with zero concurrent compute-kernel overlap (compute = all
non-comm kernels). Wire floor = payload bytes (args `In msg nelems` × dtype
size) at the empirical p95 achieved rate (611 GB/s rank 8, 586 GB/s rank 0;
max 733 — NVLink intra-node EP8). Script `lps1062_pp2/l3_a2a_decomp.py`;
JSONs beside the traces.

**Cross-check:** my exposed total 31.4s (rank 8) vs borel's bucket 30.9s —
1.5% definitional delta (compute-overlap vs his method). Counts exact:
5,760 a2a calls rank 8 (40 MoE × 16 mb × 9), 5,040 rank 0 (35 MoE × 16 × 9).

## 1. Headline split (per step, rank 8 / stage 1)

| | duration | exposed |
|---|---:|---:|
| total a2a | 34.15 s | **31.38 s** |
| transfer floor (bytes @ p95 wire) | 10.20 s (30%) | ≈9.4 s |
| wait (dur − floor) | 24.02 s (70%) | ≈22 s |

Rank 0 (35 MoE layers): duration 29.96s, exposed 27.47s, floor 9.25s, wait
20.75s. Same structure.

## 2. By pass — the full-recompute comm tax is the largest single block

| pass | exposed (rank 8) | share |
|---|---:|---:|
| FWD (first pass) | 9.85 s | 31% |
| **REPLAY (recompute re-run)** | **11.67 s** | **37%** |
| BWD (autograd backward) | 9.85 s | 31% |

Full activation recompute re-executes the dispatcher — including its
all-to-alls — so the forward-direction comm runs TWICE per microbatch. The
replay share (11.7s exposed) is load-bearing today (the dispatched activations
are consumed by the backward), and it is attackable only by reducing recompute
(the per-layer recompute dial / L0b program) or overlapping it — not by any
dispatcher-side cache. (FIX C caches the replay's host-side metadata syncs,
a different, smaller class.)

## 3. By call type — combine carries the imbalance wait; probs-grad is pure latency

| role (pass) | calls | dur | exposed | floor | wait | p50 | p90 | p99/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combine (FWD) | 640 | 7.43 | 7.23 | 1.72 | 5.73 | 10.6ms | 19.2 | 27.0 |
| combine (REPLAY) | 640 | 9.21 | 9.05 | 1.72 | 7.52 | 10.6 | 21.9 | **108.7 / 270.8ms** |
| **probs_grad (BWD)** | 640 | 5.41 | 5.41 | ~0 | **5.41 (100%)** | 7.9 | 17.0 | 24.6 |
| dispatch (FWD) | 640 | 3.06 | 2.41 | 1.68 | 1.38 | 4.8 | 5.3 | 6.0 |
| dispatch (REPLAY) | 640 | 3.06 | 2.40 | 1.68 | 1.37 | 4.8 | 5.3 | 6.0 |
| combine_grad (BWD) | 640 | 2.83 | 2.22 | 1.68 | 1.15 | 4.5 | 5.1 | 6.0 |
| dispatch_grad (BWD) | 640 | 2.73 | 2.22 | 1.72 | 1.04 | 4.2 | 5.1 | 6.7 |
| probs (FWD+REPLAY) | 1280 | 0.43 | 0.43 | ~0 | 0.43 | 0.3 | 0.7 | 1.3 |

- **Combine = 52% of exposed a2a** (16.3s of 31.4s). Its wait (13.2s) is the
  expert-imbalance straggler: combine cannot finish until the slowest expert
  rank returns. The REPLAY-combine p99 of 108ms (max 270ms) is the long tail.
- **probs_grad in backward: 5.41s exposed, ~100% wait** (528KB payload → ~1µs
  floor; p50 7.9ms is pure latency/straggler). This is exactly the Aug-10 W1
  lever's target (probs a2a on a second communicator; W1 measured −3.4…−5.1s
  model). NOTE: W1 stays prohibited here — PP2/CP8/EP8 with 16 ranks is a
  multi-EP-group topology, and the fixed build is unvalidated there
  (Aug-10 SCORECARD caveat).
- Dispatch calls are tight (p50 4.8ms vs 2.7ms floor; wait ~1.4s/pass-class).

## 4. Wait attribution (as far as 2 ranks allow)

- **Own-size decorrelation:** dispatch wait vs own payload size correlation
  ≈ 0 (rank 8: −1e−16). The wait is imposed by peers, not by the call's own
  transfer — the imbalance/straggler signature, not a bandwidth effect.
- **Stage asymmetry (matches the CPU-side asymmetry in my host-sync deep
  dive):** rank 0's dispatch/combine tails are fat (dispatch p99 81–88ms,
  combine max 190ms) where rank 8's are tight (p99 ~5.7ms). Rank 0's wire
  p95 is also lower (586 vs 613 GB/s). Stage 0 pays the messier half of the
  a2a.
- **Laggard stickiness:** rank 8 per-layer mean dispatch wait is tight
  (1.5–2.5ms across 40 layers — no sticky layer). Rank 0 is sticky: per-layer
  mean wait spreads 1.9→8.3ms with persistent top offenders (stage-0 MoE
  layers 20, 3, 33, 18, 8, 4 at 6.7–8.3ms vs 4.5ms median). Within one step
  the 16 microbatches carry different data, so the stickiness is structural
  (expert assignment/capacity or EP-rank mapping), not data luck. CAVEAT:
  one step, one rank per stage — a multi-step trace would be needed to
  confirm persistence across steps.

## 5. Sizing output — what the 30.9s is made of

Per step (rank-8-anchored, step-level; stages overlap under M=N so the wall
sees ~the rank max, which is also what borel's 30.9s measured):

| component | size | lever that attacks it |
|---|---:|---|
| **(a) irreducible transfer floor** | **~10.2 s/step** | none (real data movement at 611 GB/s achieved); hideable behind compute in principle |
| **(b) straggler/imbalance wait** | **~24 s/step duration, ~22 s exposed** | rebalancing (capacity/balancing) for the combine share (13.2s); W1-class probs offload for probs_grad (5.4s, currently prohibited on multi-EP-group); dispatch wait small (2.7s) |
| **(c) schedule-hideable** | **~25–28 s/step** | L0b (big-overlap flag made feasible at 131k) / per-layer recompute dial: today only ~8% of a2a duration overlaps any compute; the steady-state exposed mass is in principle hideable behind another microbatch's compute. Bound: fill/drain portion (~2/16 of microbatches ≈ up to ~12% of calls) has no partner compute — not hideable by mb-interleave |

Prize mapping for the morning: L0b/per-layer-dial prize ceiling ≈ (c) ≈
25–28s/step (~19–21% of the 131.4s step); a rebalancing lever prize ≈ the
combine wait ~13s; the L1 flag (dispatch-bwd wgrad overlap) covers only the
dispatch_grad share (~2.2s exposed) — consistent with its +1.5–3% EV. The
replay-comm tax (11.7s exposed) is inside (c) and is *additionally* reduced
by any recompute-reduction (the dial removes the replay's a2a outright for
non-recomputed layers).

## 6. Caveats

- One traced step; kineto tax +4.0% inflates all host-adjacent timings
  slightly (durations are GPU-side kernel times, minimally taxed).
- Exposure definition: zero concurrent *compute* kernel (comm-comm overlap
  e.g. a2a ∥ p2p still counts as exposed). Borel's 30.9s vs my 31.4s is the
  definitional delta; conclusions are insensitive.
- Transfer floor at the empirical p95 achieved rate; at the Aug-9 convention
  600 GB/s the floor rises ~2% — immaterial to the splits.
- Wire floor treats the full per-rank input payload as moved once; NCCL
  send/recv ring effects inside the EP8 group are absorbed into the
  calibrated rate.
- Layer stickiness is single-step evidence (structural vs data confound
  noted in §4).
