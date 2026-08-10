# LPS-1062 round-3 leverboard — where the next step-time wins are (fibonacci, 2026-08-09 PM)

**Constraints (Jack, 2026-08-09):** config must support 131k context with **≥10 GiB/GPU headroom**; loss must be **numerically the same** (no optimization may perturb it — bitwise where achievable, and nothing beyond established reduction-order noise ≤2e-3 anywhere).

**Frontier being optimized:** golden EP16/CP16 2×8 B300 + ship NCCL env + `BT_TF32_LM_HEAD` + `BT_DSA_CP_LAYOUT_CACHE` + `BT_THD_ROPE_HOST_CACHE` → **715 tok/s/GPU steady @131k×d4 (46.8 s/step), 726 @16k×d32**, peak ~202 GiB (≈71 GiB headroom) — both constraint-compliant.

## Measured anatomy of the current step (gated-v2-4mb-steady trace, 46.68 s, trace_processor SQL)

| component | wall | note |
|---|---:|---|
| EP a2a SendRecv resident (stream 83) | **26.03 s (55.8%)** | 1,800 token calls 20.90 s + **900 probs calls 5.13 s** |
| …of which overlapped by any compute | **0.395 s (1.5%)** | comm/compute fully serialized (matches Jack's eyeball) |
| compute union (all non-NCCL kernels) | 16.99 s (36.4%) | |
| CP AG/RS + misc NCCL on compute stream | ~2.2 s | serialized with compute by construction |
| GPU idle | 1.82 s | host-blocks already eliminated by B+F |

Key sub-facts: dispatch-size token a2a (805 MB) p50 ≈ 10.7 ms but **min 4.3 ms (187 GB/s)** → ~2.3× wait/skew slack per call at p50; probs a2a averages **66 KB yet 5.7 ms** — 5.13 s of pure latency serialized on the same NCCL stream; combine sizes spread up to 3.5× (routing imbalance, rank-0 view only). Perfect-overlap floor = max(26.0, 17.0) ≈ **28–30 s/step (+55–65 % ceiling)**.

## Ranked levers

| # | lever | expected win (131k×d4) | numerics | memory | status |
|---|---|---|---|---|---|
| **W1** | **De-serialize the probs a2a** (2nd communicator/stream, or pack with tokens; 900 calls/step, 7,200 at 16k×d32) | **−4 to −5 s (+9–11%)**; larger share at 16k×d32 | bitwise-safe (no reductions) | ~0 | Helmholtz designing (supplement sent) |
| **W2** | **Chunked MoE pipeline, K=2 by local-expert groups** via list-form `all_to_all` over per-peer **views** (zero copies; combine buffer byte-identical ⇒ unpermute unchanged). Row-chunking variant proven unsound (dest-rank-contiguous). Design: DESIGN_helmholtz.md | **−3.0 to −4.4 s standalone** (fwd −5.1 ms/layer-pass; bwd v2 seq-bumps −6–8 ms); de-skew upside on top; −1.5 s marginal if W3 lands | bitwise by construction except V1 (GEMM kernel-schedule vs num_gemms 16→8) — `torch.equal` gate T2 blocks ship | ≈0 (views) | Helmholtz implementing (after W1) |
| **W2b** | **Lookahead recompute** (overlap bwd(L) with recompute-fwd(L−1) across layers, inside full recompute; 2/3 of a2a lives in the bwd phase) | **−8 to −12 s** | bitwise (pure scheduling; dropout=0; RNG fork/restore per checkpoint) — needs V4 DSA-holder audit + CDMC soak (prod runs CDMC **unset**; devbox pins =1 — historical numbers are =1; LPS-1003 race fix must be verified in pin first) | +2–4 GiB @131k; +8–12 GiB @16k×d32 (measure) | designed (§7 of memo); build after W1/W2 |
| **W3** | **FIX C: dispatcher replay-metadata reuse** (+fold FIX D; then re-test parked FIX A-v3) | +2–6% now; compounds with W1/W2 — every CPU stall goes critical once comm is hidden | bitwise iff replay-deterministic; VERIFY mode asserts it on-box before timing | ~0 | **Hilbert: DONE on Mac** (fixc.patch, 49-assertion CPU suite green; pass-frame keying via recompute.py marker); awaiting on-box verify-soak + A/B |
| **W4** | **F2 fix: phantom partitions for DP>1** (design approved 08-09 PM: all_reduce-MAX partition count on pure-DP group + dummy-token zero-masked suffix partitions; `f2_fix/DESIGN_F2_bohr.md`) | not a 2-node throughput lever (patched golden already ≈ CP8/DP2: 726 vs 734 @16k×d32) — value = **correctness for any DP>1 customer** + 4-node EP16/CP8/DP4 scale-out (≈23.2 K agg tok/s, 2× the 2-node winner) + unlocks the never-run expH (EP16/CP16/DP2) probe | no change at DP=1; DP factoring changes reduction order only (documented cross-config tolerance) | CP8 meshes fail the ≥10 GiB bar today (3.5 GiB @16k×d32) → not ship until trimmed; F5 thrash also open | Bohr designing |
| W5 | Multi-rank straggler/imbalance attribution | **DONE 08-09 (rank0+rank8, 318g61w)**: real cross-rank per-call tail ~1–6 ms p90 (≈ the +1.3 s box-variance delta); hot-expert→straggler correlation WEAK (ρ≈+0.09) — routing imbalance is NOT the dominant skew driver at 131k×d4; W2's "de-skew upside" downgraded to speculative bonus. `round3/anchor-318g61w/straggler_summary.md` | n/a | n/a | done (2/16 ranks caveat) |
| W6 | CP AG/RS off the compute stream | ≤−2 s | bitwise-safe (scheduling) | ~0 | opportunistic, inside W2's stream work |
| W7 | CUDA graphs / graphed callables | deferred: B+F already collapsed GPU idle to 1.8 s; routed shapes are dynamic per datum | — | — | revisit only if launch pressure returns at 16k×d32 post-W1/W2 |

**Stacked estimate if W1+W2+W3 land** (static, before de-skew upside): 46.7 → ~30–34 s ≈ **1,010–1,150 tok/s/GPU (+40–60%)**. Conservative floor (W1 + half of W2): ~+25%.

## Ruled out under the current constraints (do not revisit without new facts)

- **fp8 / compressed a2a payload** — the one big unexplored comm-volume lever, but it changes loss numerically → **excluded by constraint**.
- **DeepEP / flex dispatcher** — NVSHMEM IBGDA dead on LAG-bonded RoCE (`data_direct support: 0`); infra qualification ticket remains open; not code-fixable here.
- **mcore `overlap_moe_expert_parallel_comm`** — forbidden under full recompute; non-recomputed MoE needs ~150 GB/rank even at 131k (>71 GiB headroom); and the 1F1B schedule degenerates at num_microbatches=1 per THD partition. W2 is the replacement that works under recompute.
- **NCCL channel escalation (16 chan / MAX=64)** — +1–3% for −3.5 GiB headroom; knob space beyond (ALGO/PROTO/BUFFSIZE) untried but token-a2a already hits 187 GB/s best-case → the residual is wait, not wire; overlap dominates.
- **CP8/DP2 as the 2-node ship** — 3.5 GiB headroom at customer shape (violates ≥10 GiB) + F5 thrash open; patched golden matches it within 1% anyway.
- **FIX A alone** ("the torch.nonzero thing") — already tried, measured ≈0 (−1%, noise), parked default-OFF; its win is gated behind FIX C (the dispatcher replay sync upstream throttles the same path). Re-test as A-v3 **after** W3.

## Sequencing on the box (single 2×8 B300, benches per PROTOCOL.md)

1. Anchor re-verify: patched-BF golden @131k×d4 (expect ~715) and @16k×d32 (~726); gates armed via WARNING lines; env from `/proc/<pid>/environ`.
2. **W1 A/B** (smallest patch first): step time + bitwise canary (same `--warmup-datums`), then trace capture → probs-a2a wall → ~0.
3. **W3 verify-soak** (BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1, ≥20 steps, replay-vs-fwd bitwise assert) → timed A/B → check_acceptance rows.
4. **W2 prototype A/B** at K=2 then K=4: step time, canary, peak-mem ≤ cap−10 GiB at 131k **and** 16k×d32, trace → overlap% (target ≥40% of token-a2a hidden).
5. Piggyback rank-8 trace on every profiled boot (W5).

Protocol invariants: steady windows 2–3 only; canary ≤2e-3 pass / >5e-3 stop; identical warmup-datums vs reference; telemetry counters mandatory (inert-gate lesson); trace_processor v56.1 for the checker.
