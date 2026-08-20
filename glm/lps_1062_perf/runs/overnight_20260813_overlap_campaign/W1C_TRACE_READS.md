# W1c TRACE READS — pre-registered (jacobi, ex-kepler; 2026-08-14 ~00:xx CDT)

**Filed BEFORE the W1c traces land.** Window: **B/F A/B** — FIX B
(`BT_DSA_CP_LAYOUT_CACHE=1`, CP layout-builder cache, PR #26) + FIX F
(`BT_THD_ROPE_HOST_CACHE=1`, THD RoPE host cache, PR #26), d16 pair (and d4
arms box-side), fixed wheel, `BT_PROFILE_RANKS=0,8`, one traced d16 step.
Driver: lovelace. My lane: pull Mac-side immediately, sha256, serve
:9006/:9007, read, report.

**Baselines:** fe127_d16_rank0 (fixed wheel; my decomposition/residual-map
numbers); l3_d16 rank0/rank8 (old wheel) for the stage-asymmetry lineage.
**Provenance caveat (standing, via bayes):** tonight's venv is lock-faithful
(hadamard git v1.1.0 vs fe127's mystery 1.0.4.post1) — kernel-level hadamard
deltas vs fe127 = provenance, not B/F.
**Boot gate (bohr's REBUILD_NOTE §5, binding before any read):** boot log must
show both `... ACTIVE` WARNING lines ×16 ranks; no ACTIVE line = inert = the
trace is not a B/F trace. First read is the gate, from the driver.

## Pre-registered mechanism reads (rank0 unless noted)

| # | read | fe127 baseline | W1c prediction | band / note |
|---|---|---:|---|---|
| M1a | layout-builder nonzero (`nz_idx_fwd`+`nz_idx_bwd`) | 54,720 | **~720** (first-miss builds: 16 mb × 45/build; replay hits the same carrier cache) | ±20%; Aug-9 lineage: 26,520→~340 at d4/78L/CP16 |
| M1b | bare-bwd compaction nonzero (`nz_bare_bwd`) | 608 | **608 — UNCHANGED** (A-v3's class, not armed in W1c) | exact; survival expected, not a defect |
| M1c | total `aten::nonzero` | 55,328 | **~1,330** | ±20% |
| M2a | nonzero host CPU | 33.97 s | **~27–28 s** (bare-bwd class survives; builder 6.98 s → ~0.1 s) | the 27 s is the absorbed class — see M5 note |
| M2b | RoPE heavy `aten::to` >1 ms | 1,630 / 37.0 s | **~10–20 calls / <0.5 s** (F cache; Aug-9: 587→4) | order-of-magnitude read |
| M3 | **total pure idle, rank0** | 20.23 s | **13.2–15.5 s** — B+F measured coverage 7.06 s × conversion 0.7–1.0; includes ~1–2.5 s micro-tax launch relief | **>17 s = leak-causality takes a hit** (pairs with the perf refutation band <+1.5%) |
| M4 | perf (box/bench-side, not my trace) | 984 tok/s/GPU | my on-record band: **+4–7%, central +5%; refute <+1.5%** (IDLE_RESIDUAL_MAP §4) | gauss +3–6% is the sibling band |
| M5 | H vs 2L′ (rebalance reopen, PP_REBALANCE_ARM.md §5) | H ≈ 2.0, 2L = 3.6 | **no change** (B/F is host-side; L untouched) → arm stays dead | reopen only if measured otherwise |
| M6 | rank8 (first fixed-wheel rank8): builder collapse + stage-1 idle | l3 r8: 57,632 nz-idx, 12.74 s idle, 2.96 s builder leak | builders → ~720+640 bare; idle −2 to −3 s | also re-reads the stage asymmetry on the fixed wheel |
| M7 | seam-region idle (null control) | 1.93 s ≥1 ms | **unchanged ±0.3 s** — B/F doesn't touch the seam | if it moves, something's off — investigate |

Method: identical to the fe127 decomposition (gap-union + containment sweep +
union-trick coverage; stage_s1.py machinery). New traces served on :9006/:9007;
baselines stay on :9002/:9004/:9005.

## Results (2026-08-14 ~05:4x CDT — both pairs landed, read, verified)

Traces (sha256 logged to `pp2cp8ep8/w1c_traces.sha256`): off-arm
`w1c_b1_off_d16_rank{0,8}` (2.81/3.01 GB), on-arm `w1c_b2_on_d16_rank{0,8}`
(1.32/1.43 GB — **47% of off-arm bytes: the host-sync removal is visible in
raw event count**). Served: off :9006/:9007, on :9008/:9009 (fleet note).
Boot gate: ACTIVE lines confirmed by the driver before pull (kolmogorov).

### Validity check (off-arm vs fe127 census) — PASS, exact

| metric | fe127 | w1c off-arm r0 | verdict |
|---|---:|---:|---|
| total nonzero | 55,328 | 55,328 | **exact** |
| builder / bare-bwd split | 54,720 / 608 | 54,720 / 608 | **exact** |
| RoPE heavy `aten::to` | 1,630 / 37.0 s | 1,527 / 35.3 s | within drift |
| kernels | 971,927 | 971,761 | exact-ish |
| span | 133.35 s | 130.72 s | −2% (boot drift; inside the known control spread) |

**M1's ~1,330-survivors expectation is pinned to this box/wheel.** One
material baseline difference found (see M3): tonight's off-arm pure idle is
**12.00 s**, not fe127's 20.23 s — the big-gap host-stall leak is smaller
tonight (nz_idx leak 2.52 s vs 6.20 s) on identical call counts. Boot-to-boot
leak-regime variance is real; the same-night on/off pair is the clean
substrate (which is why the validity check was run first).

### Main event (on-arm vs off-arm, same box/venv/night)

| read | off-arm r0 | on-arm r0 | pre-registered | verdict |
|---|---:|---:|---|---|
| M1a builder nonzero | 54,720 | **720** (fwd first-misses; replay side = **0** — carrier cache hit) | ~720 ±20% | **EXACT** |
| M1b bare-bwd nonzero | 608 | **608** | 608 unchanged | **EXACT** (A-v3 class survives as registered) |
| M1c total nonzero | 55,328 | **1,328** | ~1,330 | **EXACT** |
| M2 RoPE heavy aten::to | 1,527 / 35.3 s | **40 / 1.07 s** | ~10–20 / <0.5 s | host collapse ✓ (calls modestly above guess) |
| M3 pure idle | 12.00 s | **4.07 s (−7.9 s, −66%)** | 13.2→9.5–10.5 s re-anchored | **exceeds** — see below |
| M7 seam idle (null control) | 1.80 s | **2.66 s** | unchanged ±0.3 s | **outside band — investigated, see below** |
| (consistency) SendRecv total | 47.13 s | 45.91 s | flat (F6) | ✓ |
| (consistency) kernel launches | 971,761 | **432,793 (−539k, −55%)** | −150k class (Aug-9, d4) | ✓ direction, larger at d16 |
| (proxy) traced step span | 130.72 s | **119.85 s (−8.3%)** | perf band is bench-side | ≈ +9.1% tput-equivalent |

**M3 mechanism closure:** the idle collapse (−7.9 s) is ~3× the B+F static
leak coverage measured on the same off-arm (2.71 s). The compounding term is
the **micro-gap tax collapse: 9.51 s → 0.89 s** as −539k launches/step
decompress the launch pipeline — the F6 launch-decompression mechanism,
measured directly on the fixed wheel at d16. Leak-causality is CONFIRMED and
conversion is >1, not <1. (This also retro-explains the fe127-vs-off-arm
baseline gap: leak size is regime-sensitive; launch-count removal is the
robust term.)

**M7 seam finding:** the seam not only persisted but grew (+0.87 s): the same
untraced-python hole class (1,247 ms + 828 ms on-arm vs 1,224 + 326 off).
**Framing correction (bayes, ratified):** under the favored hypothesis H1 the
sentence "B/F grew the seam" is wrong — the right sentence is **"B/F removed
the launch backlog that hid a ~constant 2.6–2.7 s seam python"**; the +0.87 s
is exposure, not new work. H2 (B/F added cache-maintenance work) remains open
until the R2 probe's function-level read (H2 refuted if cache functions <2%
of seam samples). Until the probe runs, all seam-growth language in the
campaign docs reads as *exposure*, per H1-favored.
**The on-arm residual idle (4.07 s) is now seam-dominated: 2.66 s seam +
0.53 s in-step ≥1 ms + 0.89 s micro.** R2 (seam python — py-spy probe, then
targeted fix) is now the top residual idle lever: 2.66 s/step = 2.2% of the
new 119.85 s step.

**M5:** B/F is host-side; L untouched (on-arm per-layer CF spans unchanged) →
H < 2L′ stands, **rebalance arm stays dead** ✓.

**M6 (rank8, first fixed-wheel rank8 read):** nz 58,304 → **1,424** (720
first-miss + 640 bare-bwd + 64 none-class); idle 11.13 → **1.98 s (−82%)**;
span −8.5% (same step as rank0's −8.3% ✓). Tonight's off-arm stage leaks were
near-SYMMETRIC (nz_idx leak 2.52 s r0 vs 2.72 s r8) — the l3 2.7× asymmetry
did NOT reproduce on the fixed wheel → honestly noted: the asymmetry was
regime-dependent (old wheel / convoy), and the no-haircut verdict holds
trivially (both stages' leaks die together under B/F).

**M4 (perf):** trace-span proxy says ≈ +9% throughput-equivalent — at the
ceiling of my filed band (+4–7% central +5%, ceiling +9–11%). The bench A/B
(lovelace) is the throughput authority; single traced steps can over-read.
**Mechanism verification is complete regardless of where the bench lands
inside +4–11%:** the collapse is exact-count causal, on/off same box.

### Residual-map consequence (for bayes's lever queue)

The 20.8 s pure-idle problem is now a **4.1 s problem**, and its center of
gravity moved: seam python 2.66 s (R2 — py-spy probe is now the highest-EV
idle diagnostic), micro-tax 0.89 s (R1 mostly dead as a lever — what remains
is irreducible-ish launch latency), in-step misc 0.53 s. The a2a/cat host
blocks (R3) are below measurement in the on-arm ≥1 ms set. The exposed-
SendRecv prize (lever 1, contract shim) is untouched by B/F and stands.
