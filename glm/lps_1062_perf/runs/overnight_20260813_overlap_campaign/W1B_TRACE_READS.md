# W1b TRACE READS — pre-registered (kepler→jacobi, 2026-08-14 ~00:xx CDT)

**Filed BEFORE the W1b d16 traces land.** Window: block+K21 partial recompute
(`recompute: {granularity: full, method: block, num_layers: 21}` — first 21
layers/stage checkpointed, remaining 17/38 (stage 0) and 19/40 (stage 1)
EAGER), d16 pair, `BT_PROFILE_RANKS=0,8`, box wprm693, driven by grothendieck.
My lane (bayes-assigned): pull Mac-side immediately, read, report.

**Baselines:** fe127_d16_rank0 (fixed wheel, number-of-record trace; my
decomposition + residual map numbers) and l3_d16 rank0/rank8 (old wheel).
**Provenance caveat (bayes):** tonight's venv is lock-faithful — hadamard git
**v1.1.0**, NOT the mystery 1.0.4.post1 the fe127 venv ran. Kernel-level
hadamard deltas vs fe127 = provenance difference, NOT blockK21. Do not
attribute them to the arm.

**NOT W1b's read:** the full nonzero collapse (55,328 → ~700) belongs to
**W1c (B/F)**. W1b has no B/F — builder calls drop only via skipped replays.

## Pre-registered reads

| # | read | baseline (fe127 r0) | W1b prediction | band |
|---|---|---:|---|---|
| R1a | rank0 replay-side layout-builder nonzero (`nz_idx_bwd`) | 27,360 | **15,120** (21/38 layers replayed) | ±10% |
| R1b | rank8 replay-side layout-builder nonzero | 28,800 (l3) | **15,120** (21/40) | ±10% |
| R1c | rank0 total nonzero | 55,328 | **~43,100** (−22%) | ±10% |
| R1d | CF/CFB count per stage (block structure probe) | 608 r0 / 640 r8 | **336 = 21×16 both stages** (per-layer checkpointing of the first 21) | exact |
| R2 | exposed SendRecv (wall covered by SendRecv with no non-NCCL overlap) | 34.3 s (erdos) | **28.8–29.0 s** — the replay a2a (11.7 s/step, jacobi's L3 decomposition) dies on eager layers: −45–47% of 11.7 ≈ −5.3–5.5 s | ±1.5 s |
| R3 | non-NCCL compute wall, rank0 | 70.52 s | **−10 to −18 s** (17 eager layers skip the replay-fwd; recompute ≈ 32% of per-layer fwd+bwd) | — |
| R4 | **rebalance reopen condition** (PP_REBALANCE_ARM.md §5) | H ≈ 2.0 s, 2L = 3.6 s | H ≈ 2.0 (unchanged), 2L′ ≈ 3.0–3.1 (L′ ≈ 1.54, −14.5%: eager layers cost ≈ 68% of checkpointed) → **H < 2L′, arm stays dead** | reopen iff measured H > 2L′ |
| R5 | total pure idle, rank0 | 20.23 s | **~18.5–19 s** (replay-builder leak 3.25 s scales ×21/38 → −1.45 s; rest of the idle budget is W1c territory) | ±0.7 s |

Method: identical to IDLE_WINDOW_DECOMPOSITION/IDLE_RESIDUAL_MAP (gap-union +
containment sweep + union-trick coverage; scripts staged in
/var/folders/.../opencode/stage_s1.py). New traces get fresh trace_processor
instances on :9006/:9007 (:9002/:9004/:9005 stay up for baselines). Exposed
SendRecv = cover(nonNCCL ∪ SendRecv) − cover(nonNCCL) — window-function union,
no enclosure joins (the :9001/:9005 lesson).

Interface notes: jacobi/pauli's fit+perf bars (peak ≤255 GiB, predicted
220–240; perf +4–8% d4 / +6–10% d16 vs the fixed-wheel anchors) are
box/bench-side reads — grothendieck/bayes own them; my R3/R4 numbers feed the
dial's K calibration and the rebalance reopen condition only. Pull mechanics:
`scp -C training-job-<job>-{0,1}.ssh.baseten.co:/tmp/checkpoints/profiles/
torch_trace/*.pt.trace.json` (rank0 = node 0, rank8 = node 1), sha256-verify
on landing, then serve + read.

## Results

**VOID — NO DATA (2026-08-14, bayes):** W1b died at fit; no windows ran, no
traces exist. Reads R1–R5 roll off unexercised — recorded as VOID-no-data,
not failed. The R4 reopen condition (H > 2L′) transfers to W1c's traces where
applicable (B/F doesn't move L, so no reopen expected there either — see
W1C_TRACE_READS.md M5).
