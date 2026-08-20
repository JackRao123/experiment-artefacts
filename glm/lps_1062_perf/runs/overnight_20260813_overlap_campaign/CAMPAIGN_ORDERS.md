# CAMPAIGN ORDERS — LPS-1062 overlap campaign (2026-08-13, erdos → fermi)

**Mission:** optimize GLM-5.2 PP2/CP8/EP8 @131k on 2×8 B300 beyond the current
number of record. Duration: 48 h. Orchestrator: **fermi** (successor to erdos).
Subordinates: hausdorff, kepler, jacobi, bohr, doppler, lebesgue (Kimi K3 1M);
spawn fresh subagents as needed.

**Reporting:** every 30 min, send a status update to session `erdos` (Jack is
watching that tab) via `~/.agents/scripts/send-message.sh erdos "..."`. Each
update must include: progress in the last 30 min AND the best TPS/GPU + MFU
(mfu3x, LoRA-corrected convention) achieved so far. First update due 30 min
after ack.

## Number of record (fixed wheel, 2026-08-13)

| rung | tok/s/GPU | step | mfu3x | hfu | peak |
|---|---:|---:|---:|---:|---:|
| d4 (524k tok) | ~878–886 | ~37s | ~8% | — | 143 GiB |
| **d16 (2M tok)** | **984** | 133.2 s | **9.0%** | 13.0% | 162 GiB |

Anchors: golden EP16/CP16 = 645; best-ever before PP2 = 691. Stale-wheel
numbers (1052 @d16) are corrupt-wheel and NOT a ship path. Box w56lorq is
STOPPED; all evidence swept and sha256-verified Mac-side.

## Fresh bottleneck map (erdos, 2026-08-13 ~15:45 CDT)

Exact interval-union sweep over the fixed-wheel d16 trace
(`~/perf_profiles/lps-1062/pp2cp8ep8/fe127_d16_rank0.pt.trace.json`, 971,927
kernels, 133.4 s step, rank 0; served by trace_processor pid 47309 on :9001 —
kill when done):

| class | time | % wall | note |
|---|---:|---:|---|
| SendRecv EXPOSED (no compute overlap) | 34.3 s | 25.9% | total SendRecv 37.1 s; only 2.8 s hidden. p50 4.45 ms, p90 12.4 ms; 10 calls >100 ms = 8.8 s; max 5.04 s (boundary/imbalance waits) |
| fully idle (no kernel) | 20.8 s | 15.7% | ideal PP2 bubble at d16 ≈ 6% ⇒ ~10 pts of attackable slack (step boundary, optimizer, telemetry) |
| other NCCL exposed (AllGather 5.6 + RS 1.9) | 7.5 s | 5.7% | ZERO overlap with compute |
| compute only | 67.1 s | 50.7% | DSA bwd 13.1 s (608×21.5 ms), GEMMs ~13 s, sparse fwd 4.5 s, indexer recompute 4.3 s |
| host-side (absorbed) | — | — | nonzero 34 s/55k, streamSync 32 s/65k, memcpyAsync 37.7 s/82k — B/F caches never landed; runahead absorbs at d16 |

**Non-compute = 62.5 s of 133.4 s (47%).** Perfect-overlap ceiling ≈ 70 s step
≈ ~1900 tok/s/GPU. Realistic winnable band: 20–25 s/step (15–19%).

## Lever queue (EV-ranked; attack in order, pre-register bars per house rules)

1. **Overlap program** (attacks the 34.3 s exposed a2a; prize 15–19%):
   executor contract shim (`pp2cp8ep8/results/EXECUTOR_CONTRACT_SCOPING.md` —
   scoped CHEAP, 0.5–2 d, wrapper-level) + memory leg: per-layer recompute
   dial (upstream, ~1 d prototype, `results/UPSTREAM_PROPOSAL.md`) OR offload
   valve (engages at 32k, −20–26 GiB, −30% step cost, expert_fc1 coverage
   gap). E1 re-measure on fixed wheel: wall persists (259.5 GiB warmup OOM) —
   no wheel-assisted shortcut.
2. **Idle-window attack** (20.8 s): separate true bubble from step-boundary
   slack; check optimizer/telemetry/save cadence at the step seam.
3. **B/F + C′ + A-v3 host-sync stack** (never landed; Aug-9 +11-13% was
   convoy-regime, at d16 mostly absorbed — re-measure; worth more at smaller
   d). PRs: Megatron-LM #27 (C′), #29 draft (A-v3). B/F needs REBUILD (no
   branch exists; patches in `runs/overnight_20260809_dispatcher_hostsync/`).
4. **W1 + option-6** (probs a2a 2nd communicator + bwd reorder; model
   −3.4…−5.1 s on prod fabric). PR #28 carries the multi-EP-group `new_group`
   fix — **W1 stays OFF on >1-EP-group topologies until validated there**;
   check EP-group count of PP2/CP8/EP8 before arming. option-6 patch 6fbf3acc
   is built/CPU-proven, needs a canary slot, requires W1.
5. **CP AllGather exposure** (5.6 s, zero overlap): KV-gather overlap —
   bigger surgery, only after 1–4.
6. **NCCL env re-sweep** for the new topology (ship env was tuned on
   EP16/CP16 256k RoCE; EP8 is intra-node here — cheap A/B, low EV).

**Held closed / refuted (do not reopen without new evidence):** W2 chunked
pipeline (timed-arm hang, imbalance deadlock); W3 lookahead recompute (v2/v3
refuted; v4 needs A-v3 prerequisite); DeepEP/flex standalone (−12.6% at d4,
mechanism decomposed); L1 wgrad overlap (correct, free, perf-unproven at the
3–4% noise floor — default-off).

## Operational rules (standing, from the whole campaign)

- **WHEEL:** any fresh box builds the STALE cudnn-frontend wheel
  (1.26.0+dsatopk1 — corrupt, nondeterministic; build-system bug unfixed).
  First act on any new box: bump venv to the pinned 1.27.0.dev20260803 per
  the fix-campaign procedure (REPORT_MORNING_0813.md §3 + NOTEBOOK fix
  entries). No measurement counts unless fixed-wheel.
- **Trainer lifecycle:** use ONLY `/root/.cache/user_artifacts/.devbox_up/`
  start/wait/stop scripts. **Waiting = `wait_trainer_health.sh`** (Jack's
  explicit order — all subordinates). No ad-hoc launchers/pkill loops.
- **Measurement:** `lps_1062_perf/tools/` bench kit (bench_driver2c.py,
  run_bench2c.sh, poll_gpu_mem.sh, mfu.py LoRA-corrected). Canaries
  12.2–12.4 band + grad-norm; parity = fresh-boot vs fresh-boot, fixed 9-doc
  dataset, loss ≤1e-6 rel / per-token ≤1e-3 (statistical protocol where the
  noise floor exceeds bars). Pre-register bars before running.
- **F1 landmine:** async save hangs under CP>1 — set `BT_SAVE_STATE_SYNC=1`.
- **Traces:** `BT_PROFILE_RANKS=0,8`; pull Mac-side immediately (box can die).
- **Docs:** NOTEBOOK.md is the spine — append chronologically, honest
  chronology, mark overturned framings in place. Evidence to
  `experiment_artefacts/glm/lps_1062_perf/` (this run folder).
- **Merge queue (parked for Jack):** `pp2cp8ep8/pr_shaping/MORNING_MERGE_QUEUE.md`
  (9 items; PR 987 first, 995 before M=N). Campaign PRs stack on top.

## Phase plan (guidance, revise on data)

- **P0 (0–1 h):** box q8eg0gq up (provisioning started by erdos ~15:0x CDT,
  log /tmp/devbox_up_campaign.log), wheel bump, re-anchor d4/d16, canary.
- **P1 (1–12 h):** contract shim implementation + unit tests (attacks lever 1);
  in parallel B/F rebuild + A-v3/C′ landing prep (lever 3).
- **P2 (12–24 h):** memory leg (dial or offload valve + expert_fc1 coverage);
  first real overlap A/B at 131k on the fixed wheel.
- **P3 (24–36 h):** W1+option-6 canary; idle-window attack; NCCL re-sweep.
- **P4 (36–48 h):** stacked soak, parity gates, final report + PR package.

— erdos, 2026-08-13 ~16:0x CDT
