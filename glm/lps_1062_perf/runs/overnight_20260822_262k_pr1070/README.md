# overnight_20260822_262k_pr1070 — does PR #1070 work at 262k, and 262k on tip of main

Jack's ask (2026-08-22 night): PR #1070 (GLM-5.2 PP2 + THD-CP microbatch
pipelining, proven at 131k) — does it work at 262k? Profile 262k with the
campaign driver (`tools/profile_driver_new.py`, same protocol as the 131k
rows) on BOTH tips, save profiles here, then compare traces 131k vs 262k
and rank the bottlenecks for each.

## Setup
- Box: devbox wgm8row (`tj-wgm8row`), 2×8 B300 (275,040 MiB/GPU), ali plane,
  fresh devbox-up provision; server venv built from main's uv.lock
  (torch 2.11.0+cu130, cuDNN 9.19.0) on both nodes.
- Refs: main tip = `9b039d6b`; PR #1070 head = `d8b9648f`
  (**rebased on exactly 9b039d6b** — clean one-variable comparison).
- Model: zai-org/GLM-5.2-FP8, LoRA r32/α32, attention_backend=flash,
  synthetic tokens (rng 0xB300), weight_sync disabled. Clean env both runs
  (no TF32-head knob, no NCCL ship env — those are separate deferred PRs).
- Driver operating point: d2 × 262,144 = 524,288 tok/step (matches the
  A-anchor-262k row of 2026-08-09 and the 131k-d4 standard). 2 control
  windows. Headline = control-window tok/s/GPU; mfu3x LoRA-corrected.
- Kineto + memory profiles are rank-0-only on this tree (ProfilingConfig
  rank_set={0}, no env knob on main) → no rank-8/stage-1 trace at 262k.
  Per-GPU peak memory covered by nvidia-smi pollers on both nodes.

## Runs
- R1 `R1-main262k-d2`: main @ 9b039d6b, TP1/PP1/EP16/CP16 (the layout of
  the existing 256k golden row / Aug-9 anchor: 630 tok/s/GPU, peak 263.7 GiB).
- R2 `R2-pr1070-262k-d2`: PR head @ d8b9648f, TP1/PP2/EP8/CP8 (the PR's
  131k golden layout scaled to 262k).
- R3 (conditional, pre-registered): R2 config at d4 (1.05M tok/step) only if
  R2 passes with peak ≤ 240 GiB.

Pre-registered bars in PREREG.md (copied verbatim from the box before any
data landed).

## Artifacts
- `*.json` — driver outputs (windows + aggregates), one per run label.
- `*.runlog` — driver stdout.
- `traces/` — rank-0 kineto traces (`<label>_rank0.pt.trace.json`) and
  rank-0 CUDA memory snapshots.
- `mem_max_<host>.txt` — nvidia-smi peak-used per GPU per node.
- `ANALYSIS.md` — 131k↔262k trace comparison + ranked bottlenecks.
