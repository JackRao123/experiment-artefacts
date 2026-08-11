# LPS-1062 — GLM-5.2 B300 throughput optimization

Multi-night optimization campaign: GLM-5.2-FP8, golden config TP1/PP1/EP16/CP16,
256k, LoRA r32, on 2×8 B300 (ali RoCE). **416 → 715 tok/s/GPU @131k** across the
campaign; the Aug-7 ship config (+51%) and the round-3 lever stack are documented
per-run below.

**Path convention:** all relative paths in these docs are relative to this
folder (`glm/lps_1062_perf/`) unless absolute. On-box paths (`/root/.cache/...`,
`lps1062/`, `lps1062_bench/`) refer to the flat kit staged on devboxes, not this
repo. Large traces/memory snapshots are Mac-only and gitignored — see `DATA.md`.

## Layout

- `NOTEBOOK.md` — chronological lab notebook across all runs (the spine)
- `REPORT.md` — headline report: the Aug-7 ship config (+51%, memory-flat)
- `DATA.md` — manifest of Mac-only large artifacts (traces, memory snapshots)
- `tools/` — canonical reusable bench kit: `bench_driver2c.py`, `mfu.py`
  (LoRA-corrected), `run_bench2c.sh`, `poll_gpu_mem.sh`, `fold_mem.py`
- `runs/` — one folder per run, self-contained (configs, results, patches,
  docs, and the as-used driver copies). New open runs: `runs/overnight_YYYYMMDD_<slug>/`

## Runs

| run | what | headline |
|---|---|---|
| `runs/overnight_20260807_baseline_shipconfig/` | baseline profile + optimization night 1 (exp00–06) | **+51% ship config**: TF32 LM head + NCCL QP/channel env; REPORT.md at root |
| `runs/overnight_20260808_context_sweep/` | B300 context-length sweep (EP8CP8 1-node, EP16CP16 2-node) | sizing data for 16k–262k |
| `runs/overnight_20260809_mfu_sweep/` | parallelism/MFU sweep of the customer regime (A/B/C boxes) | anchors: 645 @131k-d4, 734 @16k-d32; LoRA-corrected MFU convention landed here |
| `runs/overnight_20260809_dispatcher_hostsync/` | MoE dispatcher host-sync elimination (FIX A/B/F + C′ + A-v3) | 26.7k `nonzero`/step class eliminated; B/F shipped into golden config |
| `runs/overnight_20260810_round3/` | overlap levers W1/W2/W3 + F2 phantom partitions | W1 + C′ shipped-as-v1; W2 hung (imbalance deadlock, STOPPED); W3 v2/v3 refuted; F2 validated. `overlap/`, `f2/`, `verdicts/`, `results/` |

## Headline (2026-08-07, devbox q480z53 → round-3 boxes 318g61w/wxlgv5w)

- Baseline: **446 tok/s/GPU** (73.5 s/step), MFU ≈ 4.7% LoRA-corrected — see
  `runs/overnight_20260807_baseline_shipconfig/glm52-b300-s256k/REPORT.md`
- Aug-7 ship config: **629 (steady ~660)**, memory flat — `REPORT.md`
- Round-3 frontier: **715 tok/s/GPU @131k×d4, 726 @16k×d32** —
  `runs/overnight_20260810_round3/overlap/SCORECARD_MORNING.md`
- MFU convention is LoRA-corrected throughout:
  `runs/overnight_20260809_mfu_sweep/mfu_lora_correction.md`
