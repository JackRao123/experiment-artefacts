# LPS-1062 — GLM-5.2 B300 throughput optimization

Multi-night optimization campaign: GLM-5.2-FP8 on 2×8 B300 (ali RoCE), LoRA r32.
**416 → 715 tok/s/GPU @131k** across the early campaign (Aug-7 ship config +51%,
round-3 lever stack), then **1103 tok/s/GPU @d16/131k** from the Aug-13/14
overlap campaign. PR #1070 (PP2 + THD-CP microbatch pipelining) merged
2026-08-24 (`71a9f3b75`); current work: activation offload at full-model scale.

## Current progress checkpoint

**2026-08-25, main `95fe75e9`: 756 tok/s/GPU at d4 and 847 tok/s/GPU at d8
for GLM-5.2 131k TP1/PP2/EP8/CP8 on 2x8 B300.** Full provenance, control
windows, MFU/HFU, memory, and config:
[`runs/progress_checkpoint_20260825_main_95fe75e9_d4_d8/`](runs/progress_checkpoint_20260825_main_95fe75e9_d4_d8/).

**Path convention:** all relative paths in these docs are relative to this
folder (`glm/lps_1062_perf/`) unless absolute. On-box paths (`/root/.cache/...`,
`lps1062/`, `lps1062_bench/`) refer to the flat kit staged on devboxes, not this
repo. Large traces/memory snapshots are Mac-only and gitignored — see `DATA.md`.

## Layout

- `NOTEBOOK.md` — chronological lab notebook across all runs (the spine)
- `REPORT.md` — headline report: the Aug-7 ship config (+51%, memory-flat)
- `B300_HOST_OFFLOAD_BANDWIDTH.md` — measured 8xB300 host-memory topology,
  NUMA-local/remote bandwidth, and activation-offload guidance
- `DATA.md` — manifest of Mac-only large artifacts (traces, memory snapshots)
- `tools/` — canonical reusable bench kit: `bench_driver2c.py`, `mfu.py`
  (LoRA-corrected), `run_bench2c.sh`, `poll_gpu_mem.sh`, `fold_mem.py`, and
  `cuda_host_bw.cu` for pinned host/device NUMA bandwidth
- `pp2cp8ep8/` — the PP2/CP8/EP8 activation-placement workstream (rung reports,
  parity evidence, census runbooks; concluded at proxy scale, continues at
  full-model scale in `runs/fullmodel_offload_20260823/`)
- `glm-5.2-moe-layer/` — editable D2 architecture diagrams of one GLM-5.2
  sparse decoder layer (`.d2` sources + rendered svg/png/pdf)
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
| `runs/overnight_20260813_overlap_campaign/` | 48h overlap campaign (B/F validation, W1 legs, P4 soak, blockK) | **number of record 1103 tok/s/GPU @d16/131k**; ship stack (mission + B/F) VALIDATED. `CAMPAIGN_REPORT.md` is the final word |
| `runs/kimi_k27_20260815/` | GLM-5.2 vs Kimi-K2.7-Code, same box shape + driver | GLM ~10% faster in tok/s/GPU; Kimi extracts ~2.6× the MFU. `COMPARE.md` |
| `runs/overnight_20260822_262k_pr1070/` | does PR #1070 work at 262k + 131k↔262k trace comparison | **#1070 works at 262k: +23% (d2) / +67% (d4) vs main tip**; merged 2026-08-24. `ANALYSIS.md` is the deliverable |
| `runs/debug_proxy_20260821/` | single-B300 real-code GLM activation-placement proxy | 0D1M boots in ~70s; mission closed at −17.6% (forward host-sync × saturated-copy-engine collision = proven floor) |
| `runs/fullmodel_offload_20260823/` | full-model activation offload at PP2/CP8/EP8 (**active**) | offload −40% vs baseline at 32k/d4; root cause = PP2 backward-order reload-miss bug (12.8% uniform misses). Fix in flight — `HANDOFF_WEIL.md` |
| `runs/postmerge_20260824_131k/` | 131k/d4 PP2/EP8/CP8 on merged main (post-#1070) | **~330±40 tok/s/GPU, noisy** — 68% of the traced window is NCCL wait (stage-1-bound); compute healthy (DSA halves vs R3 per-call). The new current-state 131k reference |
| `runs/progress_checkpoint_20260825_main_95fe75e9_d4_d8/` | current main checkpoint, 131k d4+d8 on 2x8 B300 | **756 tok/s/GPU d4; 847 d8**; LoRA-corrected MFU 6.9% / 7.7%; d8 is +12.1% |
| `runs/cross_node_p2p_20260826/` | direct cross-node GPU P2P fabric ceiling on 2x8 B300 | **74.83 GB/s one pair; 515.59 GB/s eight pairs one way; 771.72 GB/s full-duplex combined** |

## Headline (2026-08-07, devbox q480z53 → round-3 boxes 318g61w/wxlgv5w)

- Baseline: **446 tok/s/GPU** (73.5 s/step), MFU ≈ 4.7% LoRA-corrected — see
  `runs/overnight_20260807_baseline_shipconfig/glm52-b300-s256k/REPORT.md`
- Aug-7 ship config: **629 (steady ~660)**, memory flat — `REPORT.md`
- Round-3 frontier: **715 tok/s/GPU @131k×d4, 726 @16k×d32** —
  `runs/overnight_20260810_round3/overlap/SCORECARD_MORNING.md`
- MFU convention is LoRA-corrected throughout:
  `runs/overnight_20260809_mfu_sweep/mfu_lora_correction.md`
