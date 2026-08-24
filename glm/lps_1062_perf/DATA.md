# DATA.md — Mac-only large artifacts (gitignored, not uploaded)

Decision (Jack, 2026-08-10): large traces / memory snapshots stay **local on
Jack's Mac**, gitignored — no upload to CPFS/object storage. This file is the
manifest; if the Mac copy is lost, the numbers derived from them remain in the
run docs but the raw evidence is gone.

**Prune policy (ratified 2026-08-24):** a concluded run's traces are deleted
once its verdicts are banked in run docs and the lever has shipped or been
refuted; the current-state reference is re-recorded on merged main instead of
hoarding historical traces. Active-bug evidence stays until the fix verifies.

## In-repo (gitignored by pattern: `*.pt.trace.json[.gz]`, `memory.rank*.pickle[.gz]`, `glm/lps_1062_perf/**/*.log[.gz]`)

| path | size | status | what |
|---|---:|---|---|
| `runs/overnight_20260822_262k_pr1070/traces/R1-main262k-d2_rank0.pt.trace.json.gz` | 78M | keep | pre-merge main @ 9b039d6b, 262k d2 (evidence for the open 312-vs-417 clean-main regression flag) |
| `runs/overnight_20260822_262k_pr1070/traces/R2-pr1070-262k-d2_rank0.pt.trace.json.gz` | 28M | keep | PR #1070 @ d8b9648f, 262k d2 — tree-identical to merged main's 1070 content |
| `runs/overnight_20260822_262k_pr1070/traces/R3-pr1070-262k-d4_rank0.pt.trace.json.gz` | 56M | keep | PR #1070, 262k d4 |
| `runs/overnight_20260822_262k_pr1070/traces/R{1,2,3}_memory.rank0.pickle.gz` | 22M | keep | rank-0 CUDA memory snapshots for the above |
| `runs/fullmodel_offload_20260823/traces/b300-1-5abzeeir-0002_27461.*.pt.trace.json.gz` | 31M | **active** | offload_32k arm — PP2 reload-miss bug evidence |
| `runs/fullmodel_offload_20260823/traces/baseline/b300-1-5abzeeir-0002_24007.*.pt.trace.json.gz` | 30M | **active** | baseline_32k arm (the "before" pair) |
| `runs/fullmodel_offload_20260823/traces/diag/*.pt.trace.json.gz` (2 files) | 63M | **active** | diag captures incl. stage-1 attempt. **Delete all four fullmodel traces once the offload fix verifies.** |
| `pp2cp8ep8/logs/box_a/unfused_pp2_watchdog_death_20260813.log.gz` | 1.9M | keep | watchdog-death forensics, concluded campaign |
| `pp2cp8ep8/kernel_src_snapshot/{kernel_src_snapshot,cutlass_dsl_src_snapshot}.tar.gz` | 72M | keep | DSA cuDNN-binding + CUTLASS DSL source snapshots (see MANIFEST.md there); extracted trees deleted 2026-08-24, restore via untar |

## Deleted 2026-08-24 (conclusions banked, raw evidence dropped)

| what | was | where the conclusions live |
|---|---:|---|
| `runs/overnight_20260809_dispatcher_hostsync/traces/` (3 traces) | 3.2G | `runs/overnight_20260809_dispatcher_hostsync/ATTRIBUTION.md`, root NOTEBOOK.md; B/F shipped + P4-validated |
| `runs/debug_proxy_20260821/results/overlap_trace_20260822/traces/` (10 traces) | 84M | `.../overlap_trace_20260822/MISSION_WITHIN2PCT.md`, `PATCHED_RESULTS.md` (E1–E7 verdicts); analysis CSVs/JSONs kept in `raw/` + `stat_ab/` |
| `runs/fullmodel_offload_20260823/traces/` uncompressed `.json` (3 files) | 1.2G | exact duplicates — md5-verified identical to the retained `.gz` copies |

## Outside the repo: `~/perf_profiles/lps-1062/` (~36 GB)

| path | size | status | what |
|---|---:|---|---|
| `pp2cp8ep8/` | 27G | **concluded — prunable** | activation-placement workstream traces; superseded by full-model-scale runs. Largest single prune candidate |
| `round3/` | 5.0G | concluded — prunable | round-3 arm trace bundles (C′, W1, W3, anchor); verdicts in `runs/overnight_20260810_round3/verdicts/` |
| `incoming/` | 1.6G | concluded — prunable | A-v3 final leg + W3-v3 canary traces/memsnaps; adjudicated in NOTEBOOK |
| `opt-night/exp05d.pt.trace.json` | 1.0G | concluded | canonical post-optimization baseline trace (Aug-7), md5 32d30aa15c467db6f19caf2fedebef80 |
| `glm52-b300-s256k/` | 1.0G | concluded | Aug-6 baseline rank-0 trace + memory snapshots, md5s in git history |
| `kimi27/` | 461M | concluded | Kimi-K2.7-Code rank-0 trace + 16-rank memory snapshots; comparison banked in `runs/kimi_k27_20260815/COMPARE.md` |
| `w4_parity/`, `debug-proxy/`, `s1_soak_mem/` | ~40M | concluded | small evidence dirs |

Small text evidence that USED to live only in `~/perf_profiles/` was imported
into the repo on 2026-08-10: round-3 verdicts/adjudications →
`runs/overnight_20260810_round3/verdicts/`; round-3 bench JSONs, arm logs, FAIL
logs → `runs/overnight_20260810_round3/results/`; Aug-6 baseline report/driver
→ `runs/overnight_20260807_baseline_shipconfig/`. Originals left in place.

## Planned

- **Fresh 131k trace on merged main** (post-#1070, `71a9f3b75`): 131k×d4,
  PP2/CP8/EP8, LoRA r32, clean env, campaign driver protocol (warmup → traced →
  ≥2 controls), rank-0 kineto + memory pickle + nvidia-smi pollers. Lands in a
  new run dir and becomes the current-state 131k reference; pairs with R3 for
  the same-tree 131k↔262k comparison. Known blind spot: rank-0-only profiling
  on main (no `BT_PROFILE_RANKS`) — PP2 stage 1 invisible.
