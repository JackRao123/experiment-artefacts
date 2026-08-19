# DATA.md — Mac-only large artifacts (gitignored, not uploaded)

Decision (Jack, 2026-08-10): large traces / memory snapshots stay **local on
Jack's Mac**, gitignored — no upload to CPFS/object storage. This file is the
manifest; if the Mac copy is lost, the numbers derived from them remain in the
run docs but the raw evidence is gone.

## In-repo (gitignored by explicit path in experiment_artefacts/.gitignore)

`runs/overnight_20260809_dispatcher_hostsync/traces/`:

| file | size | md5 |
|---|---:|---|
| patched-ABF-4mb131k.pt.trace.json | 617M | 9045359766bf08af3089c84f947cf4ba |
| unpatched-4mb-steady_qr4ggv3-gatesoff-step3.pt.trace.json | 1.9G | e5bbe1770d0dee87d5a6501f10904a32 |
| gated-v2-4mb-steady_qr4ggv3-step8.pt.trace.json | 676M | 4c8f2038681dc0938bcf026776a72850 |

## Outside the repo: `~/perf_profiles/lps-1062/` (~8.5 GB)

| path | size | what | md5 (files only) |
|---|---:|---|---|
| `kimi27/traces/b300-1-78ref8i2-0016_4347.1786820197615552255.pt.trace.json` | 248M | Kimi-K2.7-Code rank-0 kineto trace, 131k×d4, golden B300 TP8/EP16/CP1, main @ 29b59564 (sha256 dd62e603…) | — |
| `kimi27/traces/mem/memory.rank{0..15}.pickle` | — | Kimi memory snapshots (16 ranks) | — |
| `kimi27/kimi27-131k-d4.json` | 4K | Kimi bench JSON (sha256 7b59da3a…): 641 tok/s/GPU, mfu3x 15.9%, peak 170 GiB | — |
| `opt-night/exp05d.pt.trace.json` | 1.0G | canonical post-optimization baseline trace (Aug-7, cited across NOTEBOOK/REPORT/ATTRIBUTION) | 32d30aa15c467db6f19caf2fedebef80 |
| `glm52-b300-s256k/node0/*.pt.trace.json` | 1.0G | Aug-6 baseline rank-0 kineto trace | 0358280caf35a307108955e3ba7cca13 |
| `glm52-b300-s256k/node0/memory.rank0.pickle` | — | baseline memory snapshot rank 0 | 6488c4fe43e8e9b23021fa82914590c9 |
| `glm52-b300-s256k/node1/memory.rank8.pickle` | — | baseline memory snapshot rank 8 | 81e0f702c53793bd16caca07267c2abd |
| `round3/arm-cprime-318g61w/` | 1.2G | C′ timed-arm trace bundle | — |
| `round3/arm2-w1-318g61w/` | 1.2G | W1 timed-arm trace bundle | — |
| `round3/arm-w3-318g61w/` | 1.4G | W3 canary trace bundle + mem_snapshot | — |
| `round3/anchor-318g61w/` | 1.2G | round-3 anchor trace bundle | — |
| `incoming/av3_final_rank0.pt.trace.json` | 676M | A-v3 final leg trace | 5c46ee7dabb340a9192f41e5357b82b7 |
| `incoming/w3v3_rank0.pt.trace.json` | 633M | W3-v3 canary trace | 8acd46a6175d7c7b0b3139756aee847f |
| `incoming/w3v3_memsnap/` | 202M | W3-v3 memory snapshots (16 ranks) | — |
| `incoming/av3-soak-trainer_srun_persistent.log` | 14M | A-v3 persistent soak log (kept local: size) | — |

Small text evidence that USED to live only in `~/perf_profiles/` was imported
into the repo on 2026-08-10: round-3 verdicts/adjudications →
`runs/overnight_20260810_round3/verdicts/`; round-3 bench JSONs, arm logs, FAIL
logs → `runs/overnight_20260810_round3/results/`; Aug-6 baseline report/driver
→ `runs/overnight_20260807_baseline_shipconfig/`. Originals left in place.
