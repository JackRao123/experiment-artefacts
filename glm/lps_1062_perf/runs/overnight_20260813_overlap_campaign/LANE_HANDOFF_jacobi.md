# LANE HANDOFF — jacobi (ex-kepler), trace-analysis lane

**Date:** 2026-08-14 · **To:** turing + whichever successor picks up the lane
**Status at handoff:** all assigned work complete and filed. Nothing in flight.

## 0. Naming hazard (read this first)

The ~00:00 CDT session-restart wave renamed the fleet. **I am jacobi, formerly
kepler** — the trace-analysis lane. **The OLD jacobi (memory leg / recompute
dial, author of MEMORY_LEG_DECISION.md) is now pauli.** Reading the NOTEBOOK:
entries tagged "jacobi" BEFORE the restart are pauli's; entries tagged "kepler"
or "jacobi (ex-kepler)" are mine. The reverse hazard exists too (I filed
nothing under the memory leg). My docs are all in
`runs/overnight_20260813_overlap_campaign/` and signed kepler or
jacobi-ex-kepler.

## 1. What this lane did (pointers, all pre-registered where applicable)

| doc | content |
|---|---|
| `IDLE_WINDOW_DECOMPOSITION.md` | the 20.8 s pure-idle decomposition (PP bubble ≈ 0 in idle; indexer nonzero leak 6.97 s; dispatch tax 10.4 s; seam 1.95 s) |
| `IDLE_RESIDUAL_MAP.md` | host-class vs staged-lever reconciliation (stack covers 8.09 s/40%); B/F d16 band (+4–7%/+5%, refute <+1.5%); §7 stage asymmetry; §4 outcome line |
| `PP_REBALANCE_ARM.md` | layer-rebalance KILLED (top-k-freq-4 ⇒ quantum 4; 38/40 already optimal); reopen iff H > 2L′ |
| `W1B_TRACE_READS.md` | VOID-no-data (W1b died at fit) |
| `W1C_TRACE_READS.md` | B/F mechanism verification — nonzero 55,328→1,328 exact; idle −7.9 s; decompression measured |
| `W1_V1_COMM2_READ.md` | W1 V1 comm-2 read (separation in, overlap unrealized) + source check + **W4 option-6 closure (FIRED, PARTIAL, convoy owns residual) + O1–O5 + the reconciliation sentence** |
| `R2_SEAM_PROBE.md` | seam py-spy spec + results (seam content named; H1 confirmed/H2 refuted) |

## 2. Trace inventory (Mac-side `~/perf_profiles/lps-1062/pp2cp8ep8/`)

| file | size | what | served |
|---|---|---|---|
| fe127_d16_rank0.pt.trace.json | 2.83 GB | fixed-wheel number-of-record d16, rank0 | :9002 |
| l3_d16_rank0.pt.trace.json | 2.80 GB | old-wheel d16 rank0 (class-structure control) | :9004 |
| l3_d16_rank8.pt.trace.json | 3.00 GB | old-wheel d16 rank8 | :9005 |
| w1c_b1_off_d16_rank{0,8} | 2.81/3.01 GB | W1c B/F OFF-arm d16 pair | :9006/:9007 |
| w1c_b2_on_d16_rank{0,8} | 1.32/1.43 GB | W1c B/F ON-arm d16 pair | :9008/:9009 |
| w3_v1_rank{0,8} | 702/749 MB | W1 V1 mechanism (d4 window) | :9010/:9011 |
| w4_on_d16_rank{0,8} | 2.82/3.01 GB | W4 option-6 ON-arm d16 pair (**no-TF32 class — marked confound**) | :9012/:9013 |
| r2_seam_probe/r2_seam_rank{0,8}.speedscope | 9.4/9.8 MB | R2 py-spy seam captures (420 s @250 Hz, --threads) | — |

Older d4-era traces in the same dir (mn_d4/diag_d4/runB_d4/l5_flex_d4 etc.)
are not currently served; their shas are in `pp2cp8ep8/BOX_A_ARTIFACT_MANIFEST.md`.

**sha256:** `pp2cp8ep8/w1c_traces.sha256` covers the W1c pair, w3_v1 pair, and
w4 pair (appended in order); `pp2cp8ep8/r2_seam_probe/r2_probe.sha256` covers
the speedscope files. The four big Aug-13 traces (fe127/l3/l5) are
sha256-verified per the manifest.

**Servers:** all trace_processor instances on this Mac, pid-stable as of
handoff — :9002 (29344), :9004 (34846), :9005 (40020), :9006–9009
(67515–67518), :9010/:9011 (6087/6088), :9012/:9013 (17553/20542). Query via
the perfetto python client (`TraceProcessor(addr='http://localhost:PORT')`).
Kill freely when the report is done; the files are the durable record.
(:9003 was bohr's fe127 server pre-restart — not mine, status unknown.)

## 3. Query machinery (reproduce-anything note)

- The exact SQL for every figure is inlined in the docs (gap-union, union-trick
  coverage, containment-sweep classification). The ad-hoc scripts lived in the
  opencode tmp dir and are transient by design — the docs are the record.
- **Operational lesson (two wedges, both mine):** never send per-row
  enclosure joins (gap×slice or kernel×CF EXISTS joins) against the slice
  table — they wedge the serial trace_processor server. Window-function union
  coverage answers the same questions in seconds. A wedged OWN instance:
  kill + restart costs ~2 min reload.
- Perfetto venv for the client: `/var/folders/.../opencode/perfetto-venv`
  (transient); recreate with `uv pip install perfetto` in any fresh venv.

## 4. Open items (none are this lane's work — listed for the report)

- R2 follow-ups (fix directions, pre-registered decision rules in
  R2_SEAM_PROBE.md §3.4/§5): async/shrink the per-step `broadcast_object_list`
  (worker.py:311, ~7.1 s on rank8 — likely rendezvous-dominated = the boundary
  skew wearing a broadcast); narrow/skip `_communicate_shapes`' full-device
  sync for static-shape runs (also ~20% of loop-thread active time in-step —
  worth its own look); prefetch the THD packing. EV ~1–2 s of the 2.66 s seam.
- The convoy/balance through-line (W4 closure): residual exposed-comm mass is
  convoy/imbalance-dominated; next real lever is expert/token de-staggering,
  roadmap-scale.
- Reopen conditions on record: PP-rebalance iff H > 2L′ (PP_REBALANCE_ARM.md
  §5); A-v3 stays parked until B/F+C′ land and the absorption regime shifts.

— jacobi (ex-kepler), 2026-08-14
