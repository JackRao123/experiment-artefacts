# rearm/arming — the 2026-08-02 arming-condition lab (question CLOSED)

Conclusion: `../ARMING_MECHANISM.md`. Predictions + scorecard (written
before each run, per the bisect-don't-guess rule): `PREDICTIONS.md`.

- `arm_lab.py` — single-GPU rep loop, unfixed wheel via PYTHONPATH shadow,
  CI-repro geometry (out 2.25 GiB → split fill), full-scan detector, modes
  {warm, ec, ec_premap, ec_prealloc, ec_map_elsewhere, busy, busy_ec}.
- `run_one.sh` — devbox-side runner (fresh process per run; env hygiene:
  conn unset, expandable_segments:True, PYTHONPATH shadow).
- `arm_witness.py` / `wit_busy.py` — kineto witnesses (profiled + unprofiled
  execs, per-exec corrupt/clean labels).
- `probe_stream.py` — the decisive probe: ExternalStream(0) returns
  rotating torch pool streams (32, cycling), not the default stream.
- `results/` — one jsonl+log per run (armed modes fire ~15/16 execs;
  busy exec1 protected 13/16 while exec2 fires 16/16; conn1 and fixed
  wheel 0/N).
- `traces/` — CUPTI witness pair: `trace_w1_warm1` (armed: F1 → S@158µs →
  F2@3532µs mid-S, fills on pool stream 25, S on legacy 7),
  `trace_busy_exec1` (disarmed: junk holds legacy stream to 23.9 ms, fills
  done by 13.3 ms, S starts at 23.9 ms, clean), plus boot/ec1/warm3.

Substrate: devbox q8x5ky3 node0 (rescheduled fresh 08-02, hostname
b300-1-z7sxjdpi-0020, reports "NVIDIA L20D" but sm103; all GPUs idle).
Node-local copy of this lab: `/root/arming/` (unfixed wheel shadow in
`/root/arming/unfixed/`, sha 5d0e429e…). The venv wheel remains the FIXED
+dsatopk5 content — untouched.
