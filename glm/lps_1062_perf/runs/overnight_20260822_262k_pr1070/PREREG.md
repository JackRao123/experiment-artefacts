# 262k profiling prereg (2026-08-22, box wgm8row 2x8 B300)
Task: does PR #1070 (PP2/CP8/EP8, head d8b9648f) work at 262k, and what does
tip of main (9b039d6b) do at 262k. Driver of record: profile_driver_new.py
(same as 131k campaign). Operating point: d2 = 524,288 tok/step (matches
A-anchor-262k of 2026-08-09: 630 tok/s/GPU, EP16/CP16, peak 263.7 GiB).

Runs (clean env both; single variable = code+layout):
  R1 main@9b039d6b, TP1/PP1/EP16/CP16, seq 262144, d2, control-repeats 2
  R2 pr1070@d8b9648f, TP1/PP2/EP8/CP8, seq 262144, d2, control-repeats 2
  R3 (only if R2 passes AND its peak mem <= 240 GiB): R2 config, d4

Pre-registered bars:
- WORKS = health 200 + all driver windows complete + loss ~= 12.3 +/- 0.15
  + grad_norm < 2 + no OOM/NCCL abort + peak < 268 GiB phys.
- PR faster only if control tok/s/GPU mean beats R1 by > 8% (window noise
  in the Aug-9 anchor was ~6% across 3 windows).
- Headroom flag: report peak vs 275,040 MiB phys; flag < 10 GiB headroom.
- Loss/gn checked vs anchor (12.34-12.36 / 0.69-0.94) as canary, not a bar.
