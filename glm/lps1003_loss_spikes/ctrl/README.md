# ctrl/ — positive control FIRED: batch-0 partition 1 alone, uniform loss, positional cutoff mapped (2026-08-01, run ctrl_p1_0801_013628)

**Design (Jack):** validate the whole day's negative chain AND shrink f in
one shot — send ONLY batch-0's first partition (docs 0–6, the `[0-6]` row
whose tails {4,5,6} were the historically destroyed datums), with **uniform
supervision on every position** so the full-position logprob dumps show
exactly where destruction switches on ("the mess-up cutoff").

**Result: FIRED, 1/1 boot — rep0 destroys the row-tail docs, heals by rep1.**
Today's earlier negatives (cp1, cp16@32k single- and multi-doc) are therefore
real, and f is now much smaller than batch-0: ONE 254.5k packed row (7 real
docs), single partition, /forward at the boot window.

## Setup

- Golden 262k config (`trainer-config.flash.json`: tp1/pp1/ep16/cp16, flash,
  lora32), prodenv conn UNSET + `BT_SKIP_FULL_WARMUP=1`, boot healthy 735s,
  3 reps at `/health` (91s/10s/10s).
- Payload `payload_b0_part1_uniform.json`: batch-0 docs 0–6, exact prod token
  ids in original order (packing reproduces), weights uniform 1/(L−1) —
  no masks. Doc lens [41381, 30061, 20896, 12169, 60181, 34647, 55182],
  total 254,517 ≤ 262,144 → ONE row, per-CP-rank local ~15.9k, zigzag chunk
  7,968 (padded S 254,976).

## Result — per-doc full-sequence NLL

| doc (row span) | rep0 | rep1 | rep2 |
|---|---|---|---|
| 0 (0–41,381) | 3.872 | 3.873 | 3.860 |
| 1 (41,381–71,442) | 3.615 | 3.630 | 3.638 |
| 2 (71,442–92,338) | 4.396 | 4.344 | 4.449 |
| 3 (92,338–104,507) | 2.818 | 2.849 | 2.815 |
| 4 (104,507–164,688) | **4.777** | 3.546 | 3.379 |
| 5 (164,688–199,335) | **8.358** | 3.762 | 3.928 |
| 6 (199,335–254,517) | **8.602** | 3.995 | 3.748 |

## The cutoff (chunk-aligned, mean rep0−rep1 logprob per 7,968-token chunk)

- **Chunks 0–12 (rows 0–103,584): clean** — |mean Δ| ≤ 0.10.
- **Chunks 13–16 (rows ~103,600–135,456): transition/flicker band** —
  mixed signs (−0.45, +1.03, +1.45, +2.68): rep1/rep2 are ALSO unstable
  here (positive mean = rep1 worse than rep0), i.e. mid-window flicker,
  not clean healing.
- **Chunks 17–31 (rows 135,456–254,517): destroyed in rep0** — mean Δ −2.0
  to −6.2 nats per chunk, sustained to end of row.

So the second-half law holds to ~1 chunk: hard onset ≈ row 135k (chunk 17
of 32), midpoint 127.5k, with a 4-chunk flicker skirt starting just before
the midpoint (doc 4 spans it — its first ~30k tokens flicker, its last ~28k
are hard-destroyed).

Also measured (uniform loss made this visible for the first time):
- Every doc's first ~2k tokens are immune with near-zero churn — shared
  template prefix, near-deterministic predictions.
- Even between "healed" reps (1 vs 2), per-chunk mean |Δ| is ~0.25–1.6 in
  the first half vs ~1.3–3.3 in the second half — the row's second half
  stays ~2× noisier after healing. (Refines the min32k "churn is
  position-independent" note, which was measured at 30k rows only.)

## Minimization state after this run

f_current = golden 262k cp16 config + conn unset + skip-warmup + ONE packed
row of 7 real docs (254.5k) + first /forward at /health. Fires 1/1.
Between 30k (clean, 3 experiments) and 254.5k (fires) the row-scale
threshold is unbisected → next: same design at ~64k and ~131k rows (drop
tail docs / truncate to fit smaller max_seq_len leaves).

## Files

`boot_ctrl.sh` (dispatch-only boot), `payload_b0_part1_uniform.json` +
`.meta.json` (exact body + doc provenance incl. original prefix_lens),
`ctrl_p1_0801_013628/` (probe.log, rep{0..2} full-position logprobs,
per-chunk analysis reproducible from dumps). Devbox originals:
`/root/.cache/user_artifacts/lps1003/ctrl/`.

## Addendum: steady-state soak (soak_0801_015956, 146 reps @ ~9.5s)

Same payload repeated on the same warm trainer (window long spent), conn
still UNSET. Result: **the second-half instability is a persistent
steady-state phenomenon, not a boot-window residue.**

| doc | median | min | max | low-state visits (of 146) |
|---|---|---|---|---|
| 0–3 | 3.856 / 3.625 / 4.425 / 2.828 | ±0.06 total range | | 0 |
| 4 | 3.454 | 2.693 | 3.834 | 4 |
| 5 | 3.984 | **2.161** | 4.190 | 11 |
| 6 | 3.852 | 2.774 | 4.358 | 11 |

- Low-state visits are coherent across docs 4–6 within a rep (row-level
  state per forward), ~8%/rep, rate FLAT over the run (4/48, 4/49, 4/49).
- Direction: the rare state is LOWER NLL (better prediction). Open
  question with two readings: (a) the rare low state is the CORRECT
  computation and ~92% of steady-state forwards are semi-degraded in the
  row's second half under prod env; or (b) the median is correct and the
  dips are something else. Decisive discriminator (not yet run): boot the
  same config with `CUDA_DEVICE_MAX_CONNECTIONS=1` (stock devbox env) and
  re-probe — the conn=1 level identifies the correct state.
- Caveat: all prior "steady state is stable" lore (phaseA wobble floor
  etc.) was measured under conn=1; this is the first long steady-state
  series under prod-faithful env at 254.5k rows.
- Exemplar full-position dump pairs (low-state rep + neighbouring median
  rep) captured in `soak_0801_015956/anomaly_*.json.gz` (17 files);
  per-rep NLLs for all 146 reps in `soak.jsonl`; driver `soak.py`.
