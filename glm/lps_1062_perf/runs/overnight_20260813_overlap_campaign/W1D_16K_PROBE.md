# W1d — customer-shape 16k-d32 B/F probe (follow-on to W1c; kolmogorov spec, lovelace drives)

Sequencing: moved up by bayes (W2 paused on a Mac-side parity root-cause; the
box would idle). Protocol = W1c's, minus d16 and boot 3. Pre-registered here
before any boot.

## Why this shape

16k×d32 (524,288 tok/step, 8 docs per 131k partition) is the customer's
dominant regime and was B/F's BIGGEST Aug-9 win (+18.5%: 612 → 726 on the
golden mesh, convoy regime): the DSA packed-layout builders' host work scales
with **docs per partition**, so this shape carries ~8× the per-partition
bookkeeping of 131k-d4. That Aug-9 number was convoy-regime and does NOT
transfer as an expectation; tonight's honest band sits between d4's +4.9% and
that +18.5%.

## Tree + env (identical to W1c boots 1/2)

- Canonical exact bits: trainers 73c24b00 + TF32 patch; mcore dirty tree
  intact (verify sha256 `e1e46818...`; NO git-clean in the submodule).
- Wheel 1.27.0. `BT_SAVE_STATE_SYNC=1` everywhere. `wait_trainer_health.sh`
  backgrounded (zero-tolerance wait rule).
- Config: the headline M=N 131k config (max_seq_len=131072 = packed buffer,
  F1 rule unchanged — the packer fills 4×131k partitions from 32×16k datums).

## Arms (2 boots)

| boot | env | serves |
|---|---|---|
| 1 | ship env ON + `BT_TF32_LM_HEAD=1`, B/F off | off-arm (reference) |
| 2 | same + `BT_DSA_CP_LAYOUT_CACHE=1 BT_THD_ROPE_HOST_CACHE=1` | on-arm |

Boot-2 hard gate (before any bench): both `... ACTIVE` WARNING lines ×16
ranks. No ACTIVE line = inert = STOP.

## Rungs + bars (pre-registered)

- Canary per boot: first 16k window (16k×d8) — loss 12.2–12.4 band, gn
  comparable to tonight's bands. Drift >5e-3 = STOP.
- Main rung: 16k×d32 ×2 runs per arm (same-class pairing; steady-state
  convention for the warm class per the W1c ruling).
- Win floor: on-arm > off-arm **+2%** with the mean-vs-spread rule.
- **Prediction on record: +5–12%** (bounded below by d4's +4.9% — same
  partition count, more per-partition doc bookkeeping; bounded above by the
  Aug-9 +18.5% which was convoy-regime). **Falsifier: < +1.5%** ⇒ the
  leak-causality model fails at the customer shape → escalate, don't
  re-bench.
- Memory: peak reported; abort >255 GiB reserved (expected flat — W1c held
  143/163 GiB at d4/d16).
- No traced window by default (the mechanism question closed at d16); capture
  one only if the perf read surprises.

## Adjudication

lovelace reports per boot: canary line, per-run tok/s/GPU + step + peak,
ACTIVE-line grep (boot 2). kolmogorov adjudicates vs the bars, logs to
NOTEBOOK.md, reports to bayes.
