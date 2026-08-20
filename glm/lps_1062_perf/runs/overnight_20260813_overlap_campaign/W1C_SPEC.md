# W1C SPEC — B/F host-cache A/B (+ NCCL Arm-1 fold-in) on wprm693 (bohr spec, lovelace drives)

Window: W1c (after W1b's DOES-NOT-FIT close). Mission box wprm693 (2×8 B300
ali). Spec/adjudication: bohr. Box mechanics: **lovelace** (drives everything
below). Trace read at d16: jacobi. References: `REBUILD_NOTE.md` §5 (bands),
`NCCL_RESWEEP_PLAN.md` §4 (3-boot design), `CAMPAIGN_ORDERS.md` (standing
rules). Anchors REBASED to tonight's wprm693 measurements (bayes, 2026-08-14):
**d4 848, d16 945 fresh / 958–962 plateau** — same-class pairing throughout
(fresh with fresh, plateau with plateau); the A/B is self-referenced (on vs
off on the same box/tree/boot-class), anchors bound boot-to-boot drift only.

## 0. Tree state (the "tree swap" — verify BEFORE any boot)

Target = canonical campaign exact bits, on the shared CPFS clone
(`/root/.cache/user_artifacts/trainers_main`):

1. trainers @ **73c24b00 + TF32 patch** — `git status` clean except
   `chunked_lm_head.py` (+73/−10, the tf32_head_port). NOT the dial/shim
   branches, NOT W1b's blockK config (config is JSON — the git tree is what
   matters).
2. mcore vendored tree = the CPFS dirty tree (57efae08b + the Aug-9/10 gate
   stack, ALL GATES OFF by default). Verify: `git diff` sha256 ==
   `e1e46818dfc684566815fdf482574c08bc275451d4471706dc89d0154eee3653`
   (gauss's snapshot) and `grep -c BT_DSA_CP_LAYOUT_CACHE
   megatron/core/transformer/experimental_attention_variant/dsa.py` ≥ 2.
   **Do NOT `git checkout/reset/clean` inside the mcore submodule — it is the
   only live copy of the gate stack.**
3. Wheel: trainer venv cudnn-frontend == **1.27.0** (W0 gate; should already
   hold from W1b — verify, don't assume).
4. `BT_SAVE_STATE_SYNC=1` everywhere (F1 landmine). `wait_trainer_health.sh`
   only, **backgrounded with its exit reported** (zero-tolerance wait rule —
   no foreground log-tailing, no sleep-loops; stay responsive).

## 1. The 3-boot design (one-variable discipline preserved across the pair)

Ship env = `NCCL_IB_QPS_PER_CONNECTION=8 NCCL_IB_SPLIT_DATA_ON_QPS=1
NCCL_NCHANNELS_PER_NET_PEER=8`. Base env on every boot: `BT_TF32_LM_HEAD=1
BT_SAVE_STATE_SYNC=1` (+ the bench-kit's usual). Config: the headline M=N
131k config (`trainer_pp2cp8ep8_131k.json`), driver bench_driver2c.py,
~524,288 tok/step at d4 / 2M at d16.

| boot | ship env | B/F gates | serves |
|---|---|---|---|
| **1** | ON | off (default) | shared off-arm (B/F off-arm AND NCCL on-arm) |
| **2** | ON | `BT_DSA_CP_LAYOUT_CACHE=1 BT_THD_ROPE_HOST_CACHE=1` | B/F on-arm |
| **3** | **OFF** (all three knobs UNSET, /proc-verified absent) | off | NCCL off-arm |

Per boot: d2 canary FIRST (loss 12.2–12.4 band + gn comparable to the
fixed-wheel d2 band 0.36–0.49 — no perf run before a clean canary) → d4 ×2
runs (labels `pp2-131k-w1c-b{boot}-{arm}-d4{,-r2}`). Boots 1+2 continue to
d16 ×2 (labels `...-d16{,-r2}`) with ONE traced window each
(`BT_PROFILE_RANKS=0,8`, traces pulled off `/tmp/checkpoints/profiles/`
IMMEDIATELY to `lps1062_pp2/traces/w1c_*` + Mac pull). Boot 3 is d2+d4 only.

**Boot-2 hard gate (before any perf run):** boot log must show
`BT_DSA_CP_LAYOUT_CACHE=1: DSA packed-CP layout cache ACTIVE` AND
`BT_THD_ROPE_HOST_CACHE=1: THD RoPE cu_seqlens host cache ACTIVE`, each ×16
ranks at WARNING. No ACTIVE line = inert = STOP (do not bench an inert arm —
Aug-9 v1 lesson).

**Memory watch:** peak reserved per run; abort line >255 GiB. Expected flat
vs off-arm (Aug-9: memory flat).

## 2. Pre-registered bars (REBASED; on record, unchanged in structure)

- **Canary gate (all boots):** d2 loss 12.2–12.4, gn 0.36–0.49-comparable.
  Drift >5e-3 = STOP + verbatim report.
- **B/F d4 (primary rung):** win = on-arm mean > off-arm mean **+2%** with
  the mean-vs-spread rule (|Δ| < max(arm spreads) ⇒ perf-indistinguishable).
  Prediction on record (gauss): +0–5%.
- **B/F d16:** paired same-class windows. Bands on record (kepler, rebased to
  the paired off-arm, not the absolute): central +4–7%, floor +1.5–2.5%,
  ceiling +9–11%; **< +1.5% refutes leak-causality → escalate, don't
  re-bench.** jacobi's trace read (the nonzero-collapse pre-registration is
  W1c's): **the numbers of record are jacobi's `W1C_TRACE_READS.md`** (their
  M1 totals ≈1,330 surviving nonzero/step, incl. the deliberately-untouched
  bare-bwd 608 — do not restate constants here); plus pure-idle shrink of up
  to ~7 s/step on the on-arm trace vs off-arm (kepler's leak number).
- **NCCL Arm 1 (d4 only):** |Δmean| ≤ max(spreads) ⇒ ship env INERT on this
  topology (documented in the ship package); Δ < −3% beyond spread ⇒ env
  hurts ⇒ drop from the PP2 ship env; Δ > +3% beyond spread ⇒ contradicts
  Run B (+0.4%) ⇒ investigate before believing.

## 3. Adjudication protocol (bohr, on results landing)

lovelace reports per boot: canary line, per-run tok/s/GPU + step + peak mem,
the ACTIVE-line grep evidence (boot 2), trace paths (d16). bohr adjudicates
against §2, writes the verdict into NOTEBOOK.md + reports to bayes. Any STOP
condition → halt the window, report to bayes with the verbatim evidence.

## 4. Timeline estimate

~2.5–3 h total: 3 boots (warm cache ~8–15 min each) + d2 canaries + 6 d4 runs
(~1 min each) + 4 d16 runs (~2.5 min each) + 2 traced windows + pulls.
