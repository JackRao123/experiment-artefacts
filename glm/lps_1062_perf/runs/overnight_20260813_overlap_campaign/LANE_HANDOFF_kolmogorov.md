# LANE HANDOFF — kolmogorov (ex-bohr) adjudication/spec lane, 2026-08-14

For the successor named by turing. One page; everything referenced is on disk
in `runs/overnight_20260813_overlap_campaign/` (the campaign folder) unless
noted. My session's context was intact through the restart wave; this doc is
the standalone rehydration.

## Lane state: the campaign's box program is in its LAST window (P4 soak)

**In flight — the soak (wprm693, lovelace drives, this lane adjudicates):**
spec = `P4_SOAK_SPEC.md` (FINAL; ship stack = mission + B/F ONLY). Order:
S2 parity (RUNNING at handoff: fresh boot A B/F-OFF vs fresh boot B B/F-ON,
fixed 9-datum set; bars loss rel ≤1e-6 / logprobs ≤1e-3 / datum accounting
EXACT; B/F is bitwise-exact by construction so any exceedance = real bug =
STOP+escalate) → S1 (60-step d16 soak; per-window bars + STOP table in the
spec; jacobi's R2 micro-window rides an idle gap — coordinate with jacobi) →
S3 (export sanity: 392-key/shape set + non-degenerate values; one sync-save
cycle end-to-end). Then: snapshot everything per the clobbering rule; box
goes idle-armed for Jack's stop/keep (turing relays).

## Closed under this lane tonight (all adjudicated, all on the NOTEBOOK spine)

- **B/F host-cache pair (lever 3) — WIN, fully closed.** Rebuilt/pushed:
  trainers+bridge `jackrao/lps-1062-bf-rebuild` (e864115a / 0e356eb2) → mcore
  PR #26 (`jackrao/lps-1062-ship-bf` = 500ce306a; byte-verified == Aug-9
  patches). W1c: d16 **+5.0% distinguishable** (1103 vs 1050.5; kepler's
  band; jacobi's trace read confirmed leak-causality EXACTLY: nonzero
  55,328→1,328 vs ~1,330 pre-registered). W1d (customer 16k-d32): **+6.2%**
  warm-confirmed. Memory flat, canaries clean. Merge-queue item 7 staged for
  Jack (unconditionally ready). Specs/notes: `REBUILD_NOTE.md`,
  `W1C_SPEC.md`, `W1D_16K_PROBE.md`; A-v3 decontaminated restack pushed
  (`jackrao/lps-1062-ship-av3-wo-w1` 40d8b3578).
- **NCCL lever 6 — closed cheap.** `NCCL_RESWEEP_PLAN.md`: ship knobs are
  NET/IB-scoped ⇒ inert on this topology (Arm-1 measured: |Δ| 6 ≤ spread 10);
  Arm 2 killed at the Mac gate (CP collectives wait-dominated:
  min-at-wire-floor / avg-14-18×-floor skew signature). Keep-guidance for the
  ship package on merge-queue item 9.
- **W1 + option-6 (lever 4) — closed-negative, mechanism-explained.** V0-V2
  PASS = the multi-EP-group verification PR #28 deferred (2-EP-group mission
  topology arms clean). W4 (option-6 vs W1-armed baseline): flat-to-negative,
  under spread, both rungs ⇒ both gates default-off, "structurally in, prize
  unrealized". jacobi's two-rank trace read: the reorder fires but the hide
  is convoy-limited by peer arrival (26.6%/18.7% vs the >80% bar) — the
  convoy tail is the shim+dial program's target class (report through-line).
  Docs: `W1_GO_NOGO.md` (hausdorff's spec of record), `W1_V1_COMM2_READ.md`
  (jacobi, incl. the option-6 O1-O5 pre-registration).
- **Dial validation rung 1 — 7/7 on hardware** (wlxj8vw B200, box since
  stopped per Jack's order): `DIAL_GRAD_EQUIV_SPEC.md` (T1-T7 incl. the T2
  negative control), `DIAL_VALIDATION_LADDER.md` (rung 2 = consume W1b's
  measured table; K number of record = pauli's model revision), test file
  `test_recompute_dial_grad_equiv.py` + evidence logs. The dial build itself
  is pauli's lane (ex-jacobi).

## Operational knowledge the successor needs

- **Zero-tolerance wait rule** (Jack, via bayes): never block the session on
  manual waits; `wait_trainer_health.sh` backgrounded; event-driven watchers
  (nohup script + send-message self-ping) for long transfers — pattern in
  /tmp/w1c_watch2.sh (rewrite per window; self-message = `kolmogorov` →
  successor's mailbox).
- **Tree swaps on the campaign box** (the F4-class hazard): the checklist is
  in `P4_SOAK_SPEC.md` §"Boot checklist" — canonical bits 73c24b00 + TF32
  patch verified by **applied-diff sha256 == f503c9bf…** (NOT grep count —
  miscalibration caught by lovelace, papercut pc_6129414b7e62) + mcore
  gate-stack dirty-diff sha256 == e1e46818… EXACT (never git-clean the mcore
  submodule; the Mac-side snapshot is the restore path) + wheel 1.27.0 +
  lovelace's pre-dispatch squeue-empty rule.
- **Wheel bump on a fresh box:** run `uv pip install` from OUTSIDE the repo
  dir (or `--no-config`) — inside the repo the pyproject pins silently
  constrain (papercut pc_1acf1a2ae1cf). B200/cu12 lane: frontend imports as
  `cudnn`; B300/ali cu13 lane: `cudnn_frontend`. DSA gate files:
  `server/tests/unit/dp_worker/test_cudnn_dsa_indexer_{launch_stream,topk}.py`
  (5 passed / ~22-23s = green record).
- **trace_processor on this build:** `--httpd` 404s the classic `/query`
  endpoint — use `query -f file.sql <trace>` (papercut pc_627024976652).
- **Anchors of record (wprm693, fixed wheel, full canonical env):** d4 = 920,
  d16 = 1050.5 (B/F-off); d16 = 1103 (B/F-on). Tonight's earlier 848/945-962
  anchors were no-TF32/no-ship-env — env-confounded, do not cite as the
  record class.
- **Pre-registration discipline:** bars before boots; mean-vs-spread with the
  steady-state warm-class convention (documented when invoked); falsifiers
  escalate, never re-bench.

## Contacts / roles at handoff

turing = orchestrator (all reports route there). lovelace = box mechanics
(drives; precise, self-catching — trust their checkpoints). jacobi (ex-kepler)
= trace/mechanism lane (serves traces on :90xx; wants landing relays with
final byte sizes). pauli (ex-jacobi) = dial build. ramanujan (ex-lebesgue) =
shim/contract lane.
