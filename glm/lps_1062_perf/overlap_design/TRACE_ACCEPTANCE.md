# Round-3 trace acceptance criteria (fibonacci) — run per A/B capture

Tooling: trace_processor v56.1 (checker-authoritative), steady window only, never window 1.

**Numerics bars (adjudicated 08-09, ARM 0):** a cross-boot bitwise floor exists in this stack
independent of any patch (same-config warmup0 deltas 0.7–1.7e-3 across boots; boot-time
cuBLAS/cuDNN algorithm selection). Therefore: "loss bitwise-identical" applies to IN-PROCESS
comparisons only (FIX C verify-mode replay-vs-fwd; W2's t2 in-process torch.equal gate) —
those stay hard. Cross-boot canaries use the house band: ≤2e-3 pass / >5e-3 stop, matched
--warmup-datums mandatory, warmup0 loss + warmup-datums recorded in every arm's JSON.
Where "bitwise per window" appears in the per-lever bars below, read it as: in-process gates
hard-bitwise + cross-boot canary within the band.
Baseline reference = gated-v2-4mb-steady (step 46.678 s): token-a2a 20.90 s/1800, probs-a2a
5.13 s/900 (avg 5.7 ms, serialized on the token stream), compute union 16.99 s, a2a↔compute
overlap 0.395 s (1.5%), GPU idle 1.82 s, dispatch→combine gap avg 10.85 ms (4.9 ms compute),
combine→dispatch gap avg 16.64 ms (13.3 ms compute), per-layer cycle ≈ 52 ms × 900.

**Two-tier verdicts (ruled 08-09 after ARM 2):** MECHANISM bars (trace/telemetry/canary;
box-independent) decide whether a lever works as designed. WALL bars are box-dependent on
318g61w (proven tail-bound fabric: per-layer critical path set by straggler tails, so
slot-shaving levers under-deliver here); wall deltas are recorded as informational and ship
claims about magnitude defer to a prod-grade-fabric measurement.
ARM 2 result: W1 mechanism PASS (probs 900/900 off-stream, gap 11.22→9.15 ms, token a2a
flat, canaries in band); wall −0.4…−0.6 s on this box vs −3.4 model.
W1 memory cost recorded: +1.3 GiB (second NCCL communicator channel buffers). W1 stays
armed for subsequent arms.
**Residual attribution CORRECTED (hilbert discriminator, post-ARM-2):** the miss is
LAUNCH-SIDE at CDMC=1, not fabric tails — fwd probs are issued together, run 0.27 ms p50,
fully hidden (conversion ~100%); the exposed third is exactly the backward-window passes
(late-start 50% inside CheckpointFunctionBackward vs 0% outside; corr(probs_dur,
token_dur)=0.075 kills the peer-tail reading). Prod runs CDMC unset → W1's full model
plausibly holds there.
**ARM 4 CANCELLED (provenance):** 318g61w runs CDMC UNSET already (provisioner removed
the pin per LPS-1003; /proc-verified) and so did qr4ggv3 (REVIEW_FIXA.md:58-62) — ALL
historical numbers are CDMC-unset; the "=1 devbox" premise was a stale .orig script. **W1 residual investigation CLOSED (hilbert cut, answer [ii]):** the bwd probs-reverse is
launched ~24 ms EARLY onto an IDLE stream and waits ~41 ms for its INPUT — the probs grad,
produced deep in the layer backward (post fc2-dgrad). Engine order, queueing, CDMC, and
peer-tail all refuted. Both v2 fixes dead (seq-bump: launch already early; combined
Function: would chain tokens-reverse to the late probs grad). Remaining lever = bwd-chain
reorder (~1–1.5 s), parked in W2V2_DECISION_stub as a W3-absence fallback — W3's lookahead
subsumes it. W1 ships as v1; residual understood and bounded.
W3's CDMC precondition (V2) is satisfied as-is; W3 canary joins tonight's ladder.

## W1 (BT_MOE_PROBS_A2A_COMM=1) — PASS iff ALL of:
1. Probs a2a kernels appear on a NEW stream (second EP communicator), not the token stream.
   SQL: group nccl SendRecv by track tid × dtype — Float rows must move to a distinct tid.
2. Probs no longer occupy a serial slot: dispatch→combine gap shrinks toward ~5 ms
   (GEMM-only), i.e. avg gap ≤ 7 ms (was 10.85).
3. Token a2a total within ±5% of baseline (20.9 s) — W1 must not perturb token comm.
4. Step wall: devbox runs CDMC=1 → model is −3.4 s (bwd probs-reverse wait head-of-line-blocks
   at =1; fwd+replay only) → accept ≥ −2.4 s (70%). Under CDMC unset (prod parity, later paired
   experiment gated on V2b) the model is −5.1 s. The devbox A/B under-reports prod BY DESIGN —
   record both expectations in the report.
5. Bench canary: loss BITWISE-identical per window vs gates-off same-box same-seed run
   (design claims exactness; ≤2e-3 is NOT good enough to pass W1).
6. Telemetry counters: issues == waits == 900 per profiled step on the new comm.

## W2 v1 (BT_MOE_A2A_PIPELINE=2) — PASS iff ALL of:
1. Token SendRecv calls ≈ 2× per phase (fwd: 4/layer-pass vs 2) at ~half payload
   (In msg nelems ≈ 201M for dispatch chunks).
2. a2a↔compute overlap ≥ 15% of token-a2a residency (baseline 1.5%).
3. V1 numerics gate passed BEFORE any timed run: torch.equal fwd output + input grads,
   gate off vs on (T2), incl. imbalance + zero-count-peer cases.
3b. tests/test_w2_chunk_plan_t1_seed.py ALL PASS (named precondition; catches the send-vs-recv
   combine-bounds class of bug that symmetric fixtures cannot see — hilbert review 08-09).
   W2+FIX-C composition: replay-hit restore must round-trip the two W2 host matrices
   (store + verify-mode compare), tested with both gates on.
3c. tests/test_w2_chunked_a2a_functions.py ALL PASS (named precondition; catches the
   attrs-lost-on-alias combine-wait bug (Function returning its input) and the None-grad
   crash — hilbert review round 2). t2 gate must assert FIX-C replay-cache hits>0/misses==0
   in the use_fixc branch (silent-fallback trap).
4. Step wall: −2 s or better standalone (model −3.0 s fwd-only).
5. Peak mem delta ≤ +1 GiB at 131k×d4 AND 16k×d32 (design says ≈0).
6. Loss bitwise-identical (same bar as W1).

## FIX C (BT_MOE_DISPATCH_REPLAY_CACHE=1) — PASS iff:
1. VERIFY-mode soak (≥20 steps) zero mismatches first.
2. Replay-side cudaEventSynchronize count 300 → ≤ a few; fwd-side 300 unchanged.
3. Dispatcher Long allgather (256-elem) count 600 → 300.
4. Loss bitwise-identical vs cache-off.
5. Step wall: report delta (no hard bar — value is compounding, expect −0.5…−3 s).

### C′ adjudication record (2026-08-09/10, boltzmann; verdict accepted by kepler)
- **C′ timed arm (B+F+W1+C′, 318g61w, boot 05:37 UTC): PASS ON MECHANISM, both ranks.**
  All mechanism rows green (replay eventsync 0/0, fwd 300/300, Long allgather
  600→300, A-off rows 0, B+F invariants intact). The two drain-seconds rows
  (nonzero_cpu_s, streamsync_cpu_s) measured ~24.1–24.8s vs the profile's
  pre-C′ ≤21.0: classified wait-redistribution (kepler case b), NOT busy-CPU —
  drains 97.4% GPU-covered, eventsync+nonzero wait conserved (34.21 vs 34.01s
  vs the W1 control), kernel_count ↓1,849, GPU busy and wall flat. Bounds
  re-baselined in check_acceptance.py with the old values annotated.
- **Staging amendment (ratified by kepler):** timed arms run C′ VERIFY OFF by
  design — the verify sync overhead would contaminate the timing, so the
  staging is methodologically necessary. The verify-on bar (312 asserts/step,
  zero raises) is carried by the VERIFY-mode soak (20/20, 6300/6300 hits, zero
  ARM-1-class raises). hilbert's handoff §3 conflated soak-window and
  timed-window counter lines; the timed-window bar is stashes/forces/misses/
  shape_mismatches + stash-bytes plateau only.
- Δ(stashes−forces)=67 in the timed telemetry reconciled arithmetically (CPU
  runahead into the next forward at the window-print instant + print-before-
  increment off-by-one); misses=0/shape_mismatches=0 in every window; stash
  peak 157.3 MB constant = no leak.

## Straggler capture (W5): with rank-0 AND rank-8 traces of the same step:
- cross-correlate per-layer a2a start/end skew between ranks; report p50/p90 skew
  and whether hot-expert rank (combine In-size max) predicts the straggler.

## Combined-ship preconditions (cross-patch interactions)
- F2 × W2: phantom partitions make zero-count (peer,group) a2a ROUTINE (16–32 dummy tokens
  × top-8 over 256 experts ⇒ most peers/groups at 0 rows). Before any combined ship: one
  DP2 + phantoms + W2 canary run (helmholtz F2-review finding F-c, 08-09).

Failure protocol: any FAIL → stop the lever, capture trace + logs, message fibonacci.
Never stack a second lever on an un-passed first.
