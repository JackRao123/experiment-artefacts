# Box 1 (318g61w) arm log — grothendieck, 2026-08-10 UTC

## A-v3 arm — INVALIDATED (under-armed boots), recorded as-run
- Soak boot 08:55 + VERIFY=0 boot 09:33 ran env ship+B/F+V3 only; C′ (BT_MOE_ROUTING_REPLAY_FORCE) and W1 (BT_MOE_PROBS_A2A_COMM) were OFF (boot-region marker lines: ROUTING_REPLAY_FORCE "present but DISABLED"; C′-era boot line 76413 shows ACTIVE + W1 telemetry live).
- Root cause: operator read "Stack: B/F+W1+C′" as tree content, not env arming. Lesson adopted: pre-drive ARM-CHECK (boot-region ARMED/ACTIVE lines for every declared gate, pasted pre-drive).
- As-run data (B/F-regime, valid as such): soak 22 windows verifies==kicks-1 constant (curie PASS, regime-scoped); canary vs r3anchor golden (B/F vs B/F+V3, regime-matched per curie): w0 +2.8e-3 m0 +1.8e-3 m1 +1.2e-3 m2 +0.9e-3; timed 131k-d4 695.4 / 16k-d32 690.1 (mismatched vs C′ refs — discarded for the A-v3 delta); rank0 traces x2 (A-v3 rows PASS: eventsync_a==312, 1.9ms, 0>1ms; nonzero_gt5ms 7; streamsync 463; 7 FAILs = post-C′ pins on a C′-off trace = regime mismatch).
- A-v3 re-arm = morning item unless box frees before 12:15 (helmholtz ordering).

## W3-v3 swap on clone mcore_318g61w_av3 (10:38-10:40)
- fixa_v3 REVERTED (git apply -R clean; dsa_cudnn_kernels 41dd2bab + dsa_kernels ece8175b == shared tree bytes; dsa.py 417bb91d)
- W3 v2 REVERTED (git apply -R clean; tree = shared bytes minus W3v2)
- W3 v3 APPLIED (patch md5 05dda37f68f3113747a812e357d9b552, applied clean on the no-W3 base — NOTE: the v3 patch is a full patch vs no-W3 base, NOT a v2->v3 delta; v2 must be reverted first. helmholtz informed.)
- Clone stack now: base-chain (B/F+A-v2+fixC+verify-instr+W1+C′) + W3-v3. Gate: BT_MOE_LOOKAHEAD_RECOMPUTE=1.

## W3-v3 CANARY boot (11:06 READY, job 41) — PRE-DRIVE ARM-CHECK (boot-region markers, all verified present)
- [sitecustomize av3] megatron_core re-pointed -> mcore_318g61w_av3
- BT_DSA_CP_LAYOUT_CACHE=1: DSA packed-CP layout cache ACTIVE
- BT_THD_ROPE_HOST_CACHE=1: THD RoPE cu_seqlens host cache ACTIVE
- BT_MOE_ROUTING_REPLAY_FORCE=1 (FIX C-prime) ACTIVE
- BT_MOE_PROBS_A2A_COMM=1: MoE probs all-to-all on a second communicator ACTIVE + armed (second communicator over 16 EP ranks)
- BT_MOE_LOOKAHEAD_RECOMPUTE=1: lookahead recompute (cross-layer recompute/backward overlap) ACTIVE
- BT_DSA_BWD_ASYNC_NONEMPTY_V3: 0 occurrences (fixa_v3 reverted from the clone — absent as designed)

## A-v3 RE-ARM soak boot (READY 12:20) — PRE-DRIVE ARM-CHECK (all 7 verified)
- re-point -> mcore_318g61w_av3 (fixa_v3 re-applied, dsa_kernels md5 969d2c32 matches the morning state)
- B (CP_LAYOUT_CACHE=1) ACTIVE / F (ROPE_HOST_CACHE=1) ACTIVE
- C-prime ROUTING_REPLAY_FORCE=1 ACTIVE / W1 PROBS_A2A_COMM=1 ACTIVE
- V3 BWD_ASYNC_NONEMPTY_V3=1 ACTIVE (+VERIFY=1)
- W3 LOOKAHEAD_RECOMPUTE explicitly DISABLED (W3-v3 bytes in tree, gate off proven)
- Fresh boot, history-symmetric.

## A-v3 RE-ARM VERIFY=0 boot (READY 13:17) — PRE-DRIVE ARM-CHECK: re-point + B/F + C′ ACTIVE + W1 ACTIVE + V3 ACTIVE (VERIFY off) + W3 explicitly DISABLED — all 7 verified in boot region. Fresh boot, history-symmetric.

## HYBRID-GATE FINDING (13:45, curie/helmholtz confirmed by my per-boot table): C′ = TWO flags — BT_MOE_ROUTING_REPLAY_FORCE=1 (routing) + BT_MOE_DISPATCH_REPLAY_CACHE=1 (metadata cache). My W3-v3 canary + both A-v3 re-arm boots set FORCE only (CACHE off): routing decisions preserved (numerics valid, canaries/soak stand) but replay syncs/allgathers NOT eliminated (perf rows polluted vs the cache-on C′-era refs). Root cause: operator mapped the C′ shorthand to one env var; arm-check now enumerates hybrid pairs. A-v3 timed wall numbers (701/677) carry cache-off overhead; clean wall needs FORCE+CACHE+V3 boot.

## A-v3 RE-ARM leg (b) — VERIFY=0, FULL HYBRID (READY 14:05) — A2.1-E ARM-CHECK
- re-point -> mcore_318g61w_av3; B/F ACTIVE
- HYBRID PAIR enumerated: BT_MOE_DISPATCH_REPLAY_CACHE=1 "replay-metadata cache ACTIVE" + BT_MOE_ROUTING_REPLAY_FORCE=1 ACTIVE (both /proc-verified)
- W1 ACTIVE; V3 ACTIVE (VERIFY off); W3 explicitly DISABLED
- /proc environ truth pasted (7 BT flags + PYTHONPATH + expA131 config)
- Fresh boot, history-symmetric. Engagement markers (first-replay-hit + window counters) to be pasted from the canary drive.
- ENGAGEMENT (leg b, post-canary-start): "first replay cache hit" line present (multi-rank); window counters e.g. window 4: stores=1200 hits=1200 misses=0 shape_mismatches=0 — cache fully engaged (hits==lookups).
