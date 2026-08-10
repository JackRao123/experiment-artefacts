# FIX A v3 on-box validation recipe (verify soak first, then timed A/B)

Prereq: FIX C's timed A/B must have PASSED first (v3 is a measurement of the
post-C regime; do not run it against a pre-C baseline). Apply `fixa_v3.patch`
on top of the current on-box stack (B/F/A + fixc + w1 + w2):

```
cd /root/.cache/user_artifacts/trainers_main/server/vendor/megatron-bridge/3rdparty/Megatron-LM
git apply /path/to/fixa_v3.patch   # dsa.py + dsa_kernels.py + dsa_cudnn_kernels.py
```

1. **Boot smoke (gate on):** `BT_DSA_BWD_ASYNC_NONEMPTY_V3=1` (+ usual B/F/C
   env; v2 gate stays OFF). Confirm in trainer_srun.log: `... V3=1: ...
   ACTIVE`, then `armed — first first-pass probe kicked`, then `first replay
   hit — first-pass probe consumed`. No `armed` line = inert — STOP.
2. **Verify soak (parity prerequisite):** add
   `BT_DSA_BWD_ASYNC_NONEMPTY_V3_VERIFY=1` for a ≥20-step soak at the 4×131k
   shape. Every replay recomputes the emptiness flag and asserts equality
   with the first-pass value; any mismatch raises RuntimeError (loud stop —
   the structural-emptiness assumption is false on this config; do not ship).
   Expect `verifies=312/step`, `misses=0`, `probe_dropped=0` in the per-window
   WARNING lines. Do NOT time this run (it syncs every layer-pass).
3. **Loss canary (verify off, gate on):** identical `--warmup-datums` as the
   gates-off reference; bitwise expected (the arange substitution is exact);
   drift > 5e-3 = STOP.
4. **Steady capture (gate on, verify off):** profile one steady window (NO
   window-1 capture). The sharp rows vs the post-C capture: `aten::nonzero`
   slices >5 ms ≈ 0 (the 312 DSA-bwd drains gone — they return iff the post-C
   regime actually exposes them), `cudaStreamSynchronize` −312, plus 312
   µs-scale `cudaEventSynchronize` (the first-pass probes; the checker's
   `eventsync_a_*` rows split these by the FusedSparseAttentionFuncBackward
   parent — same as v2's, expect p50 µs, ≤1.5 s total, ≤25 >1 ms).
5. **Timed A/B:** steady tok/s/GPU, B+F+C (reference) vs B+F+C+v3, same
   datums. Report the delta with the counter lines (`kicks/hits/misses` per
   step) and the canary drift. This is a MEASUREMENT of the post-C exposure
   hypothesis — a ~0 result is a valid, publishable answer (it would mean the
   drains are still overlapped even without the dispatcher throttle).
6. **Report:** tok/s delta, canary drift, the verify-soak result, and the
   last per-window counter line to fibonacci/pascal. Any `misses` or
   `probe_dropped` > 0 in a window — STOP and report; that voids the claim.
