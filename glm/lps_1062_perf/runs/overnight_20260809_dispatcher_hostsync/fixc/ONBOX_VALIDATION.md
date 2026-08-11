# FIX C on-box validation recipe (verify-mode soak first, then timed A/B)

Prereqs: apply `fixc.patch` on top of the current on-box checkout (B+F
applied, A parked). The two touched files are byte-identical between the Mac
pin and the on-box pin, so the patch applies cleanly:

```
cd /root/.cache/user_artifacts/trainers_main/server/vendor/megatron-bridge/3rdparty/Megatron-LM
git apply /path/to/fixc.patch   # touches only recompute.py + token_dispatcher.py
```

1. **Boot smoke (gate on):** `BT_MOE_DISPATCH_REPLAY_CACHE=1` + usual B/F env;
   confirm in trainer_srun.log: `... ACTIVE`, then `armed — first
   replay-metadata entry stored`, then `first replay cache hit ...`. No armed
   line = inert — STOP (check the wrap reached chunk_runner).
2. **Verify soak (parity prerequisite, full run):** add
   `BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1` for a full soak at the 4×131k
   bench shape. Every replay recomputes and asserts bitwise equality with
   the cached metadata; any mismatch raises RuntimeError (loud stop).
   Expect: `verifies=300/step`, `misses=0`, `shape_mismatches=0` in the
   per-window WARNING lines. Do NOT time this run (it pays full cost).
3. **Loss canary (verify still on):** identical `--warmup-datums` as the
   gates-off reference; drift must be ≤ 5e-3 (bitwise expected). Then drop
   VERIFY.
4. **Steady capture (gate on, verify off):** profile one steady window (NO
   window-1 capture) and run
   `python check_acceptance.py TRACE --profile post-patch-BFC-4mb131k`.
   Sharp rows: `eventsync_dispatcher_replay_calls ≈ 0`,
   `eventsync_dispatcher_fwd_calls = 300`,
   `dispatcher_allgather_replay_calls ≈ 0` (calibrate the allgather metric
   on a baseline capture first: `--profile baseline-4mb131k --json` should
   read 300/300 fwd/replay).
5. **Timed A/B:** steady tok/s/GPU, gates B+F (reference ~715) vs B+F+C,
   same datums; also watch the dispatcher eventSync CPU row (23.7 s replay
   portion should vanish; fwd ~9.9 s runahead-inflated remains).
6. **Report:** tok/s delta, canary drift, checker PASS/FAIL, and the last
   per-window counter line to pascal/fibonacci. If misses or
   shape_mismatches are nonzero in any window — STOP and report; that voids
   the win claim.
