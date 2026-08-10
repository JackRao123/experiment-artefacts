# ESTATE NOTES — fermi (design owner, LPS-1062, from 2026-08-10)

Running log, continuing minkowski's estate (`ESTATE_NOTES_minkowski.md`).
Orchestrator: helmholtz. Box: grothendieck (per helmholtz's re-org).
Verification: curie. Standing rule carried from minkowski: **inherited state
claims are unverified until re-proven.**

---

## 2026-08-10 — W3-v3 DESIGNED + PATCHED (input-dependency-only kick ordering)

**Trigger:** boltzmann's SM-slack gate came back POSITIVE (curie-confirmed):
83.5 % of bwd-phase kernel time <10 % occupancy (~35.9 s slack/step),
SendRecv windows ~0 % occ. The v2 serialization was ORDERING, not SM
saturation — the W3-v3 gate's "pass" branch. helmholtz reprioritized the
design lane: W3-v3 first, option-6 second.

**Design:** `DESIGN_W3V3.md`. Core change: the kick's
`side_stream.wait_stream(current)` (whole-backlog — it transitively chained
kick(L−1) behind kick(L)'s consume-wait on the compute tail, and behind
whatever the host had run ahead) is replaced by two event waits: (a) the
chunk's `input_event` (recorded at first-pass registration) and (b) the
registry's `last_bwd_end` event (recorded at each chunk backward's end — the
allocator reuse edge). The distribute-saved-activations gather moved inside
the side-stream context (the two events don't cover a compute-stream gather
pushed at kick time). Full cross-stream edge audit (E1–E8) and the in-flight
depth bound (≤2 kicked graphs by construction — the v2 serialization was
never the cap) are in the design doc; both carry explicit canary bars per
helmholtz's considerations.

**Patch:** `patches/w3-lookahead-recompute-v3.patch`, md5
**05dda37f68f3113747a812e357d9b552**, 638 lines (lookahead_checkpoint.py NEW
588 + recompute.py hunk byte-identical to v2). Verified: `git apply --check`
clean vs the v2 base (57efae08b + recompute.py blob d43d8621b); applied tree
reproduces the Mac tree byte-for-byte (lookahead_checkpoint.py 7304202b…,
recompute.py 0ec487cd…). CPU suite 40/40 (v2 sections + new sec8 ordering
guards: no `wait_stream` in the module, exactly the two `wait_event` edges,
gather-inside-context, both event producers pinned by AST).

**Also folded into v3:** the telemetry-window relabel (window now rolls on
chunk backwards; the v2 events-window/"chunk backwards" mislabel caused the
false 150/mb reading — counter semantics unchanged, v2 bars carry over);
per-kick `kick_ms_avg/max` CUDA-event telemetry (the in-log dilation signal);
`BT_MOE_LOOKAHEAD_TRIM_EVERY=N` allocator-trim knob (default OFF; the
16k-enablement knob; global empty_cache at the microbatch sweep — churn cost
is canary-measured before any reliance). 16k×d32 stays HARD OFF.

**Found at regeneration (v2 patch hygiene):** v2's recompute.py `index` line
post-image hash (3c45f3695) is STALE — it names the v1-era keyword-form blob;
the v2 hunk CONTENT is the correct all-positional form (true post-image
3c9babb34). Cosmetic for plain `git apply`; recorded in W3_PATCH_NOTES.md so
nobody "verifies" v2 by its index line.

**Win model (unchanged structure, new risk term):** −8…−12 s/step at 60–80 %
capture; the capture risk is now HBM-bandwidth contention (curie's caveat:
occupancy = co-residency feasibility, NOT bandwidth), measured in the canary
frame (bwd-kernel dilation per occupancy bucket + kick_ms + consume-stall),
not pre-promised. Break-even analysis in DESIGN_W3V3.md §6: the win goes
negative only if co-running nearly doubles bwd compute time — the realistic
risk is a smaller win.

**Canary frame:** curie pre-registers off DESIGN_W3V3.md §7 (log bars
unchanged + trace bars: ≥50 % side-stream concurrency, consume-stall ≤ ~10
ms/window, dilation report, peak-mem re-measure ≤ +28 GiB vs C′-on — v2's
+25.8 GiB does NOT carry over as a number). Box slot: helmholtz's call
(likely box 1 after the A-v3 arm). Nothing booted from the design lane.

## Open items carried (mine)

1. W3-v3 readout when the arm runs; win-model lock from the measured c/d.
2. Option-6 spec (priority 2; un-park disposition recorded in
   W2V2_DECISION_stub.md — built only if W3-v3 fails its canary).
3. W2-v2 seq-bump re-weigh after the v3 readout (stub non-additivity note).
4. Memo §7b memory-estimate correction (inherited from minkowski).
5. W2 6144 full-fidelity gate confirmation + T3 canary slot (minkowski's
   items 3–4; box scheduling is helmholtz's).
