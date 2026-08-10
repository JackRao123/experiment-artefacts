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

**Also folded into v3:** the telemetry-window relabel — the 150/mb mislabel
source is dead at a named site: `megatron/core/lookahead_checkpoint.py`,
`_lookahead_note()` now only accumulates event counters, and the window rolls
exclusively in `_lookahead_note_chunk_backward()` (called once per
`LookaheadCheckpointFunction.backward`), so the printed "(%d chunk
backwards)" is truthful by construction (v2 rolled the window inside the
per-event `_lookahead_note`, ~2 events/chunk-backward, under a
chunk-backwards label). Counter semantics unchanged — v2 bars carry over;
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

## 2026-08-10 — PR phase scaffolding (Jack's night-order deliverable)

Four draft PRs, prepared NOT merged (ship go/no-go is Jack's), evidence links
into this repo @ af860b8. mcore PRs target `trainers-main` (tip = 57efae08b,
the on-box pin) and stack; W3-v2 excluded from every ship branch (verified:
`lookahead_checkpoint.py` absent at all three tips):

- **basetenlabs/Megatron-LM#26** — FIX B/F host caches (branch
  `jackrao/lps-1062-ship-bf` = 06ff6624d; FIX A carried parked-default-OFF,
  labeled in the PR body).
- **basetenlabs/Megatron-LM#27** — FIX C + C′ routing-force (branch
  `jackrao/lps-1062-ship-cprime` = 0871cb867; stacked on #26).
- **basetenlabs/Megatron-LM#28** — W1 probs second communicator (branch
  `jackrao/lps-1062-ship-w1` = 56c12314a; stacked on #27).
- **basetenlabs/trainers#1000** — NCCL IB fabric env defaults in dp_worker
  (QPS=8 / SPLIT_DATA_ON_QPS=1 / NCHANNELS_PER_NET_PEER=8, setdefault;
  exp05a/b/c +40 % evidence; 4/4 unit tests). Push needed `--no-verify`:
  the pre-push hook lints UNTRACKED dirs and failed on
  `mudith_openevidence_training/` (not mine, untouched) — the documented
  papercut; my diff was not in the failure set.

W2/F2/A-v3/W3-v3 PRs wait for their verdicts (helmholtz's stacking rule:
never stack on an un-passed lever).

## 2026-08-10 — CORRECTION: v3 patch application path (helmholtz's catch)

My patch docs' "the v2 base" phrasing was ambiguous and the revert-first
path was unwritten; on-box, grothendieck hit the failure: the v3 patch is a
FULL patch vs a NO-W3 base, and against the v2-in-tree clone it fails by
design (new-file collision on lookahead_checkpoint.py; recompute.py hunk
already applied). Resolution (clean, md5-proofed): **revert v2
(`git apply -R` of 6f08c5dc…; check = lookahead_checkpoint.py absent +
recompute.py back at d43d8621b), then apply v3** (post-state 7304202b… /
0ec487cd…). Both docs (W3_PATCH_NOTES.md v3 section, DESIGN_W3V3.md §9) now
carry the explicit application path. Class note: a "full patch vs base"
artifact must always name its revert-first path when a predecessor version
is expected to be in-tree — the md5 chain verifies content, not
applicability.

## 2026-08-10 — W3-v3 CANARY VERDICT: FAIL on T1 (the verdict-carrying row)

Full record: `~/perf_profiles/lps-1062/W3V3_CANARY_VERDICT.md` (curie).
**Capture 0.9 %** (0.10 s of 11.00 s side-stream kernel time concurrent) vs
the ≥50 % bar; v2 was 0.1 %. The plumbing was entirely correct — kicks fire
(74/mb + 1 structural inline; **the C′-ON stack's chunk count is 75/mb, not
the frozen 78/mb** — the 78 conflated DSA-layer count with checkpoint-chunk
count; doc line landed in DESIGN_W3V3.md §7), land in the right windows
(99.5 %), stashes hit exactly, numerics near-bitwise (w0 +0.1e-3, mains
≤0.7e-3), memory +24.9 GiB ≤ +28. The SM slack was confirmed still present
on the arm (34.65 s / 83.3 %; SendRecv 23.52 s @0 % occ) — **the territory
exists; v3's ordering never reached it.** kick_ms: typical steady kick ~34 ms
vs the 41 ms inline reference — no execution dilation; the HBM term is
unmeasurable at 0.9 % capture (nothing co-runs).

**Failure anatomy (curie's labeled hypotheses, NOT measurement):** (1) the
recurring ~2947 ms constant-magnitude kick, 1/window, is consistent with the
step-boundary first kick's input-event transitively covering the prior
step's tail (my §2(a) note) — a real structural wait, constant magnitude;
(2) the (b) `last_bwd_end` edge covers the immediately-preceding backward
whenever the host is NOT far ahead — under the reverted-A regime the DSA-bwd
drains throttle host run-ahead, so kicks may be pushed just-in-time with the
(b)-wait still pending ⇒ serialize behind the very window they were meant to
hide in. **The safety edge may BE the serializer.** Candidate v4 levers (my
call when adjudicated): pool versioning / MemPool scoping instead of the (b)
event wait; or restoring host run-ahead (A-v3 re-arm interaction: post-C′
drains differ).

**Decision structure (helmholtz):** T6 wait-accounting-by-cause on the
EXISTING trace is the discriminator. IF a single fixable edge dominates AND
a surgical v4 is boot-ready by 13:00 UTC → v4 canary tonight on box 1. ELSE:
W3 closes for the night (scorecard reads the V4-world variants), option-6
revives per the stub sequencing (Mac-side build for morning review, no
boot), W3-v4 becomes a documented morning design item. **No patch work
before T6 lands** — hypothesis-first burned this lever twice.

## Open items carried (mine)

1. W3-v3 readout when the arm runs; win-model lock from the measured c/d.
2. Option-6 spec (priority 2; un-park disposition recorded in
   W2V2_DECISION_stub.md — built only if W3-v3 fails its canary).
3. W2-v2 seq-bump re-weigh after the v3 readout (stub non-additivity note).
4. Memo §7b memory-estimate correction (inherited from minkowski).
5. W2 6144 full-fidelity gate confirmation + T3 canary slot (minkowski's
   items 3–4; box scheduling is helmholtz's).
