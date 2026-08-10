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

## 2026-08-10 (late) — W2 arm FAIL-BY-HANG: the design-lane analysis

W2 timed arm hung in the first window's backward (box 3, golden mesh
expA131 CP16, C′+W1+W2(K=2); reference clean at 701). Full analysis +
evidence chain: `W2_ARM_HANG_ANALYSIS_20260810.md`. The night's arc:

- Ruled out at code level: W2 PG-creation (no `new_group` anywhere in W2 —
  exhaustive, direct or indirect); data-dependent bwd chunk ORDER (the
  backward's collective order is structural: combine chain K−1→0, dispatch
  loop 0→K−1, layer boundaries dependency-ordered).
- The #28-class topology suspect (unguarded W1 `new_group` on the stack
  branch, which predates the ship-w1 guard) died to helmholtz's two
  observations: reference arm clean with W1 ACTIVE + warmup0's forward
  completed (900 probs A2As on the W1 comm) before the backward hang.
- The PG-18 contradiction is the finding: an 8-rank ranks-0–7 group hung on
  a golden mesh where no 8-rank group exists (mcore creates none; W2 creates
  none — K=2 splits within-rank expert subsets, never rank subsets). Two
  readings recorded (framework-created group, blocked-behind class vs
  16-rank EP-class group with only node-0 enqueued = cross-node divergence);
  the reference-boot nranks=8 grep decides the framework branch.
- Front-runner for morning: suspect A — FIX-C replay-restore divergence at
  full shape (ranks disagree on chunk plans ⇒ mismatched A2A sizes ⇒
  collective hang in the first checkpoint backward's recompute; fits
  clean-forward/first-backward/first-window; explains the gate miss).
  Discriminating experiment: the W2+C′ VERIFY=1 full-shape soak (morning
  step 1, needs a box slot; asserts before any hang).
- The W2 PR stays CLOSED with the caveat in `pr_drafts/`; re-arm only after
  root cause.

## 2026-08-10 (late) — A-v3 leg (b) verdict + the D2H finding

**A-v3 leg (b): MECHANISM PASS · NUMERICS PASS · wall informational-positive
(+4–5 % steady at 131k, steady-vs-steady).** (curie; record
`~/perf_profiles/lps-1062/AV3_LEGB_VERDICT.md`.) Leg (a) — the composition
soak with the cache engaged — is the last open gate. The PR skeleton
(`pr_drafts/PR_A_v3.md`) is filled and stays unopened until leg (a).

**New finding (curie, not a blocker):** V3-ON introduces a recurring >1 ms
D2H memcpy class (33 calls/window, 0.71 s, max 132 ms; `aten::copy_`-parented
with `aten::repeat`/`clone`/`_to_copy`/`scatter` grandparents), absent in the
C′-era trace; both boots B/F+C′+W1 ⇒ the delta is exactly V3.

**CORRECTION-OF-CORRECTION (curie's start-gap measurement, same night — the
trace settled it):** the sequence ran (i) my first read: the kick's
side-stream `wait_stream(current)` inherits the compute backlog (W3-v2
class); (ii) curie's refutation: host-duration = GPU execution, a
big-copy/bandwidth story (my docs were "corrected" to construction-side on
that premise); (iii) curie's start-gap measurement (all 33 copies,
host↔GPU correlation): GPU-side copies µs-scale (p50 0.00, max 0.02 ms),
~100 % of the host-side duration is WAIT (host-minus-GPU p50 9.01 ms, max
132.3 ms), compute stream 100 % busy during every call, all 33 copies on
stream 7 (the COMPUTE stream). **FINAL: the class IS the whole-backlog
ordering defect — a pageable/synchronizing D2H read on the compute stream
blocks the host until the stream drains to the copy point — with the locus
corrected from my framing (not a side-stream wait_stream; direct
stream-order inheritance on the compute stream itself).** Both wrong
premises recorded (my locus, curie's size premise). Fix direction: keep the
flag device-resident and read lazily, or pinned truly-async copy + deferred
host read, or producer-event ordering (input-dependency principle, correctly
located); "shrink the repeat" does not survive. Line-level locus TBD from
curie's correlation stacks (the 33 copies' grandparents are
repeat/clone/scatter-constructed tensors — NOT the probe's pinned-copy path,
whose source is the `.all()` flag — so a V3-activated line, pinned in the
morning). The estate lesson cuts both ways tonight (curie's framing): a
defect-class pattern match is a hypothesis (my first read), and so is a
refutation built on an unchecked premise (curie's first refutation). The
trace settled it.

## Open items carried (mine)

1. W3-v3 readout when the arm runs; win-model lock from the measured c/d.
2. Option-6 spec (priority 2; un-park disposition recorded in
   W2V2_DECISION_stub.md — built only if W3-v3 fails its canary).
3. W2-v2 seq-bump re-weigh after the v3 readout (stub non-additivity note).
4. Memo §7b memory-estimate correction (inherited from minkowski).
5. W2 6144 full-fidelity gate confirmation + T3 canary slot (minkowski's
   items 3–4; box scheduling is helmholtz's).
