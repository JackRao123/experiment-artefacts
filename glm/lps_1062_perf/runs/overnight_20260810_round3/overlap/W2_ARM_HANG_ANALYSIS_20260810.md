# W2 arm hang — design-lane first-pass analysis (fermi, 2026-08-10)

**Event:** the W2 timed arm (K=2, `BT_MOE_A2A_PIPELINE=2`) on box 3
(wxlg05w) HUNG in the FIRST window's backward (warmup0, 2×131072).
Reference arm clean (701). Boot healthy; gate ACTIVE+armed. py-spy: rank-6
idle in autograd backward (`forward_backward_no_pipelining`); NCCL
collective-timeout dumps at ~7 min on ranks 0/1/2/6/7 (default_pg) + PG ID
18 on rank 2. Evidence (box): `lps1062_bench/wxlg05w/w2-arm-hang-trainer_srun.log`
+ `../results/box3_arm_log.md` + py-spy stacks (grothendieck's thread).
**Scope of this doc:** Mac-side code analysis + a discrimination plan. NO fix
patch tonight (morning severity). The T2 gate passed at gate scale/defaults,
so this is a full-shape/stack-composition-class failure until proven
otherwise.

## 0. TL;DR (final night form, after the engagement-table update)

- The hanging boot ran **FORCE-without-CACHE** (grothendieck's engagement
  table: `BT_MOE_ROUTING_REPLAY_FORCE` on, `BT_MOE_DISPATCH_REPLAY_CACHE`
  OFF) on the golden mesh (expA131, CP16, no F2). The reference ran the same
  cache state, clean — so cache-off alone doesn't hang; **the delta is W2's
  chunked path in the free-routing-replay regime at full shape.**
- **Suspect A as mechanized (FIX-C replay-restore divergence) is DEAD:** the
  cache was disabled — the store/restore path never executed.
- **Sharper code-read finding (this round):** C′ itself was INERT on the
  hanging boot. `_wrap_checkpoint_chunk_pass` no-ops unless the cache gate
  is on (recompute.py:130-131) ⇒ no checkpoint-pass frames existed ⇒ C′'s
  `_routing_force_frame()` returned None ⇒ no stash, no force — **the replay
  routed FREELY** (recomputed logits under enable_grad vs the no-grad first
  pass — exactly the ULP-divergence class C′ exists to prevent). So the
  hanging regime was W2(K=2)+W1 with free replay routing.
- The free-routing divergence is analyzed in §4-final: it makes the replay's
  counts/plan differ from the first pass's, but each pass stays internally
  consistent (counts are all-gathered within the pass), so it does NOT
  mismatch a collective by itself — the mechanism needs something more
  (§4-final candidates).
- The PG-18 fork from §4.6 stands (framework-group blocked-behind vs
  cross-node divergence); the reference-boot `nranks=8` grep decides the
  framework branch.
- **curie's formal adjudication (§4.7) lands the decisive detail:** the hung
  ALLTOALL_BASE had NumelIn wildly imbalanced (2…130,448) vs uniform
  NumelOut (65536) — **the K=2 pipeline deadlocks under routing imbalance
  the monolithic reference tolerates** (per-chunk sync/size-negotiation
  class; the gate's 3.48× synthetic skew never approached tonight's real
  per-peer skew). Repro step 1 reframed accordingly: an IMBALANCED datum mix
  in the same regime, dump preserved (§5).

## 0'. Earlier form (superseded by the engagement-table update)

- Hypothesis 3 (W2 PG-creation ordering) is **RULED OUT at code level**:
  W2-v3 calls `new_group` nowhere (verified); the only `new_group` in the
  stack is W1's.
- Hypothesis 1 as written (data-dependent bwd chunk ORDER) is **refuted at
  code level**: the W2 backward's collective order is structural, not
  data-dependent (§2). What IS data-dependent is the split SIZES — a per-rank
  split mismatch hangs without any order divergence (§4).
- The #28-class topology suspect (unguarded W1 `new_group` on the stack
  branch) is **EXPECTED DEAD** on two observations (helmholtz, §3): the W2
  reference arm ran clean on the same boot-class with W1 ACTIVE, and the arm
  completed warmup0's FORWARD (900 probs A2As on the W1 comm) before hanging
  in backward — a creation-time rendezvous corruption does not fit
  forward-works-backward-hangs on the same comm. grothendieck's 1-minute log
  answers (world size / EP span / env / PG-18 membership) close the branch
  formally.
- **Front-runner (§4, suspect A): a FIX-C replay-restore divergence at full
  shape** — the replay's restored W2 chunk metadata diverges from the first
  pass on some rank ⇒ ranks disagree on chunk plans ⇒ mismatched A2A sizes
  ⇒ collective hang inside the FIRST checkpoint backward's recompute (fits
  the evidence: clean forward, hang in the first backward, first window).
  Also explains why the synthetic both-gates-on gate missed it (gate scale +
  synthetic routing don't produce the diverging condition). The
  discriminating experiment is the W2+C′ VERIFY=1 full-shape soak (§5 step 1;
  morning item, needs a box slot).

## 1. Hypothesis 3 — PG-creation ordering: RULED OUT as written

`git grep new_group` over the W2-v3 surface (token_dispatcher.py,
moe_layer.py, experts.py): the only `new_group` call in the entire stack is
W1's second-communicator creation (token_dispatcher.py:1446 on the stack
tip). W2 creates NO process groups — its chunked A2As ride the existing EP
group. The #28 review's same-name-collision class needs ≥2 group-creation
sites or a non-world-spanning group; W2 provides neither. PG ID 18 is
therefore NOT a W2-created group: candidates are W1's second comm or an
mcore-init group (EP/TP/CP). Discriminator: the dump's PG-18 membership vs
the EP group's ranks (§5, check C).

## 2. Hypothesis 1 — is the chunked bwd A2A order data-dependent? NO (code-level)

The concern (helmholtz's phrasing): did v3 make the backward chunk issue
order data-dependent? Read from the W2-v3 code:

- **Combine reverses** (`_ChunkedCombineA2A.backward`): one Function per
  group, chained by the `combine_buf` passthrough (group 0 allocates, later
  groups receive-and-return). The unpermute consumes the LAST group's output,
  so the backward fires in the fixed chain order K−1 → … → 0 on every rank —
  the chain is structural (built at forward, identical graph on all ranks).
- **Dispatch reverses** (`_ChunkedDispatchA2A.backward`): one Function
  covering all K groups; the K reverses issue in a fixed `for g in
  range(K)` loop, then all waited. Fixed order 0 → K−1 on every rank.
- **Cross-layer:** layer L−1's combine reverses are data-downstream of layer
  L's dispatch reverse (through permute1.bwd → the residual junction → L−1's
  unpermute), so layer boundaries can't interleave differently per rank.
- **The probs reverse** rides comm 2 alone (one collective per layer-pass —
  no order to diverge).
- **No rank-dependent collective skipping:** both chunked Functions issue
  unconditionally (zero-count list entries are kept by construction — the
  T2 gate validated 0-count NCCL behavior).

So the per-comm collective ORDER is fixed by the graph structure, identical
across ranks, independent of the data-dependent chunk SIZES. A NCCL
order-divergence deadlock (the classic two-comm cross-pair) needs a
structural divergence that does not exist here. NOTE: this refutes the
mechanism as stated; it does NOT clear the sizes — see §4.

## 3. The #28-class topology suspect — EXPECTED DEAD (helmholtz's observations)

The arm ran the stack branch (`jackrao/lps-1062-r3-stack` @ 15d5679e — the
only branch carrying W2-v3). That tip's W1 code has the **UNGUARDED**
`new_group(get_process_group_ranks(self.ep_group))` (default
`use_local_synchronization=False`); the guard + local-sync fix landed only on
`jackrao/lps-1062-ship-w1` (`d794ca3d2`, post-review) and is NOT in the
stack. On a >16-rank box with EP16 (two EP groups), that unguarded creation
is the #28-class rendezvous hazard.

**Why it's expected dead (helmholtz, 2026-08-10):** (1) the W2 REFERENCE arm
ran CLEAN on the same boot-class with W1 ACTIVE (arm-check line); (2) the arm
boot completed warmup0's FORWARD — W1's comm carries 900 probs A2As in the
forward — before hanging in the backward. A creation-time rendezvous
corruption does not fit forward-works-backward-hangs on the same comm: the
hazard would wedge the FIRST probs A2A (forward), not let 900 pass. Expect
the topology discriminator (grothendieck's 1-minute log read: world size /
EP span / `BT_MOE_PROBS_A2A_COMM` in `/proc` env / PG-18 membership) to come
back 16-rank/EP-spans-world and close this branch formally. (The guard stays
on ship-w1 regardless — it's the correct ship form for multi-EP-group
topologies; it just isn't tonight's mechanism.)

## 4. The full-shape suspects (SUPERSEDED by §4-final — the engagement-table
   update killed suspect A and changed the regime reading)

The gate (T2 re-run, d88d8b7d) passed ALL cases at 2048/8192 with ckpt
on/off and fixc variants — so single-layer, gate-scale composition is
proven. What the gate cannot see (75 MoE layers + MTP + DSA + CP16 + 131k +
the real trainer loop):

- **Suspect A — DEAD (engagement-table update, 2026-08-10 late):** the
  hanging boot ran FORCE-WITHOUT-CACHE — the replay cache was DISABLED, so
  the store/restore path this suspect needed never executed. (Earlier drafts
  of this section had it as the front-runner — "FIX-C replay-restore
  divergence at full shape ⇒ ranks disagree on chunk plans ⇒ mismatched A2A
  sizes ⇒ hang in the first checkpoint backward's recompute". Record kept
  for provenance; the mechanism as mechanized had no code path to run. See
  §4-final for the re-ranked field.)
- **Suspect B — a zero/degenerate count pattern at 131k the gate's cases
  didn't produce** (e.g. a whole expert-group with zero rows on a rank, or
  the MTP layer's dispatcher arming W2 with a different local-expert count).
  The W2 Functions issue unconditionally (no rank-dependent skipping — §2),
  but a degenerate split set is the kind of thing only full-shape data
  produces. Discriminator: the hang dump's pending-collective sizes vs the
  expected per-group byte counts from the arm's routing.
- **Suspect C — a non-dispatcher collective** (the default_pg timeouts on
  ranks 0/1/2/6/7 may be the wedge, with PG 18 a bystander): CP16 AG/RS or
  the DSA-bwd path at full shape interacting with the W2 arm's timing.
  Discriminator: the pending collectives' PG + sizes in the dump.

## 4.5. The PG-18 evidence (grothendieck/helmholtz, 2026-08-10) — and a
       premise correction

The stuck group: **PG 18 / GUID 186, watchdog seq 619 enqueued-not-completed
on EXACTLY ranks 0–7 (the leader node) — an 8-rank group.** World 16; EP16
spans all ranks; W1 comm ACTIVE + created-over-16 (boot marker). This closes
the #28-class branch formally (W1's comm is 16-rank and worked through the
forward).

**Premise correction (fermi, code read — direct):** the W2 chunked path does
NOT run over an 8-rank group. K=2 splits the 16 LOCAL EXPERTS on every rank
into 2 groups of 8 EXPERTS; both chunked Functions
(`_ChunkedDispatchA2A`/`_ChunkedCombineA2A`) take `self.ep_group =
pg_collection.ep` — the full 16-rank EP group — for every chunk (that
every-rank-in-every-chunk balance is the design's core property, memo §2.3).
So PG 18 is neither W1's second comm (16-rank) nor the W2 chunked A2As' comm
(16-rank). It is a THIRD group, outside the W1/W2 comm structure.

**What 8-rank ranks-0–7 group can exist at world 16:** enumerated the stack's
group-creation sites (mcore parallel_state + the bridge + the trainer
server): at golden TP1/PP1/EP16/CP16/DP1 there is NO 8-rank group (no
hierarchical CP configured — `hierarchical_context_parallel_sizes` unset;
EP/TP-EP/CP groups are 16-rank; TP/DP trivial). An 8-rank ranks-0–7 group is
exactly a **CP8 ring (one replica's CP group) if the arm's mesh was
EP16/CP8/DP2 (expB-class), not golden CP16.** → the two settling questions
(both one-line log reads for grothendieck): (a) PG 18's `group_desc` in the
dump's group table (names the group directly); (b) the arm config's
`context_parallel_size` (16 vs 8) — and if 8: was box 3's server/src carrying
the F2 patch (DP2 without F2 is the deadlock class the F2 PR fixes), and note
the reference read (701 ≈ the golden-CP16 anchor class) would then be a
mesh-mismatched comparison to flag.

**If the mesh was expB (CP8/DP2):** PG 18 = replica 0's CP8 ring (attention
AG/RS). A CP-ring timeout on replica 0 is the DOWNSTREAM symptom of a
cross-replica desync — a rank stuck in a 16-rank EP collective blocks its
CP ring's progress. That re-frames but does not displace suspect A: the
first window's FORWARD completed (per-replica partition counts matched — the
textbook F2 forward deadlock needs a count mismatch, so F2-class fits
imperfectly), and the backward hang points at a backward-phase desync — the
replay/restore divergence (suspect A) remains the front-runner, now with the
mechanism: the replay's A2A on the 16-rank EP group mismatches across
replicas → replica-1 ranks stuck there → replica 0's CP8 ring (PG 18) times
out waiting.

## 4.6. The contradiction IS the finding (helmholtz, closing the night) —
       and the two readings of "8-rank ranks 0–7"

Confirmed facts (grothendieck's dispatch record + log reads, authoritative):
the arm ran the GOLDEN mesh (expA131, CP16, no F2, box-3 server/src
pristine; the reference ran the same config — the A/B was valid; the
mesh-mismatch branch is CLOSED). So an 8-rank ranks-0–7 group existed and
hung on a mesh where the enumeration says none should.

**Code-side verdict (fermi, exhaustive):** the W2 patch creates NO process
group, directly or indirectly. The full W2 diff surface (token_dispatcher.py,
moe_layer.py, experts.py; `_ChunkedDispatchA2A`, `_ChunkedCombineA2A`,
`_W2ChunkPlan`, `_w2_compute_chunk_plan`, `_W2PipelineConfig`) contains no
`new_group`, no `new_subgroups`, no `get_*_subgroup`, no comm-split, no
group-splitting utility — every `group=` argument is the 16-rank
`pg_collection.ep` (or W1's 16-rank second comm for the probs path). And at
golden TP1/PP1/EP16/CP16/DP1 on 16 ranks, mcore's parallel_state creates no
8-rank group either (TP/DP/ETP trivial; CP/EP/TP-EP/TP-CP 16-rank; no
hierarchical CP configured; partial-expert-DP groups need
num_distributed_optimizer_instances > 1). grothendieck's "ranks 0–7 host
experts 0–7" reading also fails on the design: at EP16 every rank hosts
local experts 0–7 — a K=2 expert-group split is a within-rank expert subset,
not a rank subset, so no W2 structure maps to an 8-rank group under ANY
reading.

**So PG 18 is framework-created outside the W2/dispatcher surface** — OR the
"8-rank" reading itself is the artifact. Two readings, because the NCCL dump
files did not survive the kill and the membership came from the watchdog
lines:

- **Reading 1 (membership):** PG 18 is a genuinely 8-rank group (ranks 0–7).
  Then it is framework-created (trainer/bridge/DCP/TE/optimizer path — none
  found in the W2 or dispatcher code), and the W2 backward hung ON it in the
  blocked-behind class: the real wedge is elsewhere and this group's
  collective was enqueued behind it. helmholtz's cheap discriminator:
  **grep the REFERENCE boot for any nranks=8 NCCL comm-init line** — if
  PG-18-class groups exist in the clean reference boot, they're
  framework-created and W2 merely hung on one; if only the arm boot has one,
  something in the armed path creates it lazily (the code read says the W2
  patch cannot — so that outcome would indict a stack-composition
  interaction, not the W2 diff).
- **Reading 2 (enqueue-list):** PG 18 may be a 16-rank group (EP-class) on
  which only ranks 0–7 (node 0) ENQUEUED seq 619 — "enqueued-not-completed
  on exactly ranks 0–7" then reads as a cross-NODE divergence: node 0 posted
  the collective and node 1 never reached it. That reading points straight
  back at suspect A (a replay/restore divergence that splits the world
  between the nodes) — the two nodes' backward streams diverged. (The dump
  is gone, so this stays ambiguous until a re-run preserves it — the
  verify-on soak asserts BEFORE any hang, which is why it's step 1
  regardless.)

**Night's close on W2 (helmholtz):** the W2 row stays FAIL-by-hang with this
as the sharpest open question + the VERIFY=1 soak as morning step 1.

## 4-final. The free-routing regime (the actual hanging configuration) and
            the re-ranked field

The engagement table (grothendieck, authoritative): the hanging boot ran
**FORCE-on / CACHE-off**. Two consequences landed by code read:

1. **Suspect A is dead** (the cache's store/restore never executed).
2. **C′ was INERT on the hanging boot** — `_wrap_checkpoint_chunk_pass`
   no-ops unless `BT_MOE_DISPATCH_REPLAY_CACHE=1` (recompute.py:130-131), so
   no checkpoint-pass frames existed, so `_routing_force_frame()` returned
   None on every pass: no stash, no force. **The replay routed FREELY** —
   recomputed logits under `enable_grad` vs the no-grad first pass, the exact
   ULP-divergence class C′ was built to prevent. (Regime note for the
   estate: any FORCE-without-CACHE arm tonight had the force inert; the W2
   arm's A/B vs the reference stays valid for the hang — both boots shared
   the regime, the delta is W2.)

What free replay routing does and does NOT do (the honest bound of the
code read): the replay's recomputed logits diverge from the first pass at
ULP level ⇒ boundary tokens flip top-k ⇒ the replay's counts and chunk plan
differ from the first pass's. BUT each pass is internally consistent (the
counts are all-gathered within the pass), so no single collective mismatches
across ranks from this alone — and the per-pass W2 state (`_w2_pass`, plans,
buffers) is created and cleared per pass (verified: set in token_dispatch,
cleared in combine_postprocess with the rest of the forward state), so no
mixed-pass contamination. **The hang mechanism needs something more than
free routing.** Re-ranked candidates:

- **(a) A full-shape data condition in the fresh-compute path** (suspect B
  stands): something in the split-size/chunk-plan computation consuming
  rank-local or non-deterministic state that only 131k real routing produces
  (the gate's synthetic cases at 2048/8192 can't see it). Code-read limit
  stated: the plan computation is "pure host math" from all-gathered counts
  (consistent across ranks by construction) — so this candidate needs a
  condition the code read can't see; the repro (§5) is the instrument.
- **(b) The blocked-behind reading of PG 18** (§4.6 reading 1): PG 18 is a
  framework-created 8-rank group and the W2 backward hung ON it — the real
  wedge is elsewhere (a host-side stall or a different stuck collective);
  the reference-boot `nranks=8` grep decides whether such groups exist in
  the clean boot at all.
- **(c) The cross-node divergence reading of PG 18** (§4.6 reading 2):
  node-0-only enqueues on a 16-rank group = the two nodes' backward streams
  diverged — under the free-routing regime this needs a node-level
  divergence source (none found at code level; the repro decides).
- **(d) A W2×free-routing interaction not visible at code level** — e.g. a
  per-pass structure in the chunked path that assumes pass-invariant counts
  (none found: the plan is recomputed per pass) or a grad-mode-only code
  path in the chunked Functions (none: the Functions' forward is
  grad-agnostic). Recorded for completeness; the repro discriminates.

## 4.7. curie's formal adjudication (2026-08-10, late) — the decisive
       mechanism detail

The hung collective (watchdog seq 619, all-enqueued-none-completed, 600 s
watchdog) was an `ALLTOALL_BASE` with **NumelIn wildly imbalanced across
ranks (2, 4, 134, 353, 21287, 105000, 113247, 130448 elements) vs uniform
NumelOut (65536)**. Verdict detail: **the K=2 pipeline deadlocks under
routing imbalance that the gate-OFF (monolithic) reference tolerates** — the
per-chunk sync/size-negotiation class. Full evidence:
`runs/overnight_20260810_round3/verdicts/W2_ARM_ADJUDICATION.md`.

Reads on the numbers (fermi): uniform NumelOut 65536 = 8,192 tok/rank/mb ×
topk 8 (a rank's local token×topk count at 131k×d4) and imbalanced NumelIn =
the expert-work distribution — the shape of a **combine-direction A2A**
(each rank receives its own tokens back, uniform; sends expert outputs,
skewed). The gate record is consistent: the T2 gate's engineered imbalance
(3.48× expert-0 skew) passed at 2048/8192 — tonight's real-routing
full-shape skew is orders of magnitude wider, and the gate's synthetic cases
never produced a per-peer 2-vs-130k split. The reference tolerating the same
skew (clean 701) localizes the sensitivity to the chunked path's per-peer
split structure, not the imbalance itself.

**Residual tension (honest, still open):** 8 NumelIn values = 8 ranks — so
EITHER the group is genuinely 8-rank (which maps to NO W2/W1 comm at golden
mesh per the §4.5/§4.6 enumeration — the W2 chunked A2As run on the 16-rank
`ep_group`) OR the watchdog printed only the 8 enqueuing ranks of a 16-rank
group (the §4.6 reading-2 fork: node-0 enqueued, node-1 never posted =
cross-node divergence). The group_desc never survived (dump lost on the
kill). The dump-preserving repro (§5 step 1) settles both the mechanism and
the group identity.

Design-level consequence for the W2 PR (recorded, not fixed tonight): the
chunked path has a deadlock class under wide routing imbalance that the
monolithic path tolerates — a ship blocker until root-caused and covered
(the gate needs a wide-skew case at the real per-peer scale; the fix
direction depends on where the size negotiation actually wedges — see the
reframed repro, §5).

## 5. Discrimination plan (morning; repro guidance CHANGED per the
   engagement-table update)

**Step 1 (the repro — REFRAMED per curie's adjudication):** drive an
**IMBALANCED datum mix** in the same regime (W2(K=2)+W1, FORCE-on/CACHE-off,
golden 131k×d4) — a wide-skew routing pattern at the real per-peer scale
(the hung collective's NumelIn spanned 2…130,448; the gate's 3.48× synthetic
skew never approached this) — with the NCCL dump PRESERVED this time (the
kill tonight lost it). The B_custmix customer-histogram recipe (the F2 (b)
datum set) is the closest existing driver; a synthetic hot-expert mix is the
sharper instrument. A repro confirms the imbalance-deadlock class; the
preserved dump settles the PG-18 fork (membership vs enqueue-list) and
localizes the wedge (which chunk, dispatch vs combine, zero-corner vs
size-mismatch).

**Step 2 (the regime discriminator):** the intended full-C′ regime (cache ON
⇒ frames exist ⇒ force live + cache live), same shape. A hang in exactly one
regime is itself the discriminator: hangs-only-with-cache-off ⇒ the
free-routing divergence is load-bearing (and C′-on is the mitigation already
built); hangs-in-both ⇒ regime-independent, pointing at (b)/(c).

**Step 3 (only if step 1 reproduces):** the W1-off W2 arm (probs on the EP
comm) isolates the two-comm composition; and a W2-arm at the gate's
2048/8192 shape with REAL routing data (if portable) bisects
full-shape-data vs full-shape-structure.

Log reads (no boot): the reference-boot `nranks=8` grep (step 0b, §4.6);
the arm log's W2 per-window counters before the hang (did the first
forward's `{dispatch_issues, combine_issues, waits}` advance as expected?);
py-spy on MORE ranks if a re-run hangs (rank 6's "idle in autograd backward"
needs its wait target named).

**Step 0 (log reads, no boot — grothendieck):** (a) DONE — mesh confirmed
golden CP16, no F2 (§4.6); (b) **the reference-boot grep (helmholtz's
discriminator):** any nranks=8 NCCL comm-init lines in the clean reference
boot — present-there-too ⇒ framework-created (blocked-behind class);
arm-only ⇒ lazy creation in the armed path (the code read says the W2 patch
cannot — that outcome indicts a stack-composition interaction, not the W2
diff); (c) PG 18's `group_desc` if any dump survives a future re-run
(settles membership-vs-enqueue-list, §4.6 reading 1 vs 2).

**Step 1 (the discriminating experiment — morning item, needs a box slot):
the W2+C′ VERIFY=1 full-shape soak.** Boot the arm's exact config at 131k
with W2 armed + `BT_MOE_DISPATCH_REPLAY_CACHE=1` +
`BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1` (verify mode re-derives the metadata
on every replay and asserts bitwise equality against the restored cache —
including the W2 slots via `_verify_host_metadata`). Outcomes: a verify
ASSERTION names the diverging field (suspect A confirmed, with the field
localized) before any collective can hang; a clean verify soak clears the
replay-restore path and re-ranks suspect B/C (then read the dump's pending
collectives per steps B/C). NOTE: verify mode syncs every layer-pass — it is
a correctness instrument, never a timed run. If the hang reproduces EVEN
under verify (the assert is on the hit path's comparison, so a hang before
the assert would mean the divergence is upstream of the restore — e.g. in
the first pass's own store), that ordering evidence itself localizes the
defect.

Then, only as needed (log reads, no boot):
- **A. Arm topology + env (1 min, grothendieck collecting):** world size,
  EP/CP/DP, whether W1 was armed, the W1 armed line. → closes §3 formally.
- **B. The timeout dump's pending collectives:** PG, sizes, per-rank order.
  Match sizes against the W2 signatures (dispatch chunks ≈ half the
  monolithic 805 MB; combine likewise; probs = 66 KB on comm 2). The FIRST
  mismatched collective across ranks localizes the wedge.
- **C. PG 18's membership** (the dump's group table).
- **D. Where in the backward:** the log's recompute/FIX-C markers around the
  hang — replay (suspect A) vs graph-backward (suspect B/C).
- **E. The W2 per-window counters** in trainer_srun.log before the hang:
  did the first forward's `{dispatch_issues, combine_issues, waits}` advance
  as expected (K=2 per layer-pass)?
- **F. Fallback re-arm (helmholtz's call):** a W1-off W2 arm (probs on the
  EP comm) isolates the two-comm composition entirely.

## 6. What this is NOT (scope guards)

- Not a T2-gate redo: the gate passed at its scale; the fidelity caveat
  (2048/8192, 6144 confirmation queued) is already recorded.
- Not a W2-design refutation: nothing in tonight's evidence indicts the
  chunked-pipeline design itself; the two live branches are a
  stack-composition hazard (§3, fix already exists on ship-w1) or a
  full-shape data condition (§4).
- Not a fix patch: morning severity per helmholtz.
