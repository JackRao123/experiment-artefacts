# W2 arm hang — design-lane first-pass analysis (fermi, 2026-08-10)

**Event:** the W2 timed arm (K=2, `BT_MOE_A2A_PIPELINE=2`) on box 3
(wxlg05w) HUNG in the FIRST window's backward (warmup0, 2×131072).
Reference arm clean (701). Boot healthy; gate ACTIVE+armed. py-spy: rank-6
idle in autograd backward (`forward_backward_no_pipelining`); NCCL
collective-timeout dumps at ~7 min on ranks 0/1/2/6/7 (default_pg) + PG ID
18 on rank 2. Evidence (box): `lps1062_bench/wxlg05w/w2-arm-hang-trainer_srun.log`
+ `box3_arm_log.md` + py-spy stacks (grothendieck's thread).
**Scope of this doc:** Mac-side code analysis + a discrimination plan. NO fix
patch tonight (morning severity). The T2 gate passed at gate scale/defaults,
so this is a full-shape/stack-composition-class failure until proven
otherwise.

## 0. TL;DR (updated after helmholtz's re-ranking observations)

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

## 4. The full-shape suspects (re-ranked; A is the front-runner)

The gate (T2 re-run, d88d8b7d) passed ALL cases at 2048/8192 with ckpt
on/off and fixc variants — so single-layer, gate-scale composition is
proven. What the gate cannot see (75 MoE layers + MTP + DSA + CP16 + 131k +
the real trainer loop):

- **Suspect A (FRONT-RUNNER) — FIX-C replay-restore divergence at full
  shape.** Under C′, the replay pass restores the W2 chunk metadata from the
  cache (`w2_local_counts_host` / `w2_global_counts_host`, the two W2-gated
  slots on the FIX-C entry) instead of re-D2H-ing. The chunk plan is computed
  from those host matrices; if the restore diverges from the first pass on
  ANY rank (stale entry, a keying miss at full shape, an MTP-layer
  interaction), the replay's A2A splits mismatch across ranks → collective
  hang INSIDE the first checkpoint backward's recompute — which fits the
  evidence precisely (clean forward; hang in the first backward; first
  window). It also explains the gate miss: the gate's synthetic routing +
  2048/8192 single-layer scale ran the composition green, so the diverging
  condition is full-shape-specific (a real-routing count pattern, or the
  75-chunk/mb structure — curie's correction — interacting with the cache
  keying). Code-read caveat (honest limits): the store/hit path for the W2
  slots reads consistent — host values are fresh per-pass copies
  (`maybe_move_tensor_to_cpu` allocates anew; entries store references to
  per-pass objects, no buffer aliasing), keyed `store[id(dispatcher)]` per
  layer, restored on hit. So the divergence, if real, lives in a condition
  the code read can't see from here — which is exactly why the
  discriminating experiment (§5 step 1) exists.
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

## 5. Discrimination plan

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
