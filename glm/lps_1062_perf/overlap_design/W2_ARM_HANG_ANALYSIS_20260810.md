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

## 0. TL;DR

- Hypothesis 3 (W2 PG-creation ordering) is **RULED OUT at code level**:
  W2-v3 calls `new_group` nowhere (verified); the only `new_group` in the
  stack is W1's.
- Hypothesis 1 as written (data-dependent bwd chunk ORDER) is **refuted at
  code level**: the W2 backward's collective order is structural, not
  data-dependent (§2). What IS data-dependent is the split SIZES — a per-rank
  split mismatch hangs without any order divergence (§4, suspect B).
- **Prime suspect, topology-dependent (§3):** the stack branch the arm ran
  (`15d5679e`) predates the ship-w1 `new_group` guard (`d794ca3d2`). IF the
  arm ran on >16 ranks with EP16 (EP group NOT spanning the world — e.g. a
  4-node box), W1's unguarded second-communicator creation is the #28-class
  rendezvous hazard, and it fits the evidence (non-default PG 18 timeout +
  multi-rank default_pg dumps + one rank idle). **First discriminator: the
  arm's world size / EP span, and PG 18's identity.**
- If the arm was 16-rank golden: the code read yields no order-divergence
  mechanism (§2), which points at a full-shape-only condition the gate can't
  see (§4) — the FIX-C replay restore at full shape is the sharpest.

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

## 3. Prime suspect if the arm ran >16 ranks: the unguarded W1 `new_group`

The arm ran the stack branch (`jackrao/lps-1062-r3-stack` @ 15d5679e — the
only branch carrying W2-v3). That tip's W1 code has the **UNGUARDED**
`new_group(get_process_group_ranks(self.ep_group))` (default
`use_local_synchronization=False`); the guard + local-sync fix landed only on
`jackrao/lps-1062-ship-w1` (`d794ca3d2`, post-review) and is NOT in the
stack.

torch's contract: with `use_local_synchronization=False`, EVERY process must
enumerate the same subgroups in the same order. On 16-rank EP16 the single
EP group spans the world — safe (all tonight's 2-node arms). On a >16-rank
box with EP16 (e.g. 32 ranks: two EP groups), the two EP groups' creations
cross-talk the same-name store rendezvous → hang or a corrupt communicator —
**and the evidence fits**: a non-default PG (ID 18) timing out on one rank
while default_pg collectives time out elsewhere, and one rank idle in
backward. This is exactly the hazard the #28 review flagged for the 4-node
DP4 topology.

**This is checkable in one minute from the arm log:** the arm's world size
and EP span (the config json + the boot's rank lines), and whether
`BT_MOE_PROBS_A2A_COMM=1` was in the env (`/proc/<pid>/environ` per house
rule). If EP did not span the world with W1 armed: this is the mechanism,
and the fix already exists (ship-w1's guard refuses to arm W1 there — the
arm would have fallen back, loudly, instead of hanging). If 16-rank golden:
rule this out and go to §4.

## 4. If 16-rank golden: the full-shape-only suspects (ranked)

The gate (T2 re-run, d88d8b7d) passed ALL cases at 2048/8192 with ckpt
on/off and fixc variants — so single-layer, gate-scale composition is
proven. What the gate cannot see (75 MoE layers + MTP + DSA + CP16 + 131k +
the real trainer loop):

- **Suspect A — FIX-C replay restore divergence at full shape.** Under C′,
  the replay pass restores the W2 chunk metadata from the cache
  (`w2_local_counts_host` / `w2_global_counts_host`) instead of re-D2H-ing.
  If the restored plan diverges from the first pass's on ANY rank (stale
  entry, a keying miss at full shape, an MTP-layer interaction), the
  replay's A2A splits mismatch across ranks → collective hang INSIDE the
  checkpoint backward's recompute — which fits "hung in backward, first
  window, full stack" precisely. The gate exercised ckpt+fixc at gate scale,
  so this needs a full-shape-specific trigger (e.g. a count pattern at 131k
  real routing that the synthetic gate cases don't produce, or the
  75-chunk/mb structure — curie's correction — interacting with the cache
  keying). Discriminator: the verify-mode soak
  (`BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1`) re-derives and asserts bitwise
  per replay — a verify-on boot reproduces-or-clears this in one run.
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

## 5. Discrimination plan (box-side, ~30 min, no new boot needed for most)

- **A. Arm topology + env (log read, 1 min):** world size, EP/CP/DP, whether
  W1 was armed (`BT_MOE_PROBS_A2A_COMM` in `/proc/<pid>/environ`), whether
  the W1 "armed — second communicator" line appears. → settles §3.
- **B. The timeout dump's pending collectives:** PG, sizes, and per-rank
  order. Match sizes against the W2 signatures (dispatch chunks ≈ half the
  monolithic 805 MB; combine likewise; probs = 66 KB on comm 2). The FIRST
  unmatched/mismatched collective across ranks localizes the wedge. →
  discriminates §4's suspects; a probs-sized pending op on PG 18 points at
  §3/W1; a dispatch-sized mismatch across ranks points at §4-A/B.
- **C. PG 18's membership** (the dump's group table): == EP-group ranks ⇒
  W1's comm or an mcore EP subgroup; ⊂ EP ⇒ something else.
- **D. Where in the backward:** the log's recompute/FIX-C markers around the
  hang — replay (suspect A) vs graph-backward (suspect B/C).
- **E. The W2 per-window counters** in trainer_srun.log before the hang:
  did the first forward's `{dispatch_issues, combine_issues, waits}` advance
  as expected (K=2 per layer-pass)? Silence pattern localizes the stall.
- **F. If a re-run is warranted (helmholtz's call, NOT tonight):** the
  verify-on soak (suspect A) and/or a W1-off W2 arm (isolates the two-comm
  composition entirely — W2 without W1 puts probs on the EP comm, removing
  comm 2 from the picture).

## 6. What this is NOT (scope guards)

- Not a T2-gate redo: the gate passed at its scale; the fidelity caveat
  (2048/8192, 6144 confirmation queued) is already recorded.
- Not a W2-design refutation: nothing in tonight's evidence indicts the
  chunked-pipeline design itself; the two live branches are a
  stack-composition hazard (§3, fix already exists on ship-w1) or a
  full-shape data condition (§4).
- Not a fix patch: morning severity per helmholtz.
