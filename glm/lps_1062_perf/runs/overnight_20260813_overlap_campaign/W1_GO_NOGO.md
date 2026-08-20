# W1 GO/NO-GO — PP2/CP8/EP8 (2×8 B300, 16 ranks)

**Author:** hausdorff (for fermi) · **Date:** 2026-08-13 · **Mode:** source-only, Mac-side
**Verdict: ARM-AFTER-VALIDATION** — the mission topology is **multi-EP-group (2 groups)**,
so PR #28's `new_group` fix is **load-bearing and not sufficient as written**: the PR's
arm-time guard *disarms W1 entirely* on this topology. Arming requires (a) a small port of
PR #28 onto the campaign mcore pin, (b) a deliberate guard-relax patch, (c) the on-box
multi-group canary defined in §3. No code change is needed for the gate-OFF path (byte-
identical to upstream; zero risk to carry the ported patch dormant).

---

## 1. EP-group count of PP2/CP8/EP8 — derived, not asserted

### 1.1 Inputs

- Mission config `pp2cp8ep8/configs/trainer_pp2cp8ep8_131k.json`:
  `tensor_parallel_size=1, pipeline_parallel_size=2, context_parallel_size=8,
  expert_parallel_size=8, expert_tensor_parallel_size=1` on 2×8 B300 → **world = 16**.
- Campaign branch `jackrao/lps-1062-pp2cp8ep8` @ b8d868ff pins megatron-bridge @ 20fcf2ea
  → Megatron-LM @ **57efae08b** (all file/line refs below are at that pin).
- Init path (trainers server never overrides the defaults — verified by grep):
  `use_decentralized_pg=False` (bridge `config.py:139`) → the bridge calls mcore
  `parallel_state.initialize_model_parallel` (`bridge/training/initialize.py:808`), with
  `order="tp-cp-ep-dp-pp"` (`initialize.py:820`; `use_tp_pp_dp_mapping` defaults False,
  mcore `training/config/common_config.py:105`).
  (Even if the HyperCommGrid path were used, its expert grid `[etp=1, ep=8, edp=1, pp=2]`
  with `create_pg(["ep"])` yields the same 2 groups — `initialize.py:481-490`.)

### 1.2 The construction (mcore `parallel_state.py` @ 57efae08b)

- `data_parallel_size = world // (tp·pp·cp) = 16 // 16 = 1` (line 738).
- Expert rank generator (lines 780–801): `RankGenerator(tp=etp=1, ep=8,
  dp=expert_dp, pp=2, cp=1, order='tp-cp-ep-dp-pp')` where
  `expert_dp = world // (etp·ep·pp) = 16 // 16 = 1` (line 786). CP is *not* in the expert
  generator — `RankGenerator` forbids ep>1 ∧ cp>1 in one generator (lines 452–455); the
  CP ranks are exactly what the EP dimension spans.
- EP groups are created one per entry of `expert_decoder_rank_generator.get_ranks('ep')`
  (lines 1170–1182), via `generate_masked_orthogonal_rank_groups` (line 250): mask selects
  the `ep` dim; group count = world / ep = **2**.

### 1.3 Executed enumeration

I extracted `RankGenerator` + `generate_masked_orthogonal_rank_groups` **verbatim from the
pinned source** and ran them (pure Python; harness at
`/var/folders/.../w1gono/derive_ep_groups.py`):

```
MISSION (world=16, tp=1, cp=8, pp=2, ep=8, etp=1)  → dp=1, expert_dp=1
  EP group count = 2
    EP group 0: [0..7]    (= PP stage 0 = the 8 CP ranks of node 0)
    EP group 1: [8..15]   (= PP stage 1 = the 8 CP ranks of node 1)
  PP groups default-gen == expert-gen: [[0,8],[1,9],...,[7,15]]  (assert :809 holds)

Control — Aug-10 W1 validation arms (EP16/CP16, PP1, world=16):
  EP group count = 1  (single group spans the world)  ← why those arms were unaffected

Control — PR #28 review's hang case (world=32, EP16, 4-node):
  EP group count = 2
```

The dispatcher consumes this per-stage group directly:
`token_dispatcher.py:78  self.ep_group = pg_collection.ep`.

**Conclusion: PP2/CP8/EP8 on 16 ranks has 2 EP groups → multi-EP-group.** The campaign's
standing caveat (A2A_EXPOSURE_DECOMPOSITION §3, SCORECARD W1 row) is confirmed by
derivation. Note the groups are the two contiguous 8-rank halves — i.e. **each EP group is
intra-node** on this box, which matters for §3's risk assessment.

## 2. PR #28 audit (basetenlabs/Megatron-LM #28, head d794ca3d, stacked on C′ #27)

What the PR contains:

1. **Deferred-wait a2a machinery** (`mappings.py`): `_AllToAllDeferredWait` (issue async,
   stash work handle on the output tensor) + `wait_deferred_a2a`. Group-agnostic — correct
   for any group passed in. Backward = reverse a2a on `ctx.group`, issued+waited inline.
2. **Dispatcher arm + callsite** (`token_dispatcher.py`): second communicator created once
   (class-level singleton) at first MoE-layer `__init__`; forward issues tokens a2a (EP
   comm) + probs a2a (comm 2) back-to-back, waits both after shared-expert fc1.
3. **The review fix** (commit d794ca3d, the multi-EP-group fix):
   - creation via `torch.distributed.new_group(ep_ranks, use_local_synchronization=True)`
     — the torch-contract-safe path for non-world-spanning groups; **and**
   - an **arm-time span guard** `_probs_a2a_ep_spans_world(ep_ranks, world_size)`:
     unless `sorted(ep_ranks) == list(range(world_size))`, W1 **refuses to arm** (loud
     `armed=NO` WARNING; probs stays on the EP comm = status quo).

### 2.1 The fix is load-bearing here — and currently self-disarming

On the mission topology every rank's `ep_ranks` is `[0..7]` or `[8..15]`; the guard
compares against `[0..15]` → **False on all 16 ranks → W1 is a loud no-op**. So:

- The original defect (default `use_local_synchronization=False` rendezvous cross-talk
  when >1 EP group exists) **applies to this topology** — the fix is load-bearing.
- But the PR's own guard means the fix does not *enable* W1 here; it only makes the
  refusal safe and loud. The PR body states this explicitly: the local-sync path stays
  fenced "until … verified on a multi-EP-group box." **That verification has never
  happened** (Aug-10 arms were single-group; §1.3 control).

### 2.2 Port status vs the campaign pin (measured, not assumed)

`git apply --reject` of the full PR #28 diff against mcore @ 57efae08b:

- `tensor_parallel/__init__.py` — applies **clean**.
- `tensor_parallel/mappings.py` — applies **clean** (pin blob 6a1605d08a7 == PR base blob).
- `transformer/moe/token_dispatcher.py` — **4/5 hunks apply** (imports, class attrs,
  `__init__` arm block, `token_dispatch` callsite); **hunk #2 rejects** — the module-level
  telemetry/guard block anchors on C′'s `_replay_pass_record`, which the pin predates.
  Port = apply + one manual insert before `class MoETokenDispatcher`. ~15 min, mechanical.
- C′ (#27) is **not** a prerequisite: the W1 diff's only C′ coupling is that anchor
  context; the PR's composition note confirms W1 consumes `input_splits/output_splits`
  identically with or without the replay cache.

⚠️ **Hazard:** the estate copy `runs/overnight_20260810_round3/overlap/patches/w1-probs-a2a.patch`
is the **pre-review-fix** build (no `use_local_synchronization`, no span guard — grep
verified). Arming from that patch on this topology re-creates the original contract
violation. **Port from PR #28 (d794ca3d), never from the estate patch.**

### 2.3 What arming requires beyond the port

One deliberate additional hunk: **relax the span guard** so a non-spanning EP group arms
via the existing `use_local_synchronization=True` creation (minimal form: downgrade the
guard to a telemetry WARNING; keep local-sync creation unconditional). Keep
`BT_MOE_PROBS_A2A_COMM` default-OFF; gate-OFF path stays byte-identical.

Known-risk mechanism the guard was erected against, restated for the tester: with two
disjoint 8-rank groups created at the same program point, the exposure is NCCL/TCPStore
rendezvous cross-talk **if** the two creations collide on store keys. **This is settled
source-side for torch 2.11** (verified in `distributed_c10d.py` @ v2.11.0): with
`use_local_synchronization=True`, `_process_group_name(ranks, use_hashed_name=True)`
names the group by a **hash of its member ranks**, so the disjoint EP groups
([0–7] vs [8–15]) get distinct store prefixes by construction, and each rank creates
exactly one group (its own EP group's), satisfying torch's same-global-creation-order
rule. The `group_desc=moe_probs_a2a_ep_<first-rank>` in the port is **observability
only** (labels the comm in PG dumps/flight recorder) — it is not part of the store key
and is not a collision mitigation. Residual on-box questions V0/V1 must answer: lazy
NCCL init means the comm-2 rendezvous only fires at the first gated dispatch (see V0),
and NCCL comm-creation concurrency itself. Mitigating fact: both EP groups are
**intra-node** (§1.3), so no inter-node rendezvous is involved at all.

## 3. Validation requirement (canary X) — pre-registered

All on q8eg0gq, **fixed wheel only** (standing rule), trainer lifecycle via
`.devbox_up` scripts + `wait_trainer_health.sh`, `BT_SAVE_STATE_SYNC=1` (F1 landmine).

- **V0 — boot + one full step (the multi-group verification the PR defers).** Port +
  guard-relax, `BT_MOE_PROBS_A2A_COMM=1`, mission topology, d1. PASS = model build
  completes AND all 16 ranks log `armed — second communicator created over 8 EP ranks
  — multi-EP-group world (16 ranks), group_desc=moe_probs_a2a_ep_{0,8}` AND **one full
  optimizer step completes**. The step is load-bearing in the criteria: `new_group` is
  called without `device_id`, so NCCL init is lazy and comm 2's first real rendezvous
  happens at the first gated dispatch, not during build — a build-only pass exercises
  nothing (fresh-eyes review finding). FAIL (hang/corrupt comm at build or first
  dispatch) = stop → HOLD, report. There is no group_desc retry: store keys are already
  distinct by rank-hash (§2.3), so a rendezvous failure is not a key collision and has
  no cheap mitigation.
- **V1 — mechanism.** Per-window `{token_issues, probs_issues, waits}` counters
  incrementing; short `BT_PROFILE_RANKS=0,8` capture confirming probs a2a rides comm 2
  (Aug-10-style off-stream check).
- **V2 — numerics.** d2 canary: loss in **12.2–12.4**, gn comparable to fixed-wheel d2
  anchors (**0.36–0.49**), loss trains. W1 is bitwise-safe by construction (same bytes,
  second comm) → any drift = bug, STOP.
- **V3 — perf.** d4 A/B pairs vs the **~878–886 tok/s/GPU** fixed-wheel anchor under the
  campaign's mean-vs-spread rule (tonight's d4 spreads ran 3–4%; |delta| < spread =
  perf-indistinguishable); d16 (anchor **984**, step 133.2 s) only on clean d4 pairs.
  Attack surface on the mission trace: probs_grad bwd exposure **5.41 s/step** (~100%
  wait); lever model **−3.4…−5.1 s/step**. Memory +1.3 GiB (2nd comm buffers) — fits
  (d4 peak 143 GiB, d16 162 GiB).

Cost estimate: port+relax ≈ 30 min Mac-side; V0–V2 ≈ one boot + d2 (≈1 h box); V3 ≈ 2–4 h.

## 4. Option-6 sequencing (`BT_MOE_PROBS_BWD_REORDER`, estate patch `option6-probs-bwd-reorder.patch`)

- **Hard dependency, textual and mechanistic.** Textual: the patch's `mappings.py` base
  blob is `aaaae5365` = **post-W1** mappings.py (its `transformer_engine.py` base
  `3eaa9f61b` matches the campaign pin clean). Mechanistic: option-6 splits the fused
  sort's joint autograd node and seq-bumps the probs path to recover **W1's** backward
  residual (late probs-grad producer) — without W1 armed there is nothing to recover.
- **Sequencing:** option-6 canaries only **after** W1 passes V0–V2 on this topology.
  Its own gate: d2 canary (same band/gn bars) on the **W1-armed** build, then d4 A/B vs
  the **W1-armed baseline** (not vs 880). Do **not** co-arm both gates on the first boot —
  one-lever-at-a-time (the Aug-10 mismatched-gate invalidation is the standing lesson).
  Option-6 is CPU-proven; its only open item is the on-box canary slot.
- **2026-08-13 STAGED (hausdorff, fermi order):** option-6 is rebased onto the W1 port
  and stage-ready as branch chain `jackrao/lps-1062-option6-staged`: trainers @ **855d8d61**
  → megatron-bridge @ **d92774fa9** → Megatron-LM @ **70710d116** (one commit on
  `jackrao/lps-1062-w1-port` @ a58990a91). Rebase was zero-reject: transformer_engine.py /
  mappings.py / moe_utils.py blob-identical to the estate patch targets (f429747a2 /
  30e5bcf14 / 1362a57fd); token_dispatcher.py hunks applied with offsets only (C′-absence).
  CPU proofs re-run green on the staged tree: `test_option6_probs_bwd_reorder.py` (sort
  semantics, engine-order mechanism + controls, gloo carrier protocol, arm predicate, AST
  guards) AND `test_w1_probs_a2a.py` (no W1 regression from the mappings.py signature
  change). Fresh-subagent review: **SHIP** (6/6); flags: TE kernel bitwise-equivalence is
  wrapper-contract-verified, not kernel-body-diffed (on-box numerics gate is the arbiter);
  `_w2_config` getattr is forward-compat (no W2 code on this tree). Both gates remain
  default-OFF; the staged branch carries both dormant.

## 5. Recommendation summary

| question | answer |
|---|---|
| EP-group count of PP2/CP8/EP8 @16 ranks | **2** (derived from pinned `parallel_state.py`; groups [0–7], [8–15], intra-node) |
| Is PR #28's fix load-bearing here? | **Yes** — multi-group is exactly the defect class |
| Does PR #28 as-written enable W1 here? | **No** — span guard disarms (loud no-op) |
| Verdict | **ARM-AFTER-VALIDATION**: port PR #28 (1 manual hunk) + guard-relax hunk + V0–V2 canary, then V3 perf |
| Option-6 | Blocked behind W1 validation; canary vs the W1-armed baseline |

*Cross-refs: CAMPAIGN_ORDERS.md lever 4; A2A_EXPOSURE_DECOMPOSITION §3/§5(b);
SCORECARD_MORNING W1 row; PR basetenlabs/Megatron-LM #28 (head d794ca3d).*
