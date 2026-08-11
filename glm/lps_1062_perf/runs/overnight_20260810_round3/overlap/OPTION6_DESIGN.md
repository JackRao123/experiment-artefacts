# Option 6 — backward-chain reorder for the probs reverse (design + patch spec)

**Author:** fermi · **Date:** 2026-08-10 · **Ticket:** LPS-1062
**Status:** BUILT Mac-side (patch + CPU tests green), NOT booted — morning-review
deliverable. Revived per `W2V2_DECISION_stub.md`'s sequencing (W3-v3 canary
FAILed T1 at 0.9 % capture; W3 closed for the night).
**Gate:** `BT_MOE_PROBS_BWD_REORDER=1` (default OFF; requires W1 armed).
**Patch:** `runs/overnight_20260810_round3/overlap/patches/option6-probs-bwd-reorder.patch` md5
**6fbf3acc0b170f247ff2b4f0fa71d68d** (498 lines, 4 files), over the ship-w1
stack tip (`d794ca3d2` = B/F + C′ + W1 incl. the new_group guard fix).

---

## 1. The target (what W1 left on the table)

W1 shipped the probs A2A on a second communicator; its mechanism confirmed
(probs off-stream 900/900, gap −2.07 ms/pass) but the wall converted at
~0.15× the model. The residual (hilbert's launch-vs-kernel-start cut): the
probs reverse's kernel waits ~41 ms for the **probs grad itself**, produced
deep in the layer's backward chain — so the reverse lands exposed at the end
of the backward window. Measured residual ~0.5 s/step on box 1; model
~1–1.5 s/step (one latency-bound probs reverse per layer-bwd × 300).

**Why the grad is late — two stacked causes, both must be fixed (the core
design finding of this spec):**

1. **The fused-sort hostage.** The dispatch-side sort
   (`fused_sort_chunks_by_index_with_probs`) emits d(tokens) and d(probs) in
   ONE autograd node (TE's `chunk_sort_bwd` produces both from the saved
   `row_id_map`). d(permuted_probs) is ready mid-MLP-backward (right after
   fc2-dgrad, via the scaled-activation mul's backward), but the joint node
   fires only when d(sorted tokens) ALSO arrives — i.e. after fc1-dgrad, the
   end of the MLP backward. The probs grad is held hostage behind the expert
   dgrad pack.
2. **Natural sequence order.** Even with an independent probs path, the
   autograd engine runs higher sequence numbers first among ready nodes; the
   MLP's fc1 node is created after the dispatcher's sort nodes in the forward,
   so fc1.bwd would still fire before the probs-sort's backward. The split
   alone changes readiness, not priority. (CPU-proven: test sec2's control —
   split without the bump still runs fc1 first.)

And a third requirement falls out of the first two: once the reverse ISSUES
early, its compute-stream wait must not land early with it — an inline wait
at that point would stall the not-yet-pushed fc1-dgrad kernels behind the
~5.7 ms flight (the exposure relocates, nothing is hidden). The wait must
move late.

## 2. The design (three parts, all load-bearing)

**(i) Split the probs sort edge** — `moe_utils.sort_probs_chunks_early_bwd`
(new `_SortProbsEarlyBwd` autograd Function): forward runs the SAME fused
chunk sort for the probs tensor alone
(`te_moe.chunk_sort_fwd(probs[rows,1], splits, sorted_idxs)`), and saves the
`row_id_map`; backward runs the fused inverse sort
(`te_moe.chunk_sort_bwd(d_permuted, None, row_id_map, …)`) — the same op the
fused path's own autograd uses. d(global_probs) now depends ONLY on
d(permuted_probs). The tokens keep the existing fused tokens-only sort
(`probs=None`), their schedule unchanged. Two tiny wrappers are added in
`extensions/transformer_engine.py` (the TE boundary layer) to expose the
`row_id_map` and drive the bwd op. No host sync anywhere (the unfused
split/cat alternative was rejected: its backward is ~511 tiny kernels per
layer-bwd — the FIX-B launch-pressure class; and its engine-side grad
accumulation is a reduction-order change. The fused path is one kernel each
way, device splits only, bitwise row permutations).

**(ii) Seq-bump the probs path above the fc1 pack** — the house mechanism
(`set_tensor_grad_fn_sequence_sr`, shared_experts.py:585): both the
probs-sort node (in `dispatch_postprocess`) and the probs-A2A node (in
`token_dispatch`) are bumped to `INT_MAX`, so the probs reverse issues right
after fc2-dgrad, ahead of the remaining expert dgrads. Replay-only by
construction (the first pass has no `grad_fn`; the helper no-ops — house
rule R4). **The W1-v2 inert-bump lesson, addressed head-on:** that bump was
inert because it targeted the two A2A nodes' RELATIVE order, which
construction already got right (probs=N+1 after tokens=N, and the probs
reverse already fired first). This bump's target is different: the probs
path vs the MLP's fc1 node (created later in the forward, higher natural
seq) — the relation hilbert's cut actually measured late. Test sec2 proves
the bump is load-bearing (without it, fc1 fires first).

**(iii) Defer the reverse's wait to the tokens reverse** —
`_AllToAllDeferredWait.backward` gains the optional `bwd_defer`
early/late carrier protocol (a per-pass dict created in `token_dispatch`):
the **early** sibling (probs reverse) issues its reverse A2A and stashes the
work handle WITHOUT waiting; the **late** sibling (tokens reverse) waits its
own work AND the stashed early handle. Firing order is guaranteed by data
dependency (the tokens grad needs the full MLP backward; the probs grad
under (i) needs only fc2-dgrad) — the late sibling always runs second.
Belt-and-suspenders: if the late sibling somehow ran first, the early one
waits inline and logs loud; a missing handle logs loud. Default path
(`bwd_defer=None`) is byte-identical to today (W1's CPU suite re-run green
against the patched tree).

**Safety argument for the unwaited early return (the load-bearing claim):**
the early reverse's output grad has exactly ONE consumer — permute1's
backward (the fused permute-with-probs in `dispatch_preprocess`) — which
needs BOTH reverses' outputs, so it fires only after the late sibling's
backward has waited both handles. The consumer is downstream of the wait by
data dependency; on CUDA the read is stream-ordered and safe. (CPU proof
note: a LEAF consumer would race the unwaited buffer — the engine's
AccumulateGrad copies early on CPU; that's a test artifact, and the suite
asserts the returned buffer's contents after the late wait instead. If a
future change adds a second consumer of the probs-reverse grad, it must be
downstream of the late wait too — flagged in the patch notes.)

## 3. Win model

The probs reverse's flight (~5.7 ms latency-bound class) moves from
"exposed at the end of the layer's backward window" to "issued right after
fc2-dgrad, hidden behind the act-bwd + fc1-dgrad pack (~10–20 ms of cover),
waited for free at the tokens-reverse point". Per layer-bwd: recover the
exposed reverse ≈ the W1 residual. **Model: ~1–1.5 s/step; box-1 measured
residual class ~0.5 s/step** (two-tier rule: mechanism first — the
trace-checkable prediction is the bwd-window probs-reverse late-start
fraction → ~0; wall on the tail-bound box is informational).

## 4. Numerics, memory, gates

- **Numerics:** pure scheduling + exact row permutations. Forward values
  bitwise (the fused tokens-only sort and the fused probs-only sort are the
  same permutation family as the fused with-probs sort); backward values
  bitwise (the same `chunk_sort_bwd` op, disjoint probs edge). In-process
  gate: `torch.equal`, gate off vs on (gold tier). **Pre-registered risk
  (the W1 class):** engine execution order among ready nodes changes, so
  grad accumulation at junctions (the residual hidden-grad add; the
  probs→router junction) can change summation order — ULP-class drift, not
  math. Diagnosis criterion (carried from W1): window-1 exact + window-2+
  ULP drift ⇒ scheduling, not a bug; quantify under Jack's variance-class
  bar (house band, matched warmup-datums); do NOT silently accept
  tolerance-level drift — that call is Jack's.
- **Memory:** ≈ 0 — the probs payload is KB-scale (latency-bound); the
  carrier is a per-pass dict; no new buffers beyond the sort's own output.
- **Gate / arm conditions (loud fallback with reason):**
  `BT_MOE_PROBS_BWD_REORDER=1` AND W1's second comm armed AND not W2 AND
  dropless AND TP1 AND num_local_experts>1 AND moe_permute_fusion on.
  One-time gate-state + armed/armed=NO-<reason> WARNINGs; per-window
  `{early_issues, sort_splits}` counters (an armed gate that never fires is
  loud by absence).

## 5. Files + tests

- `megatron/core/extensions/transformer_engine.py` — the two row_id_map
  wrappers (TE boundary).
- `megatron/core/transformer/moe/moe_utils.py` — `_SortProbsEarlyBwd` +
  `sort_probs_chunks_early_bwd`.
- `megatron/core/tensor_parallel/mappings.py` — the `bwd_defer` carrier
  protocol (+ module logger, which mappings.py lacked).
- `megatron/core/transformer/moe/token_dispatcher.py` — gate helper + arm
  predicate (init-time, loud), the W1-branch carrier wiring + probs-A2A
  seq-bump, the `dispatch_postprocess` sort split + probs-sort seq-bump,
  telemetry.

**CPU suite** (`tests/test_option6_probs_bwd_reorder.py`, ALL PASS on Mac):
sec1 sort values vs a reference permutation (stubbed fused ops); **sec2 the
mechanism proof** (engine order: split+bump ⇒ probs path before fc1;
controls: no-bump ⇒ fc1 first; joint sort ⇒ the hostage order); sec3 the
carrier protocol on 2-proc gloo (normal order + pathological late-first +
missing handle; values vs the reverse-A2A reference); sec4 gate + arm
predicate cases; sec5 AST source guards (no `.tolist()` host sync in the
sort Function; the fused with-probs call preserved in the else branch;
bwd_defer only under the gate; the default backward path unchanged).
Regression: the W1 CPU suite re-runs green against the patched tree.

**On-box gate recipe (when a slot exists):** gate off vs on at 131k×d4,
in-process `torch.equal` on one layer's outputs AND input grads (gold tier);
then the canary under the house band with the 20-step drift cover; trace
readout: bwd-window probs-reverse late-start fraction → ~0, tokens-reverse
start NOT regressed (the W1-v2 readiness-coupling guard carried forward),
wait accounting by CAUSE (standing rule).

## 6. Composition notes

- **W1:** required (the probs reverse needs its own communicator). The W1
  new_group guard (ship-w1 `d794ca3d2`) is in the base.
- **W2:** excluded (W2 has its own probs path; the v2 seq-bump disposition
  stays in the W2V2 stub — re-weighed separately).
- **W3:** independent; option 6 targets the engine-issue-order residual that
  W3's lookahead never addressed (and now won't — closed for the night).
- **FIX C:** no interaction (the sort split is downstream of the cached
  metadata; the replay replays the same graph).
