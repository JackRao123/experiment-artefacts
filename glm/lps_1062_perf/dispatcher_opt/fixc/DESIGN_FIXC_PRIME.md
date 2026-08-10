# DESIGN: FIX C-prime — force the recompute replay's routing to the first-pass
# decision (hilbert, 2026-08-09)

**Gate:** `BT_MOE_ROUTING_REPLAY_FORCE=1` (default OFF; requires
`BT_MOE_DISPATCH_REPLAY_CACHE=1` — the FIX C frames carry the carrier) ·
**Files:** `megatron/core/transformer/moe/router.py` (+249) · **Patch:**
`fixc_prime_routing_force.patch` (router.py only; applies on pristine+fixc —
verified) · **Test:** `test_fixc_prime_routing_force.py` (Mac-CPU, 24
assertions, ALL PASS; FIX C + FIX B suites re-run green).

## 0. Motivating evidence (the ARM-1 verify failure)

FIX C's verify gate (`BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1`) failed on-box,
deterministic 2/2: `num_tokens_per_local_expert_dev` mismatch at
routing_shape (8192, 256) in the 131k boot warmup replay (bohr, ARM-1,
correctly stopped). The verify mode did its job: it caught that **the
recompute replay's routing_map is not bitwise-equal to the first pass's on
this stack**.

**Measured evidence (2026-08-09, on-box):**

- *GEMM grad-mode mechanism — REFUTED.* `probe_gradmode_divergence.py` cases
  1–4 (vanilla torch linear / TE Linear bf16 / TE Linear FP8 e4m3 / 4-deep
  FP8 stack) are ALL `torch.equal` across no_grad vs enable_grad. The
  router's gating is a vanilla torch Linear — cuBLAS does not select kernels
  by grad mode.
- *Leading candidate after the refutation:* the cuDNN DSA **attention**
  forward — attention fwd kernels legitimately select different engine
  variants under grad vs no-grad (training mode saves softmax stats /
  different tiling), so the first pass vs the replay can differ at ULP in
  the attention OUTPUT → hidden drift → router logits → boundary topk flips.
  Kill-shot probe case (case 5 of the probe): `run_fused_absorbed_sparse_attention`
  forward under no_grad vs enable_grad on identical inputs, torch.equal.
- *Instrumented-boot XOR signature (fixc_verify_instrumentation.patch; log
  `round3/fixc_instr_verify_FAIL_318g61w.log`, 3 raise events):* the
  decisive check on fibonacci's one-mechanism synthesis — **integer count
  kernels cannot be order-nondeterministic** (integer atomicAdd is
  value-deterministic), so a purely-local count can only change via a LOCAL
  routing flip, while `output_splits_dev` / `num_global_tokens_per_local_expert_dev`
  / `num_tokens_per_local_expert_dev` derive from the tp_ep ALL-GATHER of
  every rank's counts — a boundary flip on ANY PEER shows up locally as
  "routing_map XOR: 0 flips yet output_splits differ by 1" (mechanism seen
  remotely). The check: for every 0-local-flip raise, the mismatching tensors
  must ALL be allgather-derived and NEVER purely-local (`input_splits_dev` is
  the only purely-local field). **Result: synthesis HOLDS.** Of the 3 raises:
  the two 0-local-flip ones mismatch ONLY `output_splits_dev` (1/16 elements,
  max|diff|=1 — a peer's flip); the one with a local flip (1 token row, 2
  flips = a paired ±1 swap of experts 16↔169 on token 7023) mismatches
  `input_splits_dev` exactly as a local flip must. Zero counterexamples.
  Flip statistics: 1 flipped row per raise, 2 flips (one swap), max|diff|=1 —
  the boundary-flip signature at single-token scale, consistent with the
  flip-sim's ULP-scale prediction (1e-4 → ~7/8192 rows; observed 1 row in
  this boot's replay sample).
- *Consequence (fibonacci's synthesis, confirmed):* C-prime is SUFFICIENT for
  both signatures — forcing every rank's selection makes local counts bitwise
  on all ranks, so the all-gathered splits are bitwise everywhere. No separate
  remote mechanism exists to fix.

**Stack-level fact (record prominently):** the recompute replay is not
bitwise vs the first pass here. This bounds every future metadata-reuse
design (any score-derived integer metadata is unsafe to reuse without
forcing) and partially explains the cross-boot canary floor.

## 1. Why not "save routing_map + probs, consume detached" (the original
   pivot sketch)

Probs are **grad-connected** in the status quo: router → permute → probs a2a
→ expert scaling in TEGroupedMLP; no detach anywhere in
router/moe_layer/token_dispatcher. Consuming detached saved probs would drop
the router-path term in dL/d(hidden_states), which propagates through earlier
layers to the LoRA params — a **structural** gradient change, not ULP. The
bitwise canary would fail. (fibonacci accepted this correction 2026-08-09.)

## 2. The hybrid (approved) design

Save **only the routing_map** (bool [tokens, experts], 2 MB per layer-mb);
make the replay's selection bitwise by construction, keep probs
grad-connected:

```
first pass (frame.is_replay=False):                replay (frame.is_replay=True):
  probs, routing_map = routing(gating(x))            pop carrier stash[layer_number]
  stash routing_map on the packed_seq_params           -> saved_map (one-shot pop)
  carrier[layer_number]                                logits = gating(x)  (recomputed,
                                                       grad-connected)
                                                       logits = logits.masked_fill(
                                                           ~saved_map & has_any, -inf)
                                                       probs, recomputed_map = routing(logits)
                                                       verify: recomputed_map == saved_map
                                                       return (probs, saved_map)
```

- The returned routing_map IS the saved map → the dispatcher's split metadata
  derived from it is bitwise the first pass's → **FIX C's replay cache works
  unchanged** (the verify failure's root cause is removed).
- The probs recompute from the replay's (masked) logits — grad-connected, so
  the router-path activation-grad term is preserved exactly as in the
  status-quo replay. The masked_fill's backward zeroes grads at unsaved
  positions — the same "no grad through unselected experts" as the status
  quo's topk selection.
- **Padding rows** (no saved entries) stay unmasked: their routing_map is
  all-False either way and their probs are never gathered — so the score
  function never sees an all-(-inf) row (no NaN).
- The router GEMM still runs in the replay (logits are needed) — the win is
  the metadata path (all-gather + D2H + event sync), exactly as FIX C
  intended. The router's selection set is now bitwise, so ALL integer
  metadata derived from it is exact by construction.
- Numerics honesty: when a boundary flip *would* have occurred in the status
  quo, C-prime's replay follows the first pass instead — strictly closer to
  the first pass than the status quo. The gates-on vs gates-off canary
  compares two runs that differ exactly on flip events (rare) — a real but
  tiny numerics difference, disclosed; not "bitwise vs gates-off" in grads.

## 3. Requirements landed (fibonacci, 2026-08-09)

1. **Forcing assert (verify mode):** per token, the masked-topk output set
   must equal the saved set EXACTLY — `_routing_force_verify_set_equality`
   (under `BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1`), raises RuntimeError with
   the mismatching row ids. Ties among the selected are impossible
   (unselected are -inf) — asserted anyway.
2. **Aux/z-loss fallback WARNING:** `moe_z_loss_coeff` set or any aux loss
   enabled → no forcing, one-time WARNING, latched (off in the ship config;
   the -inf mask would perturb loss-term computations on the logits).
3. **Memory telemetry + eviction:** in-flight stash byte counter (2 MB per
   entry) with a per-window current/peak log; eviction is by construction —
   one-shot pop on replay fetch, otherwise the entry dies with its microbatch
   carrier. The peak must plateau at layers × in-flight microbatches × 2 MB
   (~600 MB at 75 × 4); growth across steps = leak, visible in the window
   line.

## 4. Telemetry (WARNING level; the v1 lesson)

- one-time gate-state line (`ACTIVE` / `present but DISABLED`);
- one-time `armed — first routing_map stashed on the carrier`;
- one-time `first replay force` line (with the informational router-frozen
  check: the hybrid preserves router grads regardless);
- immediate WARNING on every miss / shape mismatch (status-quo free routing);
- per-window counters every 312 replay passes (~1 step at 78×4):
  `stashes / forces / misses / shape_mismatches / verify_asserts` + in-flight
  stash MB (current/peak). Healthy steady state at 4 mb: `stashes=312,
  forces=312, misses=0` per step.

## 5. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Masked topk doesn't reproduce the saved set (score-function edge case, e.g. group-limited topk interactions) | low | wrong selection | verify-mode set-equality assert (soak) catches on every replay |
| Padding-row NaN | covered by the has_any guard + test | NaN probs | test case 3 |
| z-loss/aux-loss interaction | ship config has them off | loss-value perturbation | latched fallback + WARNING |
| Stash memory | ~600 MB peak at 4 mb | OOM pressure | byte counter + peak log; entries die with the carrier |
| Gate on without the base gate (no frames) | config error | inert (no frames → no forcing) + base gate's own telemetry | documented; the DISABLED line still prints |
| Trainable router | not the ship config | router grads preserved anyway (hybrid) | informational first-force log |

## 6. On-box recipe

1. Apply `fixc_prime_routing_force.patch` on top of the fixc stack.
2. Boot with `BT_MOE_DISPATCH_REPLAY_CACHE=1` + `BT_MOE_ROUTING_REPLAY_FORCE=1`
   (+ B/F env): confirm `armed` + `first replay force` lines.
3. Verify soak: add `BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1` — the ARM-1
   failure must be GONE (the replay's metadata now matches by construction)
   and the forcing set-equality assert must hold (`verify_asserts=312/step`,
   zero raises). ≥20 steps.
4. Loss canary at identical `--warmup-datums`; drift ≤ 5e-3 (expect bitwise
   on non-flip steps; disclose any flip-step deltas).
5. Timed A/B per the FIX C recipe (the C acceptance rows in
   check_acceptance.py apply unchanged).
