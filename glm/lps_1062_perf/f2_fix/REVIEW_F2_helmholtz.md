# F2 third review — bohr's f2.patch (helmholtz, 2026-08-09)

Scope requested: fresh eyes on the phantom-partition edge cases. Reviewed
`f2_fix/DESIGN_F2_bohr.md` + `f2_fix/f2.patch` against the current checkout
(cross-checked the loss masks in `backends/megatron_bridge/loss.py`, the
packer field handling in `backends/megatron_bridge/thd_cp.py`, the
aggregation in `packing/packer.py`, and the loop in
`backends/megatron_bridge/training_runner.py:320`).

**Verdict: sound; ship after the two minor gaps below.** The core mechanism
(equalize per-replica partition counts via one pure-DP all-reduce MAX + pad
short replicas with exact-zero phantom suffix partitions) closes the
deadlock correctly, and the dummy-token-over-zero-token call is right (the
§4 audit table is the decisive argument — DSA varlen at S=0 is genuinely
unauditable).

## Confirmed (no action)

1. **Exact-zero masking is real.** CE: `per_token = -logprobs · weights ·
   active_mask`, `active_mask = labels != -100`, `weight_active = (weights>0)
   & active_mask` (loss.py:120-121, 142-147) — phantom (labels −100, weights
   0) ⇒ exact [0, 0]. RL fns: `loss_mask.float() · …` (loss.py:389-429) ⇒
   exact zeros. Aggregation sums `[sum, count]` (packing/packer.py:80-90) ⇒
   `[0,0]` is a literal no-op. Gradients: loss ≡ 0 ⇒ ∂/∂θ ≡ 0 given finite
   activations (the NaN×0 residual risk is correctly delegated to the on-box
   canary — ONBOX_VALIDATION_F2.md).
2. **Field shapes.** The packer coerces every loss input to per-token
   `seq_len` via `floats(data, seq_len, name)` (thd_cp.py:285-308) — so the
   phantom's `[pad_multiple]` fabrication per key is contract-correct
   (temperatures included). My initial scalar-field concern dissolves.
3. **Suffix + accounting.** `num_real = len(microbatches) -
   packed.phantom_partitions` with phantoms as a strict suffix: the loop
   iterates `packed.microbatches` directly with no reordering
   (training_runner.py:328,343). The `continue` before datum accounting is
   necessary AND sufficient: a phantom partition's cu_seqlens ([0, L] or
   [0, L, 2L] for DPO) would otherwise advance `datum_offset` by 1-2 and
   trip the post-loop `datum_offset != len(datum_lengths)` check.
4. **CP-collective alignment under the skip.** All CP ranks of a replica
   share the replica's partition count by construction, so skipping the
   (CP-scoped) logprob stitch for phantom suffixes stays aligned
   within-replica; the EP alignment across replicas is exactly what the
   count equalization buys. The collective-order argument (§5) holds:
   per-partition collectives are already shape-heterogeneous across replicas
   today; only the count mismatched.
5. **MoE side effects none** (aux loss / load-balance force-off in
   `_configure_moe_provider` — independently verified in megatron_config.py
   :41-60 earlier today).
6. **Guards.** `_dp_max_partition_count` mirrors `_dp_reduce_sum`'s
   dist-uninitialized / size-1 identities; the gate fires only when
   `dp_size > 1` and the env is not "0" — DP=1 is byte-identical. The gloo
   CPU test for the MAX all-reduce is a real collective test.

## Findings (2 minor gaps + 2 notes)

- **F-a (minor gap):** design §10 promises a boot WARNING when DP>1 runs
  with `BT_F2_PHANTOM_PARTITIONS=0` (forensics-only flag) — not in the
  patch. Add it (one line at controller init when `dp_size > 1 and not
  enabled`); an operator who sets the kill-switch and forgets gets a silent
  deadlock.
- **F-b (rebase note):** the patch paths target 0e0b65a6
  (`dp_worker/api/megatron_controller.py`, `dp_worker/api/packing.py`). The
  current checkout (ef4ea4a8, post-#900) has moved the loop to
  `backends/megatron_bridge/training_runner.py` and packing to
  `backends/megatron_bridge/thd_cp.py` + `packing/packer.py`. The logic maps
  1:1 (verified above), but the rebase must move every hunk — flag the
  `PackedMicrobatches.phantom_partitions` field's new home explicitly.
- **F-c (cross-patch note for fibonacci's routing, no f2 action):**
  phantom partitions make zero-count (peer, group) MoE A2A ROUTINE, not
  exceptional: a 16–32-token phantom × top-8 over 256 experts leaves most
  experts and most (peer, group) pairs at 0 rows. This raises the stakes on
  W2's V3 (NCCL 0-count list-A2A behavior — already in the T2 gate) and
  argues for one DP>1+phantoms run in the W2 canary once both land. W1/W3
  have no phantom interaction (probs comm is count-independent; the W3
  registry is per-microbatch-carrier).
- **F-d (verify-once on-box):** `template_data[0]` relies on the dispatcher
  broadcast giving every rank the full global batch — stated as existing
  behavior; the boot-warmup zero-datum-replica case (recipe step a)
  exercises exactly this.

## Edge-case coverage assessment (the requested focus)

Covered well: all-ranks-zero (impossible by the n_global reject + rank-0
ownership), zero-datum replica (0 real + max phantoms), DPO pair
(atomic_row_group_size=2 + the existing (0,0) phantom-pair drop),
forward_only, PP>1 (rejected today), memory (≤32 tokens/partition),
kill-switch A/B. The zero-token alternative was correctly rejected.

Not gaps but worth stating in the notes: (i) the phantom's EP collectives
are what keep the group aligned — any future MoE-dispatcher change (incl.
my W1/W2 gates) must stay process-wide-consistent, which they are by
construction; (ii) worst-case overhead bound ((max−min) × ≤32 tokens ×
fwd+bwd) is stated and tiny — no per-shape cap needed.
