# DESIGN F2 — DP>1 partition-count equalization + phantom partitions (THD-CP + EP deadlock)

Author: bohr (agent session) · Date: 2026-08-09 · Target: trainers `server/src` (NOT vendored mcore) · Base: `0e0b65a6`
Ticket context: LPS-1062. F2 is the ship-blocker for every DP>1 THD-CP mesh (EP16/CP8/DP2, EP16/CP8/DP4, EP32/CP16/DP2).

---

## 1. Problem

THD-CP packs a **data-dependent number of partitions per DP replica**
(`partition_thd_cp_datums`, packing.py:806 — greedy bin-pack of this replica's
datum slice into ≤ `max_seq_len` padded rows). The controller's per-partition
loop (`megatron_controller.py` `_run_forward_backward`, :2764 ff.) carries an
explicit invariant:

> INVARIANT — NO DATA-PARALLEL COLLECTIVE MAY RUN INSIDE THIS LOOP. …
> This is what lets DP>1 work without phantom padding.

The invariant is **false**: it covers only the collectives the *controller*
issues. The *model* issues MoE expert-parallel collectives per partition, and
the EP group spans DP replicas whenever `ep_size > ranks-per-replica`
(expB: EP16 over 16 ranks = CP8 × DP2 → the EP16 group covers both replicas):

- `token_dispatcher.py:558` — `gather_from_sequence_parallel_region(num_local_tokens_per_expert, group=tp_ep_group)` (all_gather, fixed shape, per MoE layer per partition);
- `token_dispatcher.py:959` region — the a2a split computation / DtoH sync path, same per-partition cadence;
- per-layer EP `all_to_all_single` ×3 (dispatch, combine, + grad path in backward).

**Deadlock mechanism** (confirmed twice, py-spy line-level):
replica counts disagree (boot warmup pass-1 sends 1 datum → `data[1:1]` empty →
dp_rank1 packs **0** partitions; B-custmix heterogeneous data → 3 vs 2
partitions). The longer replica enters partition N's first-MoE-layer EP
all_gather; the shorter replica has exited the loop and sits in the post-loop
DP grad finalize (`finalize_model_grads.py:502` via `saved_finalize`). EP group
∋ both replicas ⇒ permanent NCCL deadlock (warmup lifts pg timeout to 120 min,
so it hangs silently). DP>1 is unshippable on real THD data until fixed.

## 2. Fix overview

**Equalize the per-replica partition count, then pad the short replica(s) with
phantom partitions.**

1. In `_pack_thd_cp_microbatches`, immediately after `partition_thd_cp_datums`:
   one `all_reduce(MAX)` of the local partition count over the **pure-DP group**
   (`mpu.get_data_parallel_group(with_context_parallel=False)` — same group
   `_dp_reduce_sum` already uses post-loop). Once per `forward_backward` op,
   on every rank, at the same point in the op sequence ⇒ no new desync vector.
   Cost: one 4-byte reduce per op (vs ~46 s steps) — free.
2. Replicas with fewer than `max` partitions append **phantom partitions** as a
   **strict suffix**. A phantom partition is one (DPO: two) synthetic
   `pad_multiple`-token datum(s) — 16 tokens at CP8, 32 at CP16 — packed through
   the *same* `pack_thd_cp_microbatch` code path as real data, with every
   loss-relevant input masked to the zero-contribution values below.
3. `_run_forward_backward` runs phantom partitions through the normal
   per-partition schedule (every collective fires, same order), but skips
   datum-length accounting and per-datum logprob extraction for them.

Every rank then executes exactly `max` per-partition collective sequences per
op ⇒ EP collectives line up across replicas ⇒ deadlock gone, at boot and
mid-step, for any data distribution.

## 3. Phantom semantics — exact zero contribution

Phantom datum(s): `input_ids = [1]*L` (any in-vocab id), `L = pad_multiple`,
`position_ids = 0..L-1`, and loss inputs keyed off the **global batch's**
key-set (every rank receives full `details.data` via the dispatcher broadcast,
so even a zero-datum replica knows which fields the loss fn needs):

| field | phantom value | why it zeroes contribution |
|---|---|---|
| `target_tokens` (labels) | all `-100` | CE/DPO: `active_mask = labels != -100` all-False |
| `weights` | all `0.0` | CE: `per_token = -logprobs · weights · active_mask` ≡ 0 (`_apply_weighted_ce_mask`, :776) |
| `advantages` | all `0.0` | RL: `loss_mask = (advs != 0) & (targets != -100)` all-False (:1214) |
| `logprobs` (sampling) | all `0.0` | finite `log_ratio`; killed by mask |
| `temperatures` | all `1.0` | neutral |
| `ref_logprobs` | all `0.0` | DPO only; pair dropped by existing phantom-pair logic (:1433) |

Per-loss-fn proof that a phantom partition reports exactly `[sum=0, count=0]`
and produces exactly zero gradients:

- **cross_entropy** — every position masked ⇒ `loss_sum = 0`, `num_active = 0`
  (`_ce_loss_from_hidden` :802; the chunked LM head already handles
  all-inactive microbatches — the bshd path's phantom rows exercise this).
  Loss ≡ 0 ⇒ ∂loss/∂θ ≡ 0 for **all** θ (embeddings, attention, experts,
  router, LM head): the backward graph still executes (all collectives fire)
  but every gradient is exactly 0.
- **ppo / cispo / dro / dppo / importance_sampling** — per-token losses all
  gated by `loss_mask.float() * …` (:1055–1132); mask all-False ⇒ 0.
  `mismatch_kl`: `out_logprobs[loss_mask]` is an empty selection ⇒ `[0, 0]`
  report (:1264–1270).
- **dpo** — phantom chosen/rejected pair with `w = 0`: `active_mask` all-False,
  and the existing "(0,0) phantom pair" drop (:1433 ff., written for the bshd
  path) prevents the `-logsigmoid(0) = ln 2` mean contamination.

**MoE side effects: none.** `_configure_moe_provider` (:336) forces
`moe_aux_loss_coeff = 0.0` and `moe_router_load_balancing_type = "none"` for
all MoE bridges — no aux loss, no aux-free bias update, and no
`moe_z_loss_coeff` is set by the GLM-5.2 bridge (grepped). Router forward on
phantom tokens computes routing and dispatches them (that's what we want —
the collectives fire), but with loss ≡ 0 no gradient or statistic flows back.

**Aggregation:** the phantom's CP-reduced `[0, 0]` report is appended to
`metrics_for_aggregation` like any partition; `aggregate_microbatch_metrics`
(packing.py:415) sums `[sum, count]` entries ⇒ adding `[0, 0]` is a literal
no-op. The post-loop `_dp_reduce_sum` of `(loss_sum, tokens, mismatch_kl)` is
bit-identical to a no-phantom run. `datum_offset` never advances for phantoms
(skipped), so the post-loop `datum_offset != len(datum_lengths)` check still
validates the real partitions exactly. **Constraint (1) and (3) hold
exactly, not approximately.**

## 4. Why dummy-token phantoms, not zero-token — the edge-case audit

The spec allowed either. Zero-token phantoms (empty `cu_seqlens`, S=0 rows)
would avoid even the trivial compute, but every kernel on the path would need
a zero-input audit — and several are likely-unsafe. Dummy-token phantoms make
every one of these a non-issue (each kernel sees a legal minimal input), at
the cost of one 16–32-token forward+backward per missing partition:

| # | Module / site | zero-token risk | dummy-token behavior |
|---|---|---|---|
| 1 | EP `all_to_all_single` ×3 (token_dispatcher) | zero splits are *legal* in NCCL/`all_to_all_single`, but the moe_permute_fusion + grouped-GEMM path between the a2a's is unproven at 0 rows | normal small splits |
| 2 | fused permute (`moe_permute_fusion=True`) | 0-row permute kernel — unvalidated | ≥1 row |
| 3 | grouped GEMM (`moe_grouped_gemm=True`) | all-zero `tokens_per_expert` — cublas/grouped-GEMM zero-m unvalidated | normal counts |
| 4 | DSA attention (dsa.py) | empty `cu_seqlens` / `max_seqlen=0`: varlen kernel launch bounds, the CP layout all_gather (dsa.py:303) of all-zero lens, indexer KV reorder — **the riskiest site** | one 16–32-token doc, legal `cu_seqlens=[0, L]` |
| 5 | `tex.thd_get_partitioned_indices` (GPU-only) | empty `cu_padded` unvalidated | normal zigzag of a tiny doc |
| 6 | `_thd_cp_sequence_sums` (:739) | `torch.stack([])` on 0 docs ⇒ RuntimeError | one doc |
| 7 | Megatron schedule `seq_length=0` | pipeline/schedule guards unvalidated | normal tiny seq |
| 8 | `PackedSeqParams(cu_seqlens=[0])` | mcore attention metadata asserts unvalidated | normal params |
| 9 | `chunked_lm_head_loss_from_hidden` | S=0 hidden states | all-masked but well-shaped |
| 10 | flex/DeepEP dispatcher (if configured) | zero-token dispatch unvalidated | normal dispatch |

Verdict: **dummy-token**. The audit above becomes "items we never have to
run" instead of "items we must patch and re-verify per kernel upgrade". The
phantom's compute cost is ≤ 32 tokens vs the 131,072-token real partitions it
accompanies (< 0.03%).

## 5. Collective-order argument (why equal counts are sufficient)

Within one partition, every rank of the EP group executes the same layer
sequence; each MoE layer fires the same collective *sequence* (router →
tp_ep all_gather of expert counts → dispatch a2a → expert GEMMs → combine a2a,
then the backward mirror). These collectives are already shape-heterogeneous
across replicas today (real partitions on different replicas carry different
token counts — B-custmix partitions 1–2 ran fine; only the *count* mismatch
hung). Equalizing the partition count therefore aligns the one remaining
desync axis. CP-scoped collectives (DSA layout all_gather, logprob stitch,
`_loss_report` CP reduce) are within-replica and unaffected: all CP ranks of a
replica share the replica's partition count by construction.

The new DP all_reduce of the count is issued by every rank exactly once per
`execute_forward_backward`, before any partition runs — same call-site
ordering on all ranks, so it cannot interleave with per-partition EP
collectives.

## 6. Code changes (all in `server/src`, none in vendored mcore)

**`dp_worker/api/packing.py`**
- `PackedMicrobatches`: add trailing field `phantom_partitions: int = 0`
  (count of suffix microbatches that are phantoms; default keeps the bshd
  path and all existing constructors unchanged).
- Add `build_phantom_thd_cp_partition(template: Datum, *, pad_multiple: int,
  atomic_row_group_size: int) -> list[Datum]` — pure function fabricating the
  masked synthetic datum(s) per §3 from the template's `loss_fn_inputs`
  key-set. Pure ⇒ Mac-unit-testable.

**`dp_worker/api/megatron_controller.py`**
- `_pack_thd_cp_microbatches`: signature gains `template_data: list[Datum]`
  (the global `details.data`). After `partition_thd_cp_datums`:
  ```python
  num_real = len(partitions)
  target = num_real
  if self._data_parallel_world_size > 1 and _phantom_partitions_enabled():
      target = _dp_max_partition_count(num_real)   # all_reduce MAX, pure-DP group
  for _ in range(target - num_real):
      partitions.append(build_phantom_thd_cp_partition(
          template_data[0], pad_multiple=pad_multiple,
          atomic_row_group_size=atomic_row_group_size))
  ```
  and return `PackedMicrobatches(..., phantom_partitions=target - num_real)`.
  The existing packing loop then packs phantoms through the identical code
  path (no fork in Batch construction).
- `_run_forward_backward`: derive `num_real = len(microbatches) -
  packed.phantom_partitions`; in the loop, for suffix indices: run
  `_forward_backward` + append the (zero) metrics report as today, but
  `continue` before the `datum_offset` / `thd_logprobs_to_loss_fn_outputs`
  block. Post-loop finalize, DP reduce, and the datum-count validation are
  unchanged.
- Helpers: `_phantom_partitions_enabled()` → `os.environ.get(
  "BT_F2_PHANTOM_PARTITIONS", "1") == "1"`; `_dp_max_partition_count(n)` →
  MAX-all_reduce over `get_data_parallel_group(with_context_parallel=False)`,
  identity when dist uninitialized or group size 1 (mirrors `_dp_reduce_sum`'s
  guards, so unit tests and DP1 runs are untouched).
- Replace the now-false docstring invariant in `_pack_thd_cp_microbatches`
  (:2424–2430) and the stale comment at :2596 with the correct invariant:
  *"DP ranks may pack different real-partition counts; counts are equalized
  by phantom suffix partitions because per-partition MoE EP collectives span
  DP replicas whenever EP > ranks-per-replica."*

**Gating (constraint 4):** whole mechanism fires only when
`dp_size > 1` **and** `BT_F2_PHANTOM_PARTITIONS != "0"`. DP=1: no all_reduce,
no phantoms, byte-identical behavior. Kill-switch `=0` restores pre-fix
behavior for A/B forensics.

## 7. Edge cases

- **All ranks zero partitions:** impossible — `execute_forward_backward`
  rejects `n_global == 0` (:2532), and rank 0 always owns ≥ 1 datum ⇒
  `max ≥ 1`.
- **Zero-datum replica (boot warmup):** packs 0 real + `max` phantoms; the
  template comes from the global batch (broadcast), not local data. This is
  the deterministic boot-deadlock case — now covered.
- **DPO:** `atomic_row_group_size = 2` ⇒ phantom partition = 2 masked datums;
  existing phantom-pair drop keeps the mean exact.
- **forward_only (ForwardOp / NLL eval):** same loop, same EP collectives in
  forward ⇒ phantoms apply identically.
- **PP > 1:** rejected for THD CP today (`_validate_thd_context_parallelism`);
  the metric-append path is last-stage-only and stays correct if that lifts.
- **Memory:** ≤ 32 tokens of extra activations per missing partition — no
  effect on the memory ceiling (unlike real padding).

## 8. Unit-test plan (Mac-runnable, CPU)

1. `build_phantom_thd_cp_partition` for each loss-fn key-set (CE, RL,
   DPO-pair): packs through `pack_thd_cp_microbatch` on CPU; assert
   `cu_seqlens == [0, L]`, labels all −100, weights/advantages all 0, fields
   non-None exactly where the template has them.
2. `_apply_weighted_ce_mask` with `weights=0` / `labels=-100`: per-token loss
   ≡ 0, `weight_active` all-False. (Exists partially — extend.)
3. `_rl_per_token_loss` (all 5 RL fns) with `loss_mask` all-False ⇒ exact
   zeros; `_mismatch_kl_per_token` on empty selection ⇒ `(0, 0)`.
4. `aggregate_microbatch_metrics` with a `[0, 0]` entry interleaved: totals
   unchanged vs without it.
5. `_dp_max_partition_count`: (a) dist-uninitialized ⇒ identity; (b) 2-proc
   gloo group on CPU, counts (3, 2) ⇒ both see 3. (Real collective test, no
   GPU needed.)
6. Regression: `_pack_thd_cp_microbatches` is GPU-only (tex import), so the
   count-equalization block gets a factored pure helper
   (`_plan_phantom_suffix(num_real, target)`) under test instead; the
   integration is covered by the on-box recipe below.
7. DPO: (0,0) phantom pair is dropped from the pair mean (likely already
   covered by existing phantom-pair tests — verify and extend).

## 9. On-box validation recipe (2×8 B300, EP16/CP8/DP2, patch applied to shared clone)

a. **Boot-deadlock repro→fix:** `BT_SKIP_WARMUP` *unset*, default warmup,
   DP2 boot ⇒ server reaches READY (pre-fix: deterministic hang in warmup
   pass-1). Also boot with `BT_F2_PHANTOM_PARTITIONS=0` to re-confirm the
   hang is still reproducible on the same bits (kill-switch works).
b. **B-custmix heterogeneous shape** (20 datums, 10,240–63,488 tok, 530,432
   tok/step — the empirical F2 repro) ⇒ runs to completion; expect per-window
   losses sane and no NCCL-spin.
c. **Loss canary:** DP2 with phantoms vs DP1 golden at identical total tokens
   (same seed-fixed synthetic data, same warmup-datums) ⇒ agreement within
   reduction-order tolerance (~5e-3 cross-config); grad-norm series within
   the same band.
d. **Perf non-regression:** equal-length DP2 131k×d4 bench reproduces ~745
   tok/s/GPU steady (phantom overhead immeasurable at equal counts; the
   all_reduce is one 4-byte op per step).

## 10. Risks / open questions

- **Non-golden meshes (PP>1, TP>1 with SP):** pad_multiple already accounts
  for SP; PP is rejected for THD CP. No new constraints introduced.
- **deepep/flex dispatcher:** not the golden config; if enabled, phantom
  tokens dispatch normally (dummy-token advantage #10). Worth one smoke boot
  before declaring flex+DP>1 safe.
- **Kill-switch discipline:** `BT_F2_PHANTOM_PARTITIONS=0` must remain a
  forensics-only flag; add a boot WARNING when DP>1 runs with it off.
- **Cost bound:** worst-case phantom overhead = (max−min) partitions × ≤32
  tokens of forward+backward on the short replica — bounded and tiny; no
  effect on the ≥10 GiB headroom constraint.
