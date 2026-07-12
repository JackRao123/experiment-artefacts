# THD context parallelism (CP) in #592 — how it works, how to defend it

Personal crib sheet for the CP32 131k GLM-5.2 LoRA SFT stack. Every claim has a
file:line so you can pull up the code in review. Line numbers as of the
2026-07-11 rebase (branch `jackrao/glm-131k-cp`).

## 1. Why CP at all, in one breath

A 131k-token sequence doesn't fit through EP32 on one GPU: activation memory
scales with the tokens *this GPU* holds in flight. CP splits the **sequence**
across GPUs (vs DP splitting the *batch* and TP splitting the *weights*), so 32
ranks each hold ~4k tokens of the 131k row. Profiling showed the real lever is
DP=1 (global tokens in flight through the MoE dispatch), which is why the
config is TP1/PP1/EP32/**CP32/DP1** and not CP4/DP8.

## 2. The data path, end to end

```
pack (CPU)          → one global row per DP replica, labels PRE-SHIFTED per doc,
                      each doc padded to a multiple of 2·cp
zigzag shard (GPU)  → tex.thd_get_partitioned_indices picks this CP rank's chunks
forward             → DSA attention all-gathers KV across CP (cp_comm_type="allgather")
chunked LM head     → scores ALL local positions (targets_pre_shifted=True)
loss                → sum([loss, n_tokens]) all-reduced over the dp×cp group
grad                → schedule ×cp, DDP ÷(dp·cp) ⇒ net ÷dp; optim applies ÷n_tokens
```

- Pack: `packing.py` — `pack_thd_cp_microbatch` concatenates the DP replica's
  datums into one `(1, S_global)` row. Megatron-free, unit-testable.
- Shard: `megatron_controller.py` — `_pack_thd_cp_microbatches` calls
  `tex.thd_get_partitioned_indices(cu_seqlens_padded, …)` + `index_select`.
  Identical logic to upstream's `_get_batch_on_this_cp_rank_per_document_balancing`.
- `PackedSeqParams` carries **global** cu_seqlens — the kernels need global
  document boundaries, not local ones.

## 3. Zigzag: what and why

For causal attention, a contiguous split is load-imbalanced: the rank holding
the *end* of the sequence attends to everything, the rank holding the *start*
attends to almost nothing. Zigzag fixes this: split each document into
`2·cp_size` equal chunks; rank r gets chunk `r` (front) + chunk `2cp−1−r`
(back). Every rank then does the same total attention work. That's the whole
trick — it's a *permutation* of tokens across ranks, nothing more.

Consequences you must be able to state:

- **Each document is tail-padded to a multiple of `2·cp_size`** so the chunking
  is exact (`packing.py`). The divisibility check inside Megatron is CPU-only —
  on GPU it's trusted, so our packer is the guarantee.
- **Position ids are computed globally, then sharded** — each token keeps its
  true RoPE position regardless of which rank it lands on.
- **The kernel side must know absolute positions too**: the cuDNN DSA indexer
  defaults to top-left causal masking, which is wrong for any chunk that
  doesn't start at position 0. LM#14 (`545ba95cb`) passes `q_causal_offsets`
  (each chunk's global start). Without it, top-k silently selects from a
  chunk-local window — *trains without crashing at short seq* (2–25% top-k
  overlap vs reference), IMAs at 131k. This is also why the image needs
  `nvidia-cudnn-frontend==1.26.0` (1.25.0 lacks the kwarg).

## 4. The crux: pre-shifted labels (`targets_pre_shifted=True`)

Normal (non-CP) convention: labels are the same row as tokens; the LM head
scores `hidden[:-1]` against `labels[1:]` — i.e. the shift happens **at loss
time**, and the last position is dropped because it has no next token.

Under CP that convention breaks: the token that position `i` must predict can
live **on a different rank** (whenever `i` is the last position of a zigzag
chunk). You cannot shift locally.

Fix: shift **globally, before sharding**. At pack time each document's label
row is already `tokens[1:] + [-100]` (`packing.py`). After that, "position i's
target" is a *local* property — it travels with the token through any
permutation. The chunked LM head then scores **all** S_local positions with no
drop (`chunked_lm_head.py`, `targets_pre_shifted=True`).

**The invariant (memorize this):** after packing, every position carries either
its true next token or −100 (document end, padding, global end) — independent
of how the row is later partitioned. Correctness of the loss is therefore
partition-agnostic; zigzag never appears in the loss code at all.

Worked example, doc = `[A B C D]`, cp=2, chunks of 1 (`2·cp = 4` chunks):

```
global tokens   A    B    C    D
global labels   B    C    D    -100      (pre-shifted at pack time)
rank 0 gets   A(+B)  D(-100)             (chunks 0 and 3)
rank 1 gets   B(+C)  C(+D)               (chunks 1 and 2)
```

Rank 0 scores A→B locally even though B lives on rank 1, because the *label* B
was attached to A's position before the split.

## 5. Loss: sum over dp×cp, count each token once

`_loss_report` (`megatron_controller.py`) all-reduces the pair
`[loss_sum, num_active_tokens]` over the **dp×cp** process group:

- CP ranks hold **disjoint** token sets of the same row → SUM is correct
  (an average would double-normalize).
- The result is the exact same global `(Σ loss, Σ tokens)` you'd get at cp=1 —
  bit-for-bit the same *definition* of the loss.
- Non-last PP stages don't compute loss, so `loss_tokens` is **broadcast**
  along PP; without it their gradient scaling would silently no-op.

## 6. Gradients: why there is no ÷cp anywhere

Reviewer will ask: "each of the 32 CP ranks backprops its own loss share —
where do you divide by 32?" Answer: **it cancels, by construction, inside
Megatron**:

1. The pipeline schedule multiplies each rank's loss by `cp_size` before
   backward (`Megatron-LM schedules.py:340`).
2. Megatron DDP averages gradients over its data-parallel group, which is
   **dp×cp**, i.e. multiplies by `1/(dp·cp)` (`distributed_data_parallel.py:203`).
3. Net: `cp/(dp·cp) = 1/dp` — exactly the standard DP average. CP is invisible.
4. The controller then applies the single remaining factor `1/loss_tokens` at
   optim time (`megatron_controller.py`, `execute_optim_step`).

Because CP ranks hold disjoint tokens, "sum of per-rank token-sums ÷ global
token count" is *literally the same number* as the cp=1 loss — so gradients
match too. Empirical anchor: cp1 vs cp2 grad-L2 832.3 vs 845.8 (same order,
different data split); CP1-vs-CP2 loss parity 0.03%; CP32 131k descends
0.0381→0.0188 in 4 steps.

An earlier iteration had an extra ÷cp — it was wrong (gradients cp× too
small), and the removal is deliberate. Know this history; it shows the scaling
was *derived*, not guessed.

## 7. DSA-specific requirements (the submodule side)

- `cp_comm_type="allgather"` is **required**, not a choice: GLM's DSA indexer
  is frozen and selects top-k keys over the *global* sequence, so every rank
  needs the full KV (asserted upstream, `dsa.py:1807`). Ring/P2P CP variants
  don't apply.
- `variable_seq_lengths=True` because THD rows differ in length per microbatch.
- LoRA on the absorbed MLA KV up-projection is TP=1 only (upstream
  `038760cd8`) — consistent with the TP1 golden config.

## 8. What v1 deliberately does NOT do (say it before they ask)

- **No per-token logprobs to the client under CP** (`emit_logprobs=False`):
  each rank's logprobs cover only its zigzag slice; stitching them is future
  work. SFT gets loss only; RL/DPO are hard-rejected at request time.
- **No MTP**, weight-sync under CP>1 validated for export only.
- Per-request semantic nit: a `target_tokens` row whose *final* position has a
  real target is scored under CP but dropped at cp=1 (bshd path drops position
  S−1 unconditionally). Known, tiny, documented.

## 9. Rapid-fire review Q&A

- *Why pad every doc to 2·cp rather than pad the whole row?* Chunking is
  per-document (each doc is independently zigzag-split so attention stays
  within-doc); padding must make each doc's chunk size exact.
- *Why does packing produce ONE row per DP replica?* CP shards a single row
  across ranks; multiple rows would need per-row sharding bookkeeping for zero
  benefit — grad accumulation over sequential fwd/bwd calls provides batch.
- *What if a doc is shorter than 2·cp tokens?* It's padded up; pad positions
  carry −100/mask and drop out of loss and attention.
- *Do all CP ranks see the same data?* Yes — all CP ranks of a DP replica pack
  identically (the DP dataloader rank excludes CP), then each takes its shard.
- *Where's the unit-test coverage?* The packer invariants (pre-shift, padding,
  cu_seqlens, budget guard) are CPU-testable and tested; the zigzag
  index itself is TE/GPU-only, validated by the 131k probe runs and the
  exact-parity indexer harnesses (`glm_prof/topk_parity*.py`).
- *Why is the loss identical to cp=1 only up to ~1e-3?* Nondeterministic
  reduction order in the gather/FlashMLA path; the *definition* is identical,
  floating-point association isn't.
