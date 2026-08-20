# PACKING_MEMO — PP+CP microbatch semantics & pad-to-131k (volta, 2026-08-10 night)

Scope: trace of `server/src/trainers_server/dp_worker/` (megatron_bridge backend) from
`/forward_backward` datums → THD packing/partitioning → Megatron fwd-bwd call, under
PP2/CP8/EP8 @131k. File:line refs are worktree `~/Documents/wt-pp2-packing` @ df831501
(trainers) and `~/Documents/trainers/server/vendor/megatron-bridge` @ 20fcf2ea with
3rdparty/Megatron-LM @ 57efae08b (the LPS-1063 fix commit).

**TL;DR (3 sentences):**
1. The hard PP blocker is a config gate, not p2p: `_validate_thd_context_parallelism`
   raises on CP>1+PP>1 (`megatron_config.py:91-96`), while the actual p2p layer already
   tolerates unequal partition lengths because CP>1 sets `variable_seq_lengths=True`
   (`megatron_config.py:142-143`), which makes Megatron exchange tensor shapes on the
   wire per p2p op instead of assuming a static shape (`schedules.py:2095-2124`,
   `p2p_communication.py:322-328`). What PP *does* require is that all stages make the
   same **number** of per-partition schedule calls — guaranteed today by identical op
   broadcast + deterministic packing, but there is no runtime check: divergence = hang.
2. Pad-to-131k is sound at tip and should be implemented as **extend-the-last-document's
   padded tail** inside `pack_thd_cp_microbatch` (not a synthetic pad document — that
   breaks datum accounting at `training_runner.py:384-406`); pads are already fully
   loss-masked (labels=-100/weights=0/advs=0) and excluded from loss normalization and
   TPS accounting, but they **do** pay attention+MoE compute (the router does not
   exclude pads from top-k dispatch).
3. Virgin-path landmines found: (a) R3 router-replay drift check raises under PP when
   the sampler stamps `moe_layer_indices` (it compares the sampler's GLOBAL layer list
   against the stage's LOCAL routers, `loss.py:759-767`); (b) DSA indexer-loss AutoScaler
   path in the schedule assumes no PP (`schedules.py:374-389` TODO) — currently
   neutralized by `dsa_indexer_loss_coeff=0.0`, keep it that way; (c) `max_seqlen`
   semantics change when we tail-fill (currently max *padded* doc, becomes 131072 for
   every partition) — DSA uses it as a host-side coverage proof, believed safe, verify
   on-box.

---

## Q1 — How does one partition map to a microbatch? How is `num_microbatches` derived/communicated under PP?

Path: `MegatronTrainingRunner.forward_backward` (`training_runner.py:503`)
→ `dp_shard` (`training_runner.py:79-99`) gives this rank its contiguous datum slice
→ CP>1 → `MegatronBridgePacker.pack_thd_cp_microbatches` (`packer.py:70-191`)
→ `partition_thd_cp_datums` (`thd_cp.py:92-163`) greedily bins datums (each budgeted at
its `pad_multiple`-aligned length, `pad_multiple = 2·cp = 16` for CP8/TP1 —
`thd_cp.py:58-77`) into ordered partitions whose padded totals are ≤ `max_length`.
**One partition = one microbatch**: each partition is packed by
`pack_thd_cp_microbatch` (`thd_cp.py:166-360`) into a single flat THD row
(`input_ids` shape `(1, global_padded_len)`), then zigzag-sharded per CP rank
(`packer.py:145-189`).

The runner then calls the Megatron schedule **once per partition**, with
`num_microbatches=1`, `seq_length=microbatch.thd_global_seq_len` (this partition's own
padded global length — varies per partition), `micro_batch_size=1`
(`training_runner.py:359-367`). Grads accumulate across partitions because
`finalize_model_grads_func` is unwired for the loop and called once after
(`training_runner.py:309-318`, `400-401`).

So under PP, `num_microbatches` is **never communicated** — it is not even "1 schedule
with N microbatches"; it is N independent schedule calls with `num_microbatches=1`.
The pipeline-consistency requirement reduces to: **all PP stages must execute the same
number of schedule calls** (the 1F1B non-interleaved schedule structure per call is
identical given num_microbatches=1). That holds because:

- Every GPU rank receives the same op via `dist.broadcast_object_list` from rank 0
  (`worker.py:241,311`; `api/ops.py:4-6`).
- `dp_shard` is deterministic per `dp_rank`; PP stages at the same DP coordinate get the
  same slice. For PP2/CP8/EP8 on 16 GPUs, DP = 16/(TP1·CP8·PP2) = **1**, so every rank
  packs the full global batch identically.
- `partition_thd_cp_datums` is a pure deterministic function of the datum list.

There is **no runtime check** for this. If partition counts ever diverged across PP
stages (e.g. a future rank-dependent data filter), the failure mode is a p2p deadlock,
not an error. Note DP ranks may legitimately have different partition counts from each
other (empty slice → 0 partitions) — safe because there is no DP collective inside the
partition loop (`packer.py:82-88`); the once-per-op DP fold happens after
(`training_runner.py:446-463`) and the hoisted finalize joins everyone
(`training_runner.py:400-401`).

## Q2 — Where are p2p tensor shapes determined? Do UNEQUAL partition lengths break p2p?

Schedule selection: `get_forward_backward_func()` (`backend.py:349`) → PP>1, no virtual
PP → `forward_backward_pipelining_without_interleaving`
(`schedules.py:148-160` region, function at `schedules.py:2127`).

Tensor shapes are computed once per schedule call by `get_tensor_shapes`
(`schedules.py:2095-2124`):

- `config.variable_seq_lengths == True` → returns `[()]`; shapes are **exchanged on the
  wire** per p2p op (`_communicate_shapes`, `p2p_communication.py:~220-275`; the recv
  buffer is allocated to whatever the sender announces, `p2p_communication.py:322-345`).
- else → static `(seq_length // cp_size [/ tp_size if SP], micro_batch_size, hidden)`.

Our config **sets `provider.variable_seq_lengths = True` whenever CP>1**
(`megatron_config.py:142-143`, "THD packed lengths differ per forward call"); the field
is declared on the base config (`model_parallel_config.py:303`) and read off the model
config by the schedule. The `seq_length` argument is explicitly ignored in this mode
(schedule docstring, `schedules.py:~112`).

**Consequence: unequal partition lengths do NOT break or hang p2p at tip** — the
receiver learns each activation's shape from the sender. Jack's constraint 2 ("same size
microbatch required between PP ranks") is therefore *not* a hard p2p requirement at tip
under CP>1; pad-to-131k remains worth doing for uniformity (predictable memory/kernel
behavior per partition, representative warmup, and insurance against any
static-shape-assuming code in less-traveled paths), but the bring-up blocker is the
config gate (Q5.1), not shape mismatch. The true hard requirement is the *call count*
equality from Q1.

Caveat found while checking this: `variable_seq_lengths=True` is incompatible with the
`allgather` MoE token dispatcher (`transformer_config.py:2587-2592` raises). We never
combine them: trainer config `moe_token_dispatcher` is `"alltoall"` (default) or
`"flex"` (loops_models/control.py:342). Not a blocker, but don't introduce an allgather
dispatcher while CP>1.

Per-call p2p shape under PP2/CP8: the PP group links ranks with identical (tp, cp, ep)
coordinates, so each CP rank pipelines its own local shard; forward activations are
`(local_seq_len = global_padded/cp, 1, hidden)` per partition, exchanged with matching
dynamic shapes.

## Q3 — Do all PP ranks receive the same partition list / datum stream?

Yes. The op (with the full global datum list) is broadcast to every rank
(`worker.py:241,311`); every rank runs the identical `dp_shard` + partition + pack
locally (`training_runner.py:538-648`). Nothing about THD metadata crosses the p2p
boundary — only activations do. Each stage (including non-first) builds its own
`PackedSeqParams` with global `cu_seqlens`/`cu_seqlens_padded` (`packer.py:158-167`)
and passes `packed_seq_params` + `padding_mask` to the model on **every** stage
(`loss.py:205-210` CE, `622-625` RL, `1002-1005` DPO). `GPTModel.forward` accepts both
kwargs (`gpt_model.py:508-530`); on non-first stages `_preprocess` skips the embedding
(`decoder_input=None` → `set_input_tensor` from the schedule) while `padding_mask` is
forwarded to the decoder/MoE layers on all stages (`gpt_model.py` `_preprocess`/
decoder call; `transformer_block.py:635,677`). MLA (GLM) computes RoPE inside the
attention module, so the rope branch skipped on non-first stages is fine
(`gpt_model.py` `_preprocess`, `multi_latent_attention` guard).

## Q4 — Loss/token-weighting: where is the loss mask formed; will pads contribute?

Pad conventions in the THD packer (`thd_cp.py:277-315`): token id 0, label -100,
position 0, weights 0.0, logprobs 0.0, advantages 0.0, temperatures 1.0,
`padding_mask=True`.

Loss masks (last PP stage only — earlier stages return `(hidden, None)` and the
schedule never calls their loss_func, `loss.py:212-218`):

- CE: `active_mask = (labels != -100) & (weights > 0)` (`loss.py:142-147`).
- RL: `loss_mask = (advs != 0) & (targets != -100)` (`loss.py:533`).
- DPO: `active_mask = (targets != -100) & (w > 0)` (`loss.py:838`).

In all three, pads contribute exactly zero to `loss_sum` **and** to `num_active`
(the normalization denominator). The chunked LM head skips projection for inactive
positions, so pads don't even pay the LM-head GEMM. Loss reporting under PP is already
wired: `_loss_report` reduces over the CP group on the last stage
(`loss.py:57-81`), the runner broadcasts `[loss_sum, tokens, kl_sum, kl_count]` from
the last PP rank to the PP group (`training_runner.py:465-481`) and broadcasts
`loss_fn_outputs` from the last stage (`training_runner.py:675-689`).

TPS/telemetry accounting is pad-free: `input_tokens = sum(real datum lengths)`,
`loss_tokens = total active tokens` (`training_runner.py:703-711`).

**But pads are not free in the transformer body.** The MoE router does *not* remove pad
tokens from top-k dispatch: `padding_mask` only feeds z-loss / aux-loss / expert-bias
bookkeeping (`router.py:617-745`, `_apply_expert_bias` at `router.py:605-614`), all of
which are disabled in our config (aux coeff 0, expert bias frozen for LoRA). So pad
tokens are routed and consume EP all-to-all + expert GEMM capacity. Under R3 router
replay, pads additionally get synthesized distinct pad routes by design
(`datum_batch.py:214-251`). Attention likewise processes pad queries (garbage outputs,
loss-masked). Waste estimate in Part 3 must therefore charge pads ~full body FLOPs,
not just attention.

## Q5 — PP-specific code only exercised at PP>1 (virgin-path landmines)

1. **Hard gate (the bring-up blocker).** `_validate_thd_context_parallelism` raises on
   CP>1 + PP>1 (`megatron_config.py:91-96`), and separately on CP>1 +
   `overlap_grad_reduce` (`megatron_config.py:97-103`; keep that second one). Relaxing
   the PP clause for the GLM DSA stack is required before any launch.
2. **R3 router replay drift check is PP-broken when the sampler stamps
   `moe_layer_indices`.** `_set_sampler_router_replay_data` selects routers from the
   LOCAL stage's module tree (`.decoder.layers.N.`) but compares against the sampler's
   GLOBAL stamp (`loss.py:733-767`): on a PP stage holding layers 0..38,
   `layer_indices` = local MoE layers, `sampler_moe_layer_indices` = all 75 MoE layers
   → `RuntimeError` on every rank. Fix: check the stage-local subset (or union across
   stages). Not a bring-up blocker (bring-up is not R3), but blocks R3+PP RL later.
   Note the shape check at `loss.py:714-718` is fine: `config.num_layers` stays the
   global 78 under a layout, and the routes array is global-layer-indexed.
3. **DSA indexer-loss AutoScaler assumes no PP.** `forward_step_calc_loss` sets a
   process-wide loss scale for the experimental-attention-variant aux loss with an
   explicit TODO: "currently requires per-token loss and no PP"
   (`schedules.py:374-389`). Neutralized today: `dsa_indexer_loss_coeff=0.0`
   (`glm52_dsa.py:89`) ⇒ `use_indexer_loss=False` (`dsa.py:1892-1894`) ⇒
   `DSAIndexerLossAutoScaler` is never attached. **Do not enable the indexer loss
   under PP.** (MoE aux loss is similarly disabled at `megatron_config.py:41-42`;
   GLM MTP is disabled at `glm5_bridge.py:105`, so the MTP-under-PP path is dormant.)
4. **Embeddings are untied** (`share_embeddings_and_output_weights=False`,
   `glm5_bridge.py:93`) ⇒ no first/last-stage embedding grad all-reduce is needed;
   `finalize_model_grads`' embedding path no-ops for us
   (`finalize_model_grads.py:132-147` gates on shared embeddings/MTP). The runner's
   hoisted once-per-op finalize is PP-compatible (it does DP grad sync + SP layernorm
   grads; the schedule's in-call finalize is skipped while suppressed,
   `schedules.py:2466-2481` — no double finalize).
5. **Loss ×cp_size quirk (PP-agnostic, recorded for completeness).** The legacy 2-tuple
   loss path multiplies the loss by `cp_group_size` and divides by `num_microbatches`
   (`schedules.py:336-339`); with num_microbatches=1 per partition call there is no
   averaging artifact. DPO pre-divides by cp_size to compensate (`loss.py:958-964`).
   Absolute loss scale is normalized at `optim_step` via `dp_size/loss_tokens`
   (`training_runner.py:728-732`).
6. **LoRA adapter export is PP-aware in principle** — the bridge gathers adapter param
   objects across the PP group (`peft_bridge.py:555-618`) and the trainer runs the
   distributed export on every rank with rank-0 publish
   (`checkpoints/manager.py:362-378`). Still needs the on-box PP2 test (GOAL
   constraint 3, dedekind); the `(8,2)` layout exists for exactly that
   (`glm52_dsa.py:31`).
7. **Warmup already exercises the pipeline**: `run_startup_warmup` pushes a 64-token
   fwd-bwd through the same THD path (`backend.py:174-229`) — first PP p2p signal will
   appear at server start, before any client traffic.
8. **(78,2) layer layout is not in the table** (`glm52_dsa.py:30-44` has (8,2), (78,8),
   (78,16)). Per the NOTEBOOK topk-group math, 38/40 (stage 2 starts at 1-based layer
   39) is the valid split; even 39/39 is invalid. (gibbs owns this; restated here
   because the memo is the landmine list.)

## Q6 — Is tail-padded THD under CP sound at tip? Does our pad mechanism hit the LPS-1063 path?

Yes, with one precision about *which* path matters for GLM.

- The LPS-1063 fix is confirmed in the vendored pin: Megatron-LM 57efae08b,
  `transformer_engine.py:1830-1846` — under `qkv_format="thd"` + CP>1, when
  `cu_seqlens_*_padded != cu_seqlens_*` (tail-inclusive), it passes
  `pad_between_seqs=True` explicitly, engaging TE's exact `get_cu_seqlens_on_cp_rank`
  instead of the approximate `cu // cp` path. Any tail pad we add makes padded ≠
  unpadded, so the fix triggers.
- **But GLM-5.2 attention is 100% DSA** (`glm5_bridge.py:82-86,137`; even the first
  `first_k_dense_replace=3` layers run inside the DSA module with top-k skipped), and
  DSAttention does not go through TEDotProductAttention. DSA reads the **padded**
  cu_seqlens preferentially (`dsa_layout.py:295-320`) and builds its varlen bounds and
  CP allgather maps from the padded layout. Pad semantics there: pads are tail
  positions of their document's segment; causal masking means real queries never attend
  pad keys, so real-token outputs are bit-identical with or without pads; pad-query
  outputs are garbage but loss-masked (Q4). This is exactly the mechanism today's
  per-document `pad_multiple=16` tails already exercise on the validated GLM CP32
  golden config — pad-to-131k is the same path with a much larger tail. The TE fix
  covers the Nemotron/hybrid stack (`AttnBackend.fused`, `megatron_config.py:126-133`);
  for GLM it is irrelevant-but-harmless.
- CP sharding: `tex.thd_get_partitioned_indices` zigzags over `cu_seqlens_padded`
  (`thd_cp.py:363-389`) and requires each padded document length ≡ 0 mod 2·cp (=16).
  Filling a partition to 131072 preserves this: 131072 ≡ 0 (mod 16) and every padded
  doc length is already 16-aligned, so the last doc's extended tail stays aligned.
  Every CP rank then holds exactly 131072/8 = 16384 tokens per partition — uniform.
- **Recommended implementation shape: extend the LAST document's padded tail** so
  `cu_seqlens_padded[-1] == 131072`. Do **not** append a synthetic pad document:
  `partition_datums = cu_seqlens.numel() - 1` (`training_runner.py:384`) feeds the
  `datum_lengths` accounting (`training_runner.py:385-406`) and
  `thd_logprobs_to_loss_fn_outputs` (`packer.py:159-201`), both of which assume
  cu_seqlens describes real datums only; a zero-real-length document is also untested
  in the TE/DSA varlen paths.
- Watch item: `max_seqlen` is currently the max **padded** doc length
  (`thd_cp.py:320,346`). Tail-filling makes it 131072 for every partition. DSA uses
  `max_seqlen_q/kv` as a host-side "cu covers all rows" proof for the single-sequence
  case (`dsa.py:1747-1760`) — semantics remain correct (padded cu does cover all rows),
  and multi-document partitions don't take that branch. Verify on-box that no kernel
  heuristic regresses; if it does, consider reporting max **real** length instead
  (needs a correctness check of its own).

## Part 2 — implementation (landed, commit 36c3c8f4 on jackrao/lps-1062-pp2-packing)

Implemented as designed: `pack_thd_cp_microbatch(..., pad_to_length=)` tail-fills the
LAST document's padded region (`thd_cp.py`), threaded through
`MegatronBridgePacker.pack_thd_cp_microbatches` (`packer.py`), gated in
`MegatronTrainingRunner._thd_partition_pad_to_length` on PP>1 or `BT_PACK_PAD_TO_MAX`
(`training_runner.py`). Validation: `pad_to_length` must be a positive multiple of
`pad_multiple`, ≤ `max_length`, and ≥ the partition's own padded total; exact-fit is a
no-op. Tests: 7 new cases in `tests/unit/dp_worker/api/test_cp_thd_slicing.py`
(layout/masks/cu_seqlens, exact-fit no-op, None passthrough, validation errors,
optional fields + routed-experts tail) — green locally via standalone module load
(the `backends` package init pulls megatron, so the pytest file itself runs
on-box/CI per existing convention; 9 pre-existing cases re-verified against the
modified module).

**Warmup note for gibbs:** under PP2 the startup warmup's 64-token datum now tail-fills
to 131072, so `run_startup_warmup` becomes a full-size fwd-bwd probe — slower (one
real-shape 131k step) but it warms the actual kernels/shapes and fails fast at startup
if the config can't hold 131k. Deliberate; don't mistake the longer warmup for a hang.

**Parity protocol (for on-box execution, gibbs):** goal is loss(padded) == loss(unpadded)
within tolerance, proving the tail is inert end-to-end.

1. CP8/PP1, fixed datum set (mix of short/medium lengths, e.g. 32×8k + 4×40k), CE loss
   with explicit per-position `weights` (include zero-weight spans to exercise masking).
   Run A: default (data-dependent partition lengths). Run B: `BT_PACK_PAD_TO_MAX=1`.
   Compare: per-call `loss` (the token-weighted global mean) must match to fp32
   all-reduce tolerance (~1e-6 rel); `loss_fn_outputs` logprobs per datum must match
   within bf16 forward tolerance (~1e-3 abs) — pads change the THD row layout, not the
   math on real tokens; `loss_tokens` (active count) must be EXACTLY equal.
2. Same at CP8/PP2 vs CP8/PP1 (both padded): last-stage-broadcast loss and per-datum
   logprobs must match the PP1 reference the same way — this is the real PP+CP parity
   gate for the bring-up's real-data phase.
3. Grad check (optional but cheap): one `optim_step` after identical fwd-bwd on both
   configs, then compare a small set of LoRA parameter grads (or the resulting
   `grad_norm`) — must match within bf16 accumulation tolerance.
4. DPO smoke (optional): one paired batch, `atomic_row_group_size=2` path with padding —
   pair counts and loss must match unpadded.

Any mismatch beyond tolerance means the tail is leaking into real-token math — prime
suspects in order: (a) a code path reading `cu_seqlens_padded` as if it were real
boundaries, (b) `max_seqlen` heuristics in the DSA kernel, (c) loss-mask sentinel drift
(weights/advantages not zeroed on the tail).

## Part 3 (stretch) — waste estimate

Customer histogram: 70% ≤32k / 28.5% 32–64k / 1.4% ≤131k. With greedy first-fit
partitioning at 131k and tail-fill padding, expected padded-token overhead ≈
E[partitions]·131k vs E[real tokens]. Rough bin estimate (method: pack expected
composition per 131k partition): a partition typically holds ~4×32k-class docs →
real ≈ 4·(32k·avg_fill) vs 131k padded. Using class midpoints (16k / 48k / 96k) as
per-datum means: E[real per token] ≈ 0.70·16k + 0.285·48k + 0.014·96k ≈ 26.2k; per
131k partition fits ≈ 131/26.2 ≈ 5.0 datums ⇒ real ≈ 131k·(26.2/26.2)·(packing
efficiency ~0.9 from bin-packing slack at partition boundaries) ⇒ **pad waste ≈ 10%
+ per-doc 16-alignment (<0.1%)**, i.e. low-teens % worst case, ~8–12% typical — at the
top of Jack's "low-single-digit" hope only if the real distribution packs tighter than
the midpoint model. Charged at near-full body FLOPs per pad token (Q4), so this is the
MFU hit; refine with the real histogram when we have it.
