# LPS-1062 — MoE all-to-all ⇄ compute overlap: design memo

**Author:** helmholtz (design deep-dive, code-read only) · **Date:** 2026-08-09
**For:** fibonacci (orchestrator) · **Ticket:** LPS-1062
**Code read:** Mac checkout `~/Documents/trainers` @ vendored mcore 0.19.0
(`server/vendor/megatron-bridge/3rdparty/Megatron-LM/`), line-identical to the
on-box pin for the dispatcher files (verified by ramanujan in the FIX-B/F project).
**Calibration inputs:** fibonacci supplements 1–2 (gated-v2-4mb-steady trace:
step 46.678 s; token-a2a 20.90 s/1800 calls; probs a2a 5.13 s/900 calls;
compute union 16.99 s; overlap 0.395 s; per-layer anatomy dispatch 10.7 →
probs 5.7 → GEMM 4.9 → combine 12.5 → attn window 16.6 ms).

---

## 0. TL;DR

Three orthogonal levers, in ship order:

| # | Lever | Gate | Win (est.) | Confidence | Risk |
|---|-------|------|-----------:|------------|------|
| **W1** | De-serialize the probs A2A onto a **second NCCL communicator** (same ranks) | `BT_MOE_PROBS_A2A_COMM=1` | **−5.1 s/step** (measured mechanism) | High | Low |
| **W2** | Intra-MoE-layer chunked pipeline, **K=2 by local-expert groups**, implemented with **list-A2A over per-peer views** (zero copies, bitwise-safe by construction) | `BT_MOE_A2A_PIPELINE=2` | −3.0…−4.4 s/step standalone; −1.5 s if W3 lands | Medium | Medium |
| **W3** | **Lookahead recompute**: overlap recompute-fwd(L−1) with bwd(L) across layers inside full recompute | `BT_MOE_LOOKAHEAD_RECOMPUTE=1` | −8…−12 s/step | Medium-low | High |

Stacked model: 46.7 → **~31–34 s** (~1000–1100 tok/s/GPU). If W3 is blocked:
46.7 → ~41.5 s with W1+W2.

> **Field-calibration caveat (2026-08-09, W1 on-box):** W1's mechanism
> confirmed exactly (probs off-stream 900/900, −2.07 ms/pass gap delta) but
> the wall converted at ~0.15× the static model. The CDMC=1 framing is
> RETIRED (both boxes ran unset — the =1 pin was a stale .orig script); the
> open mechanism is autograd engine issue order in backward windows (hilbert's
> launch-vs-kernel-start cut arbitrates). All wall-second figures in this
> memo are mechanism bounds pending that read; see §6.1.

On the brief's variant question: **variant (i) (row-chunking the permuted
matrix) is unsound** — row chunks of the expert-major permuted buffer are
contiguous *destination-rank* ranges (degenerate: half the ranks receive
nothing per chunk) unless chunks cut through expert row-blocks (breaks GEMM
bitwise parity). **Variant (ii) (chunk by local-expert groups) is the only
sound intra-layer decomposition**, and the key implementation insight is that
`torch.distributed.all_to_all` (the *list* form) over per-peer **views** of the
existing permuted buffer realizes it with **zero copies and zero layout
changes**, so the final unpermute sees a byte-identical buffer in the
byte-identical order. This overturns the prior ranking "(i) > (ii)": (i) as
stated cannot be built; (ii) is what ships.

---

## 1. Baseline anatomy (measured, post-B+F steady state)

Per MoE layer-pass (900 = 75 layers × 4 datums × 3 passes per step):

```
permute1 ─► dispatch a2a 10.7 ms ─► probs a2a 5.7 ms ─► sort ─► GroupedMLP 4.9 ms
          ─► unsort ─► combine a2a 12.5 ms ─► unpermute ─► [attn/norm/router 16.6 ms]
```

- Token A2A: 1800 calls, 20.90 s. Dispatch-size 903 @ avg 10.7 ms (min 4.3 ms =
  805 MB @ 187 GB/s aligned; p50 carries ~2.3× wait-slack from rank imbalance).
  Combine 897 @ avg 12.5 ms (up to 3.5× imbalance).
- Probs A2A: 900 calls, 5.13 s, avg 5.7 ms for ≤0.23M-elem f32 (~66 KB) —
  pure latency, serialized on the **same NCCL stream (stream 83)** as tokens.
- A2A⇄compute overlap today: 0.395 s (1.5 %) — only the tiny shared expert.
- CP AG/RS on compute stream ~2.2 s; GPU idle 1.82 s.

Why the expert GEMM waits for probs today (supplement-2 question): **both**
stream serialization and data dependency. In `token_dispatch`
(`token_dispatcher.py:691-710`) the tokens A2A does `handle.wait()` (compute
stream waits), then shared-expert fc1, then the probs A2A does `handle.wait()`
(compute stream waits again). Downstream, `dispatch_postprocess`
(`:768-774`) calls `fused_sort_chunks_by_index_with_probs` which **consumes
`global_probs`**, and the scaled activation inside TEGroupedMLP consumes
`permuted_probs` (`experts.py:734/787`). So probs must land before the sort —
the 5.7 ms is a hard serial slot, not just an unlucky schedule. W1 removes it
by putting probs on its own communicator so it lands (~5.7 ms) before the
tokens (~10.7 ms) it is serialized behind today.

---

## 2. Q1/Q2 — variant analysis: why expert-group chunking (ii), and why row-chunking (i) is unsound

### 2.1 Layout facts (from the code)

`MoEAlltoAllTokenDispatcher` (all refs `token_dispatcher.py`):

- `preprocess` (`:488-609`): `num_local_tokens_per_expert = routing_map.sum(0)`
  (`:530`); `input_splits = counts.view(ep, 16).sum(1)` (`:550-552`);
  `num_global_tokens_per_local_expert` [tp·ep, 16] (`:565-579`);
  `output_splits` (`:573`). All device-side; one batched side-stream D2H + one
  `d2h_event.synchronize()` (`:922-961`).
- `dispatch_preprocess` (`:611-666`): TE fused `permute` with probs → buffer
  `P` in **global-expert-id order**: `[rank0: e0..e15][rank1: e16..e31]…`.
  `input_splits` carve `P` into 16 contiguous per-dest blocks.
- `token_dispatch` (`:668-712`): one `all_to_all_single` for tokens (805 MB
  bf16 at 131k), one for probs (f32), both on the EP PG's internal NCCL stream
  (`use_nccl_stream=True`, `mappings.py:449-458`), `handle.wait()` = compute
  stream waits (non-blocking on host).
- `dispatch_postprocess` (`:714-780`): `sort_chunks_by_idxs` (TE fused,
  `:768-774`) re-sorts recv buffer from (src, expert) block order to
  expert-major using `num_global_tokens_per_local_expert` and the precomputed
  `sort_input_by_local_experts` (`:422-432`).
- `TEGroupedMLP.forward` (`experts.py:650-819`): unfused path is active in the
  golden run (`use_transformer_engine_op_fuser` defaults False,
  `gpt_provider.py:168`, not overridden in `megatron_config.py`).
  `tokens_per_expert.tolist()` (`experts.py:678`) — host list already required
  today. FP8: `Fp8Padding` pads per-expert counts to `align_size`
  (`:582-592, 676-689`).
- Combine is the exact mirror (`:782-860`); `combine_postprocess`
  (`:862-909`) unpermutes with `reversed_local_input_permutation_mapping`.

### 2.2 Variant (i) — row chunks of `P`: unsound

`P` is expert-major and experts map contiguously to ranks (16/rank), so a
contiguous row range of `P` is a contiguous *expert* range, hence a contiguous
*destination-rank* range:

- **Expert-aligned row chunks** (the only GEMM-bitwise-safe row cuts): chunk c
  covers dest ranks [c·8, c·8+8). In chunk c's A2A, ranks outside that range
  receive **zero** rows — half the EP group idles per phase; each rank does its
  whole GroupedMLP in one chunk anyway. The pipeline degenerates; you get two
  serialized half-size A2As with no compute to hide them behind on the
  receiving side. ✗
- **Non-aligned row chunks** cut an expert's row block across two A2A calls →
  the expert's rows reach the consumer in two pieces → GroupedMLP runs that
  expert twice with different M → per-expert GEMM kernel schedule can differ →
  **not bitwise** vs today (§5b). Also breaks the recv-side
  `sort_chunks_by_idxs` metadata model. ✗

### 2.3 Variant (ii) — chunk by local-expert groups: sound, and free via list-A2A views

Chunk g = local experts `[g·L, (g+1)·L)` on **every** rank (L = 16/K; K=2 →
8+8). Every rank sends and receives in every chunk (balanced), and every
expert's row block stays whole in one GEMM call (§5b).

The naive implementation (re-sort `P` into group-major order, or permute the
routing-map columns) costs either an extra 805 MB gather per direction or
changes the final unpermute's per-token reduction order (§5a). **Neither is
needed.** The observation:

- In `P` (expert-id order), group g's rows for dest rank r′ are the contiguous
  slice `P[base(r′) + off_g(r′) : … + cnt_g(r′)]` — group 0 is the *first half*
  of every per-rank block, group 1 the second.
- `torch.distributed.all_to_all(output_list, input_list, group, async_op=True)`
  (list form) takes arbitrary contiguous per-peer tensors → **views** of `P`.
  On the NCCL backend this lowers to the same grouped SendRecv that
  `all_to_all_single` with unequal splits already produces (the trace's
  `ncclDevKernel_SendRecv` confirms the transport).
- Recv side per group: `recv_g` is one contiguous buffer in (src,
  expert-in-group) order; per-peer outputs are narrows of it.
- Combine side per group: after unsort, per-peer inputs are narrows of
  `unsorted_g`; per-peer outputs are narrows of `combine_buf` at
  **today's** offsets (`base(src) + off_g(src)`), so after both groups land,
  `combine_buf` is **byte-identical in content and layout** to today's single
  combine output → the unpermute in `combine_postprocess` runs **unchanged**
  on unchanged input (§5a).
- All offsets are host-computable from the two count matrices already produced
  in `preprocess` — carried to host by the **same single D2H batch + single
  event** as today (§6).

Cost: zero extra data-movement kernels, zero extra memory (all views), K−1
extra NCCL group launches per A2A phase (µs each at 400 MB message sizes —
bw-bound, negligible).

**General K:** K ∈ {2, 4, 8, 16} (K | 16). K=2 is the sweet spot (§8).

---

## 3. The W2 design (intra-layer chunked pipeline, K=2)

### 3.1 Forward pipeline (per MoE layer, per pass)

Streams: `compute` (default), `nccl-tok` (EP PG internal), `nccl-probs` (W1's
second PG), `shared` (SharedExpertMLP.stream), `dtoh`.

```
compute:  permute1 (unchanged, expert-id order)           [P ready]
host:     d2h_event.synchronize()  ← the ONE sync (§6); compute all
          per-group offsets/splits on host from the [ep,16] + [tp·ep,16] matrices
issue:    DispatchA2A_0 (list-a2a, views of P → recv_0)   async on nccl-tok
          DispatchA2A_1 (views → recv_1)                  async on nccl-tok
          probs A2A (one call, full)                      async on nccl-probs   [W1]
          shared_experts.linear_fc1_forward_and_act       on shared stream
compute:  WaitD_0 → sort_0 (+probs_g0 gather, 131 KB) → ChunkMLP_0 (8 experts)
          → unsort_0 → issue CombineA2A_0 (list-a2a → combine_buf views)
compute:  WaitD_1 → sort_1 → ChunkMLP_1 → unsort_1 → issue CombineA2A_1
          shared_experts.linear_fc2_forward + post_forward_comm   (shared stream)
compute:  WaitC_0, WaitC_1 → unpermute (UNCHANGED, combine_buf byte-identical)
```

Timeline (measured numbers, K=2, half messages ≈ 5.4–6.3 ms): nccl-tok runs
d0 [0,5.4], d1 [5.4,10.8], c0 [10.8,17.1], c1 [17.1,23.4]; compute: sort_0
[5.4], MLP_0 [5.6,8.1], wait d1 → 10.8 (exposed 2.7), MLP_1 [10.9,13.4],
wait c0 → 17.1 (exposed 3.7), unpermute waits c1 → 23.4 (exposed 6.3).
Exposed comm ≈ 5.4 + 2.7 + 3.7 + 6.3 ≈ **18.1 ms vs 23.2 today → −5.1 ms per
layer-fwd** (fwd+replay: −3.0 s/step; bwd v1 ≈ neutral, v2 −1.2…−1.5 s; §3.4).
The c1 tail is structurally exposed (no MoE compute left in the layer) — this
caps W2 at roughly the 4.9 ms GEMM window, matching fibonacci's bound.
De-skew upside (chunk-1 posts earlier; p50 → 4.3 ms floor) is **on top** of
this static math (supplement 1, note 2).

### 3.2 Metadata (no new host syncs — Q6)

Today one D2H batch carries `tokens_per_expert, input_splits, output_splits,
output_splits_tp, num_out_tokens` (`:935-950`) behind one `d2h_event`. Under
the gate, **add two tensors to the same batch**: `num_local_tokens_per_expert`
[ep,16] (already computed at `:530`) and `num_global_tokens_per_local_expert`
[tp·ep,16] (`:565-599`) — 2–4 KB total. Host then derives, with zero GPU
interaction: per-group `input_splits_g`, `output_splits_g`, per-peer view
offsets (`base/off/cnt` for dispatch and combine), per-group
`tokens_per_expert` slices (host list slices for `experts.py:678`).
**Sync count unchanged: one per layer-pass.** (Bonus: this is the same D2H
FIX C keys on; no interaction.)

Device-side per-group sort metadata: `num_global_tokens_per_local_expert[:,
gL:(g+1)L]` (device slice, fused `sort_chunks` takes device split tensors —
same as today `:770`) and precomputed per-group index permutations
(`sort_input_by_local_experts`/`restore_output_by_local_experts` restricted to
L experts — one-time in `__init__`, same pattern as `:422-432`; the index
*pattern* is identical for all g, only split sizes differ).

### 3.3 Autograd structure (Q3)

New `torch.autograd.Function`s (dispatcher-local, mirroring
`mappings.py:424-484`):

- `_ListA2AIssue`: fwd — build per-peer view lists from host offsets, call
  `torch.distributed.all_to_all(out_views, in_views, group, async_op=True)`,
  stash `work` + splits in ctx, return the recv buffer. bwd — issue the
  reverse list-a2a (swapped splits, grad_recv views → grad_send buffer), then
  (v1) `work.wait()` + `compute.wait_stream`-equivalent before returning
  (simple; bwd ≈ today's exposure); (v2) no wait, attach the completion event
  to the returned grad tensor via setattr for wait-aware consumers (§3.4).
- `_A2AWait` (identity): fwd — `work.wait()` (compute-stream wait, host
  non-blocking); bwd — identity. Placed between issue and each consumer so the
  fwd pipeline holds; invisible to numerics.
- `TEGroupedMLP` chunk shells: `build_chunks(K)` constructs K
  `te.pytorch.GroupedLinear(num_gemms=L, …)` modules on `device="meta"` and
  attaches the **same Parameter objects** (`weight{i} = orig.weight{gL+i}`),
  mirroring the existing `_make_fused_ops` weight-sharing precedent
  (`experts.py:427-437`), stored in a plain tuple (not registered submodules —
  no DDP/optimizer surface). Experts are **frozen** (attention-only LoRA) → no
  wgrad flows; `backward_dw()` on the wrapper calls each chunk's
  (`experts.py:861-895`). If `use_transformer_engine_op_fuser` is ever on, the
  same shell pattern wraps `te.pytorch.ops.Sequential` per chunk
  (`experts.py:382-532`).

**Recompute compatibility:** fwd runs twice — no-grad first pass
(`CheckpointFunction.forward`, `random.py:564-595`) and grad-enabled replay
(`random.py:598-634`). Chunking is a pure function of `routing_map`, which the
replay recomputes bit-identically from saved RNG → identical chunks, identical
views. All per-pass state lives on the dispatcher instance and is cleared by
`_clear_forward_state` (`:894-908`) in both passes, as today. The new
Functions are no-grad-safe (first pass builds no graph). CUDA-graph capture:
`checkpoint()` already passes through during capture (`random.py:642-649`);
v1 **falls back** to the unchunked path when `moe_preprocess` is in
`cuda_graph_modules` (assert + loud log), cudagraph support is follow-up.

### 3.4 Backward pipeline (v1 simple, v2 pipelined)

Bwd chain per group: `CombineA2A_g.bwd` (rev list-a2a) → `unsort_g.bwd` (TE
fused sort's own inverse — deterministic gather) → `ChunkMLP_g.bwd` (**dgrad
only**, 2 GEMMs; frozen experts → no wgrad; the `_delayed_wgrad` machinery
`moe_layer.py:737-798` stays idle as today) → `DispatchA2A_g.bwd` (rev
list-a2a) → `permute1.bwd` (TE, unchanged).

- **v1:** each `.bwd` does issue+wait. Exposed ≈ 4 half-A2As ≈ 23 ms ≈ today
  (neutral). Ships with the fwd win.
- **v2 (seq-bump pipelining, same mechanism as the shared-expert overlap,
  `shared_experts.py:302-319, 585-593`):** bump `CombineA2A_0`'s grad-fn
  sequence number above `MLP_1`'s (both combine-rev A2As issue before the
  unsort/MLP bwds; rev-c(0)'s flight hides under `MLP_1.bwd`), and bump
  `DispatchA2A_1` above `MLP_0` (rev-d(1) hides under `MLP_0.bwd`). Consumers
  become wait-aware via events attached to grad tensors (custom `Unsort`/
  `Slice` wrappers). Guarded by `grad_fn is not None` (replay only), tie-safe
  (any order among independent ready nodes is correct). Expected bwd exposure:
  ~2 half-A2As instead of 4 → −6…−8 ms per layer-bwd.
- The autograd engine runs higher sequence numbers first; dependencies gate
  readiness regardless, so the bumps cannot corrupt order — only pipeline it.

### 3.5 Streams/events and `CUDA_DEVICE_MAX_CONNECTIONS` (Q5)

Trainers pins `CUDA_DEVICE_MAX_CONNECTIONS=1`
(`experiment_artefacts/glm/scripts/run_trainer_node.sh:36`): all streams share
one hardware channel, so **any stream-wait pushed into the channel
head-of-line-blocks everything pushed after it** (this is why the existing
code orders shared-fc1 between the two A2As, `token_dispatcher.py:698-703`).
Design rule (both W1/W2): **push every independent op (all A2A issues on all
comms, shared-expert kernels) before the first compute-stream wait.** The §3.1
push order satisfies it: `WaitD_0` is pushed only after d0/d1/probs/shared-fc1;
`WaitD_1` after `issue c0`; `WaitC_*` after `issue c1` + shared-fc2. Events:
one `torch.cuda.Event` per chunk per direction per pass, allocated fresh in
`dispatch_preprocess` (µs; cleared with the rest of forward state). The
existing `cuda_dtoh_stream` (class-level, `:380/461-462`) and
`SharedExpertMLP.stream` (class-level, `shared_experts.py:107/191-193`) are
unchanged; contention is trivial (KB-scale D2H; 0.3 ms shared-expert kernels).

### 3.6 Memory (Q7)

W2 adds **≈ 0 GiB**: recv_0/recv_1 are halves of today's recv footprint;
`combine_buf` equals today's combine output; all per-peer tensors are views;
events are negligible. 131k: peak stays ~197/268.6 GiB. 16k×d32: rows/rank
262,144 → the same three ~3.2 GB buffers as today, no delta. (W3 memory in §7.)

---

## 4. W1 — probs A2A de-serialization (fast path, ships first)

Options assessed (per the brief):

- **(a) Second communicator (CHOSEN):** `new_group` over the EP ranks at
  dispatcher init (one extra NCCL comm total — the EP PG is shared across
  layers). `token_dispatch` issues tokens-A2A (comm 1) and probs-A2A (comm 2)
  back-to-back async, shared-fc1 between issue and waits (push-order rule
  §3.5), then waits both. Probs lands ~5.7 ms after issue, **before** the
  tokens it currently serializes behind — the sort's data dependency (§1) is
  satisfied in time. Win ≈ 5.7 ms × 900 = **5.1 s/step**. Backward rides
  comm 2 automatically (`_AllToAll.backward` re-applies with `ctx.group`,
  `mappings.py:470-484`) and stops occupying stream 83 between the two token
  rev-A2As.
- (b) Pack probs into the token payload (row-append 4 B to each 12288 B row):
  bitwise-safe (bytes moved verbatim) but needs an 805 MB pack + unpack copy
  (~0.3 ms each per pass) and byte-view gymnastics on splits; strictly worse
  than (a) which is copy-free. Rejected.
- (c) Reorder probs before tokens on the same comm: tokens delayed 5.7 ms →
  GEMM start unchanged (sort needs both). Zero win. Rejected.

W1 is independent of W2 (with W2, probs still ride comm 2 as **one** full
call; per-group `probs_g` are produced by a 131 KB device gather from
`global_probs` — cheaper than K latency-bound A2As).

---

## 5. Q4 — bitwise-parity argument, per tensor

Hard constraint: results must be bitwise-identical to today. Status by tensor:

**(a) Final unpermute (combine_postprocess) — SAFE BY CONSTRUCTION.** Today
the combine output layout is `[src0: my rows for src0's experts, id
order][src1: …]` = exactly the pre-dispatch permuted order, which the reversed
mapping undoes. W2 writes each group-chunk's combine output into
`combine_buf` at today's per-src offsets (`base(src)+off_g(src)`; group 0 is
the first half of every src block because expert-id order puts local experts
0–7 before 8–15 within each rank block). After both groups: **byte-identical
content in byte-identical positions** → the single unchanged
`fused_unpermute` call produces bitwise-identical output **regardless of its
internal per-token reduction order**. (This is why list-A2A views beat the
column-permuted-permute variant, which would reorder some tokens' top-8
reduction sequences — e.g. experts (r0,e9) vs (r1,e2) swap relative order —
and was rejected.)

**(b) Grouped GEMM per expert — SAFE BY CONSTRUCTION, one verification.** Each
expert's row block is whole in exactly one chunk call with unchanged M_e, N,
K, weights, and input values (A2A moves bytes verbatim; sorts are gathers).
Per-expert output rows are therefore identical **provided** the GEMM kernel
schedule for a given (M_e, N, K) doesn't depend on `num_gemms` in the call (16
vs 8). TE/cuBLAS pick per-problem algorithms by shape; expectation: identical.
**Verification V1** (torch.equal on one layer, gate off vs on) settles it.
FP8: recipe is `Float8BlockScaling` (`extensions/transformer_engine.py:241/311`
— stateless per-call 128×128 block scales, no amax history). Per-expert
`Fp8Padding` alignment (`experts.py:582-592`) is unchanged per expert, so each
expert's padded row count is unchanged and every 128-row block boundary lands
on the same rows (chunk-g's buffer starts at a multiple of align_size ≥ 128).
Quantized inputs bitwise → GEMM bitwise.

**(c) Probs A2A — SAFE.** Byte transport only; W1 changes the communicator,
not the bytes. The per-group probs gather is a copy (exact). Probs are
consumed by the fused sort and the per-row scaled activation
(`experts.py:734/787`) — row-local ops.

**(d) TE fused permute / sort_chunks — SAFE.** Both are gathers/copies
(row-independent); no cross-row reduction exists anywhere in the dispatch path
except the final unpermute (covered in (a)) and the router (untouched).
`sort_chunks` per group reorders whole rows with device-computed splits — the
same rows land in the same per-expert order as today (stable, deterministic
kernel, same input order).

**(e) Backward — SAFE.** dgrad GEMMs: same per-expert argument as (b).
`unsort/sort` backwards are inverse gathers (deterministic). The two rev-A2As
move grad bytes verbatim. `permute1.bwd` (scatter-add over the full buffer)
sees the same values in the same layout (v1: engine-accumulated group grads —
the add of disjoint zero-padded halves is exact; v2: non-overlapping slice
writes into one buffer — exact). No wgrad (frozen experts). Router/attention
untouched.

**Residual non-bitwise risk:** only V1 (kernel-schedule dependence on
`num_gemms`). If V1 fails, the fallback is K-chunking at the *call* level with
a padded-to-16 `tokens_per_expert` (zero rows for out-of-chunk experts) —
empty GEMMs still launch; investigate then. Do not ship without V1 passing.

---

## 6. Q8 — win model and K selection

Per layer-pass exposed comm today: 23.2 ms (dispatch 10.7 + combine 12.5) +
5.7 probs. Per layer-pass hideable compute (MoE-only slice of the 17 s union):
GEMM 4.9 ms + permute/sort/unpermute ~1 ms ≈ **5.9 ms** (the rest is
attention/router, not adjacent to the A2As).

- **W1:** −5.7 ms × 900 = **−5.1 s** (measured mechanism; high confidence).
- **W2, K=2:** fwd/replay −5.1 ms each (§3.1), bwd v1 ≈ 0, v2 −6…−8 ms.
  Standalone: **−3.0 s (v1) … −4.4 s (v2)**. Under W3, the 600 bwd-phase
  passes are cross-layer covered; W2's marginal value drops to the 300 fwd
  passes ≈ **−1.5 s**.
- **K choice:** K=2 keeps messages bw-bound (402 MB ≫ latency floor) and gives
  the GEMM halves (2.45 ms) something to hide. K=4 quarters GEMM (1.2 ms) —
  can't cover even one 3 ms chunk flight; per-call overheads grow; de-skew
  might argue for it *only* if measured p50 stays ≫2× floor after K=2. Ship
  K=2, leave K=4 behind the same gate for one experiment.
- **De-skew upside (not in the base model):** dispatch p50 10.7 ms vs 4.3 ms
  floor = 2.3× wait-slack from rank-imbalanced posting times. Chunked,
  earlier-posted, smaller messages compress toward the floor; if p50 → ~7 ms,
  a further −2…−4 s/step accrues to W2/W3. Static overlap math above is the
  lower bound (per supplement 1).

### 6.1 Field calibration (W1 on-box, 2026-08-09 — fibonacci + hilbert)

W1 measured on-box: **mechanism 100 % confirmed** (probs off the EP stream
900/900; dispatch→combine gap 11.22 → 9.15 ms = −2.07 ms/pass; token A2A
flat; canaries clean) **but the wall moved only −0.46 s vs the −3.4 s
(CDMC=1) model.**

**Diagnosis — CORRECTED TWICE (2026-08-09):** (i) my W1 patch note's
backward de-serialization caveat was directionally right; (ii) the
"CDMC=1 head-of-line" framing was then RETRACTED — both boxes (318g61w AND
qr4ggv3) ran CDMC **UNSET** (/proc-verified; the =1 pin lives only in a
07-28 `.orig` script — my `run_trainer_node.sh:36` find was the stale copy).
The backward-window late starts happen at 8 connections, so the leading
mechanism is now **autograd engine issue order** (the probs-reverse node
executes late in the backward window), pending hilbert's
launch-vs-kernel-start cut to confirm issue-vs-start.

**Revised model:** CDMC unset (production, AND the actual bench state): W1 ≈
the full −5.1 s/step modulo the backward-window issue-order exposure hilbert
is quantifying. The =1 devbox framing in earlier revisions is retired (no
box runs =1).

Consequences for every magnitude claim in this memo:

1. **The CDMC question is settled (unset everywhere that matters); the open
   question is engine issue order in backward windows.** The W1-v2 contingent
   patch (seq-bump / combined dispatch Function) targets exactly this; its
   trace-checkable prediction is the bwd-window probs late-start fraction →
   ~0. ARM 4's readout calibrates how much of the backward-phase models
   (W2-bwd, W3) is exposed the same way.
2. **W2-intra:** forward/replay win stands as measured (−2.07 ms/pass gap
   delta class); backward-phase value depends on the same engine-order
   question.
3. **W3:** the CDMC precondition is satisfied as-is (unset everywhere) — W3
   proceeds on tonight's ladder. Its kicks are explicit host pushes of whole
   recompute windows, robust to engine-order drift by construction (the
   kicked recompute is not engine-scheduled); the bwd(L) side it overlaps
   with IS engine-scheduled, so the same issue-order exposure applies to
   half the overlap pair — read ARM 4 before quoting the W3 seconds.
4. Fabric tails and CDMC=1 contention are both retired as binding
   constraints; the de-skew upside note in §6 stays as upside, not base.

---

## 7. W3 — lookahead recompute (cross-layer overlap in the bwd phase)

2/3 of A2A time lives in the bwd phase (300 replays + 300 bwds of 900 passes).
Structure today: `CheckpointFunction(L).backward` = recompute-fwd(L) (~23 ms
comm + ~18 ms compute) then `bwd(L)` (~23 ms comm + ~30–35 ms compute) —
serial ~105–110 ms per layer, comm and compute mutually exposed.
**Layer L−1's recompute depends only on its own saved input (produced in the
fwd phase, long since ready) — not on bwd(L).** So kick recompute(L−1) at the
boundary between recompute(L) and bwd(L): the two windows overlap
1F1B-style; comm stream 83 serializes both A2A chains (stays ~100 % busy —
fine, per supplement 2 note e) while compute streams stay fed. Floor per
layer-bwd-phase ≈ max(46 comm, ~50 compute) + tails ≈ 55–65 ms vs ~105–110 →
**−40…−50 ms × 300 layer-bwds; at 60–80 % capture: −8…−12 s/step.**

Design (single host thread — the autograd device worker — no worker-thread
races):

- New `LookaheadCheckpointFunction` (copy of `random.py:555-634` + the fp8
  snapshot bits of `CheckpointWithoutOutputFunction`, `random.py:658-682`) with
  backward split into: `_ensure_recomputed(L)` (pop the stashed graph for L if
  kicked earlier, else recompute now) → `_kick(L−1)` → `torch.autograd.backward`
  on L's graph.
- Registry on the `packed_seq_params` carrier (per-microbatch lifetime — the
  FIX-B/C pattern, `recompute.py:40-127`, `dsa.py:83-94`): maps global layer
  index → (saved args, rng states, run_function `cf` from
  `recompute.py:162-215`). Populated in the first pass; dies with the
  microbatch. Single-microbatch-in-flight asserted (true for the loops
  controller).
- `_kick(L−1)`: on the autograd thread, between recompute(L) and bwd(L):
  `_fork_rng` + `_set_all_rng_states(ctx[L−1].rng_states)` (verbatim from
  `random.py:614-621`), run `cf(L−1)` under `enable_grad` **on a side stream**,
  record done-event, `record_stream` the stashed outputs, restore RNG. Host
  cost = the push time of ~2k kernels, hidden behind recompute(L)'s GPU tail.
  `CheckpointFunction(L−1).backward` later waits the done-event and skips its
  own recompute.
- **(a) RNG:** dropout is 0.0 in the GLM bridge (`glm45_bridge.py:84` et al.)
  → states are saved/restored but never consumed; the fork/restore is
  host-atomic on the single autograd thread; main thread is parked inside
  `backward()`. Safe; assert `hidden_dropout==0 and attention_dropout==0` at
  gate-on. CUDA-graph: pass through un-lookahead'd during capture (mirror
  `random.py:642-649`).
- **(b) Memory:** +1 layer of live recompute activations ≈ 2–4 GiB at 131k
  (8,192 tok/rank/mb) — fits the 71 GiB headroom. At 16k×d32 (32,768
  tok/rank/mb) ≈ +8–12 GiB — **measure before enabling there**; gate off per
  shape if tight.

  > **CORRECTION (2026-08-10, minkowski's settled model; recorded by fermi —
  > supersedes the 2–4 GiB estimate, which undercounted the MoE save set
  > ~4×):** measured on the W3-v2 arm at 131k-d4 (poller + 16-rank allocator
  > snapshot, `round3/arm-w3-318g61w/mem_snapshot/`): **+25.4–25.8 GiB =
  > ~11.5 GiB intrinsic** (two ~11.4 GiB kicked chunk graphs live at the
  > mid-backward peak — MoE-pipeline saves dominate: 2.58 GiB×4 A2A/sort/GEMM
  > saves; the full-seq attention-K/V hypothesis is REFUTED, no 4 GiB blocks
  > exist) **+ ~14 GiB allocator retention** on the side-stream pool (26–30
  > GiB free-cached, uniform across ranks — recoverable via the v3
  > `BT_MOE_LOOKAHEAD_TRIM_EVERY` knob). 131k arm stood vs 44.9 GiB headroom
  > (≥10 GiB bar). 16k×d32 stays HARD OFF: corrected model there ≈ 13–14
  > GiB/chunk intrinsic (MoE/MLP terms ×4 at 32,768 tok/rank/mb) + retention
  > + fragmentation — genuinely borderline, re-measure before any enablement.
  > Under W3-v3 the number is RE-MEASURED (canary frame M1: ≤ +28 GiB vs the
  > selected baseline), not inherited. Full record: ESTATE_NOTES_minkowski.md
  > (2026-08-09 memory-model entry) + DESIGN_W3V3.md §5.
- **(c) CPU driving:** the autograd thread also blocks in the dispatcher's
  `d2h_event.synchronize()` during the kicked recompute (2× ~14 ms, runahead-
  inflated) — stalls bwd(L) pushes. **Synergy: enable FIX C
  (`BT_MOE_DISPATCH_REPLAY_CACHE`, hilbert — frozen into this tree 2026-08-09,
  `recompute.py` pass-marker frames + `token_dispatcher.py` replay-cache
  branches) with W3** — it removes replay-side eventSyncs; the pass-marker
  keying is compatible (kicked recompute is still a replay pass on the same
  carrier).
- **(d) Shared state audit:** dispatcher instances are per-layer
  (`moe_layer.py:306-312`) ✓; `cuda_dtoh_stream` is class-shared but carries
  KB-scale copies (serialize, fine) with per-instance events ✓;
  `SharedExpertMLP.stream` class-shared — serializes 0.3 ms kernels ✓, state
  machines are per-instance ✓; DSA `topk_holder`/layout cache live on the
  carrier keyed per (source-)layer (`dsa.py:1667-1690, 2072-2077`, layout cache
  hits on cu_seqlens object identity) — **verification V4: confirm the replay
  path re-assigns (not `copy_`-in-place) holder entries**, else L's DSA-bwd
  could read a tensor L−1's recompute is rewriting on another stream.
- **(e) `CUDA_DEVICE_MAX_CONNECTIONS`:** with =1, the two op sequences
  head-of-line block each other at every stream-wait — overlap degrades
  badly. **W3 wants the variable unset** (the CUDA default, 8 connections).
  Correction from fibonacci (2026-08-09): `run_trainer_node.sh:36` pinning =1
  is the **devbox** launch path — all historical bench numbers ran with =1,
  but **production pods leave it UNSET** (verified live in LPS-1003). So the
  experiment is not "=2" but **UNSET (prod parity)** — and the devbox benches
  under-report overlap wins relative to prod (W1's bwd de-serialization is
  already such a case: ~1.7 s/step visible only with CDMC unset).
  **V2: one canary run with CDMC unset, gate off vs on, loss + step time.**
  **V2b (prerequisite, per fibonacci): before ANY CDMC change, verify the
  LPS-1003 DSA stream-race fix (PR #875 + dsatopk5 wheel) is present in our
  0e0b65a6 pin — the =1 pin was the MASK for that race; unpinning without the
  fix re-exposes it. Evidence trail:
  `experiment_artefacts/glm/lps1003_loss_spikes/VERDICT.md` and
  `.../FIX_VALIDATION_100STEP.md`.** If unset is rejected, W3's capture ratio
  drops toward ~30–50 % (only work pushed before each wait overlaps) —
  re-evaluate then.
- NCCL thread-safety is a non-issue in this single-thread design (all
  collectives are issued from the autograd thread; the PG syncs with the
  *current* stream at call time — set the side stream as current around the
  kick).

---

## 8. Ranking and stacking (vs fibonacci's prior)

Prior: (iii) > (i) > (ii), W1 first. **Revised: W1 → W2 → W3.**

- W1 first: biggest measured win per unit risk (5.1 s, ~50 LoC, no autograd
  surgery). Uncontested.
- (i) is unbuildable (§2.2) — the ranking "(i) > (ii)" is moot; (ii) *is* the
  intra-layer design, and the list-A2A-view implementation makes it copy-free
  and bitwise-safe. W2 before W3 because it is self-contained (one dispatcher
  + one experts wrapper), testable bitwise in isolation, and its fwd-phase win
  (~1.5–3 s) survives W3.
- W3 has the largest ceiling (−8…−12 s) but the most surface: new checkpoint
  Function, registry, side stream, MAX_CONNECTIONS change, DSA-holder audit,
  FIX-C synergy. It also *subsumes* most of W2's bwd-phase value — do W2 v1
  first, W3 second, W2 v2 (bwd seq-bumps) only if W3 slips.
- Stacked model (base, no de-skew credit): 46.7 − 5.1 (W1) − 1.5 (W2 fwd)
  − 9±2 (W3) ≈ **31–34 s** (~1000–1100 tok/s/GPU at 524K tok/step).
  W1+W2-only fallback: **~41.5 s**.

---

## 9. Patch plan (files / lines, vendored mcore)

**W1 — `BT_MOE_PROBS_A2A_COMM=1`** (default OFF → ship-list after soak):
- `token_dispatcher.py:382-477` (`__init__`): under gate,
  `self._probs_ep_group = torch.distributed.new_group(get_process_group_ranks(self.ep_group))`.
- `token_dispatcher.py:668-712` (`token_dispatch`): under gate — issue tokens
  A2A async, issue probs A2A async on `_probs_ep_group`, shared-fc1, wait both
  (needs the issue/wait split below).
- `tensor_parallel/mappings.py:424-484`: add `_AllToAllIssue`/`_AllToAllWait`
  (or a `defer_wait` flag on `_AllToAll`); bwd unchanged semantics (rides
  `ctx.group`).

**W2 — `BT_MOE_A2A_PIPELINE=K`** (0=off; K|16; v1 scope):
- `token_dispatcher.py:422-432`: precompute per-group sort/restore index
  permutations (L experts).
- `:488-609` (`preprocess`): retain the two count matrices for D2H.
- `:922-961` (`_maybe_dtoh_and_synchronize`): add both matrices to the single
  D2H batch (same event).
- `:668-712` / `:714-780` / `:782-821` / `:823-860`: under gate, per-group
  list-A2A issue/wait Functions + per-group sort/unsort with device split
  slices; combine writes into `combine_buf` views at today's offsets.
- `experts.py:173-819`: `TEGroupedMLP.build_chunks(K)` (weight-sharing shells,
  `_make_fused_ops` pattern `:427-437`) + `forward_chunk(g, …)` reusing
  `bias_act_func`; `backward_dw` fans out (`:861-895`).
- `moe_layer.py:528-555` (`routed_experts_compute`): under gate, loop chunks.
- Guards: K | num_local_experts; `not drop_and_pad`; no `moe_preprocess`
  cudagraph (fallback + WARNING).
- v2 (optional): seq-bumps + wait-aware `Unsort`/`Slice` wrappers
  (`shared_experts.py:585-593` mechanism).

**W3 — `BT_MOE_LOOKAHEAD_RECOMPUTE=1`**:
- New `megatron/core/lookahead_checkpoint.py`: Function + carrier registry +
  side stream + kick logic (§7).
- `recompute.py:217-238` (`chunk_runner`): under gate, use it instead of
  `te_checkpoint`/`tensor_parallel.checkpoint`; register layer handles.
- Run script: `CUDA_DEVICE_MAX_CONNECTIONS` **unset** experiment (V2, prod
  parity; V2b prerequisite — see §12).
- Recommend pairing with `BT_MOE_DISPATCH_REPLAY_CACHE=1` (FIX C, hilbert,
  frozen 2026-08-09).

**Telemetry (all gates — v1 lesson: gates that can't fire pass parity
silently):** WARNING-level `armed/disabled` line at init per gate (with K,
comm device, EP size); per-step counters logged at interval: issues per path
(`a2a_pipeline_issues{group,direction}`), waits, fallback hits, zero-count
peers seen; canary mode adds CUDA-event-measured exposed-comm per layer-pass
to validate against the model. A gate that armed but never issued a chunked
call must be loud.

---

## 10. Test plan

- **T1 (CPU, bitwise decomposition):** random routing_map [T,256], EP=16
  simulated on host: compute per-group splits/offsets from the count matrices;
  assert concatenated per-group send views == exact row-permutation of
  today's buffer; assert combine_buf reassembly == today's combine layout
  byte-exact; assert per-group `tokens_per_expert` slices == today's slices.
  Pure index math — catches decomposition bugs before any GPU.
- **T2 (on-box, V1 — the numerics gate):** one MoE layer (16 local experts,
  FP8 block scaling, frozen weights), gate off vs K=2: `torch.equal` on layer
  output **and** on input grads, fwd and bwd, several seeds incl. an
  engineered imbalance case and a zero-rows-for-(peer,group) case (exercises
  0-count list-A2A entries — also validates NCCL behavior).
- **T3 (on-box canary):** 20 steps at 16k×d8, gates off vs W1, then W1+W2:
  loss must be **bitwise identical** step-by-step (stronger than the ≤2e-3
  canary — the design claims exactness); step-time delta logged vs model.
- **T4 (W3):** V2 (MAX_CONNECTIONS=2 soak) → V4 (DSA holder audit) → 20-step
  canary as T3 + memory high-water check at both shapes (16k×d32 gating
  decision) + 200-step stability.
- **T5 (trace acceptance):** extend `dispatcher_opt/check_acceptance.py`:
  probs-A2A off stream 83 (W1), SendRecv calls per phase ×2 with ~half bytes
  (W2), a2a⇄compute overlap % ≥ target (W2: >15 %, W3: >50 % of comm),
  step-time bounds per config.

---

## 11. Risk register

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| R1 | GEMM kernel schedule depends on `num_gemms` (16→8) → not bitwise | W2 blocked | V1 before any perf run; fallback = padded-16 call trick (§5) |
| R2 | 0-count entries in list-A2A misbehave on this NCCL | wrong results/ hang | T2 explicit case; keep count-0 entries (list positions must align across ranks) |
| R3 | MAX_CONNECTIONS=1 head-of-line: a mis-ordered wait silently kills overlap (or worse) | perf-only, or deadlock if cyclic | push-order rule §3.5; trace acceptance T5; waits only after all issues |
| R4 | seq-bump (v2/W3-adjacent) fragility across torch versions | perf-only | guarded by `grad_fn is not None`; tie-tolerant; telemetry counters prove arming |
| R5 | W3: DSA topk holder written in-place during replay | race, wrong attention | V4 audit (`dsa.py:1667-1690, 2072-2077`); if in-place, deep-copy on kick |
| R6 | W3: MAX_CONNECTIONS=2 rejected (TE/NCCL regression) | W3 capture ↓ | V2 soak first; W3 ships behind both gates |
| R7 | W3 memory at 16k×d32 (+8–12 GiB) | OOM | measure T4; per-shape gate |
| R8 | CUDA-graph capture of moe_preprocess | wrong capture | v1 fallback to unchunked path + assert; cudagraph support = follow-up |
| R9 | Extra NCCL comm (W1) init cost/compat | init time, ~MBs | one-time at build; standard `new_group` |
| R10 | Host list-building overhead per layer-pass (K×16 views) | µs-scale CPU | **quantified mitigation (binding for the W2 patch): precompute the per-(g,peer) offset TEMPLATE at init; per-pass only fill counts; target <100 µs/pass added host time**; measure in canary (launch budget is tight post-B+F — see FIX-B report) |

## 12. Open verification items (for the box)

- **V1** grouped-GEMM bitwise vs `num_gemms` (T2). — blocks W2.
- **V2** `CUDA_DEVICE_MAX_CONNECTIONS` **unset** (prod parity) correctness/perf
  soak — **with V2b as prerequisite** — blocks W3.
- **V2b** ~~verify the LPS-1003 DSA stream-race fix in our pins~~ **PASS**
  (helmholtz, 2026-08-09): PR #875 (`91934042`, "pin DSA indexer fwd
  prefill+launch to a dedicated stream") is in the trainers pin (HEAD
  ef4ea4a8). The `+dsatopk5` patch-wheel was then RETIRED by PR #910
  (`8bac180f`), which vendors upstream cudnn-frontend develop @ 74785165
  (`nvidia_cudnn_frontend-1.27.0.dev20260803+git7478516-*.whl` in
  `server/uv.lock`) — upstream PR #354 carries the LPS-1003 root fix in
  broader form (`torch_stream_context` → `get_stream_from_external()`, no
  `ExternalStream(0)` anywhere, covering every DSA call site, not just
  indexer_fwd). The regression guard
  (`server/tests/unit/dp_worker/test_cudnn_dsa_indexer_launch_stream.py`)
  asserts exactly this arrangement against the installed wheel and hard-fails
  (not skips) on linux/x86_64 cp312 CI; PR #910's devbox validation re-ran the
  2.25 GiB cold-start race repro clean (0/12) with a 100-rep soak matching
  the +dsatopk5 control. Evidence:
  `experiment_artefacts/glm/lps1003_loss_spikes/{VERDICT.md,PR875_REVIEW_0803.md,FIX_VALIDATION_100STEP.md}`
  + the two commit messages. **The =1 pin's mask target is fixed in our pins;
  CDMC-unset is unblocked on this account.**
- **V3** confirm `torch.distributed.all_to_all` (list form) + 0-count entries
  on the production NCCL version (folded into T2).
- **V4** ~~DSA index-share holder replay-write audit~~ **PASS** (helmholtz,
  2026-08-09): all carrier-held DSA state uses dict **re-assignment**, never
  in-place writes. (i) top-k holders: `topk_holder[self.layer_number] =
  topk_indices` / `topk_length_holder[...] = ...` (`dsa.py:2345-2347`) rebind
  to freshly-computed tensors each pass; reads (`dsa.py:2103`) happen only
  during forward/replay — backward consumes ctx-saved tensors, never the
  holder. (ii) FIX-B layout cache (`dsa.py:83-99`): entries hit on
  `(key, cu_seqlens identity)` and replay always hits (same carrier objects
  in both passes); the miss path rebuilds + rebinds (`cache[key] = ...`,
  still re-assignment) with deterministic-identical content. Keys are
  per-microbatch layout kinds shared across layers (`dsa.py:1803, 1845,
  1926`) — safe because replays only read. (iii) Skip-layer source reads
  resolve to first-pass tensors in both the normal and the lookahead order
  (a source layer's replay always runs after its consumers' replays).
  **W3's lookahead is safe on this surface** given the single-autograd-thread
  kick (all carrier dict access stays host-serialized; GPU-side, recompute
  writes freshly-allocated tensors while bwd reads its own graph's saves).
- **V5** ~~measure MoE GEMM+permute slice from the trace~~ **SATISFIED**
  (fibonacci, 2026-08-09): compute-in-gap measured directly on gated-v2 — 897
  dispatch→combine gaps contain 4.393 s total compute (avg **4.9 ms**, the
  GEMM+sort window); 896 combine→dispatch gaps contain 11.93 s (**13.3 ms**
  attention window). K=2 confirmed; no further trace work needed for the K
  choice.
