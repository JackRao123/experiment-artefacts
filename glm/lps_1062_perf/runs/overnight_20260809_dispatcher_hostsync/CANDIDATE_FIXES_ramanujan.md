# Candidate fixes — LPS-1062 dispatcher host-sync project (ramanujan, 2026-08-09, PHASE 1)

Status: code read + trace cross-check done. **Design NOT committed** — gated on
laplace's attribution (this note is a cross-check of that, not a substitute).

## 0. Headline: the mission's target is mis-attributed

The brief targets "the mcore alltoall dispatcher host-side split bookkeeping —
26,684 aten::nonzero + ~29k cudaStreamSynchronize + 33.6k cudaMemcpyAsync".
A targeted query of `~/perf_profiles/lps-1062/opt-night/exp05d.pt.trace.json`
(rank 0, one 48.8s step) shows the nonzero class is **not** in the dispatcher:

| site | calls | CPU time | where |
|---|---:|---:|---|
| `torch.nonzero(topk_length > 0)` — DSA sparse-attn **backward** compaction, `dsa_cudnn_kernels.py:2130` (under `FusedSparseAttentionFuncBackward`) | **156** (78 layers × 2 mb) | **19.2 s** (93% of all nonzero CPU; ~123 ms each = full pipeline drain per layer-mb in bwd) | DSA |
| boolean-mask index (`aten::index` → internal nonzero) — CP packed-layout builder `dsa_layout.py:159-161,188-189` via `build_packed_allgather_cp_query_positions_and_key_reorder` (17 internal calls × 5 mask-index = **85/layer-mb/pass**), called per layer at `dsa.py:1761` and replayed by full recompute | **26,520** (13,260 fwd `CheckpointFunction` + 13,260 replay `CheckpointFunctionBackward`) | 1.34 s (~50 µs each) + launch storm | DSA CP layout |
| `aten::_index_put_impl_` | 4 | 46 ms | noise |
| dispatcher `d2h_event.synchronize()` (`token_dispatcher.py:959`) — the *actual* dispatcher host sync | 300 `cudaEventSynchronize` (150 fwd + 150 replay; 75 MoE layers × 2 mb) | **4.13 s** (~14 ms avg) | MoE dispatcher |
| `aten::item`/`_local_scalar_dense` | 1,618 | 285 ms | misc |

Sanity: 1,343 ms + 19,204 ms + 46 ms = 20,594 ms = the trace's aten::nonzero
total exactly. cudaStreamSynchronize 29,131/20.0 s ≈ the 26,684 nonzero-internal
syncs (19.65 s) + 1,618 `aten::item` (0.29 s). The 26,684 count conflates a
huge-count/cheap class (mask-index) with a tiny-count/catastrophic class (DSA
bwd). The Aug-7 "MoE routing/dispatch" attribution was wrong about both.

Also verified: Mac checkout (ef4ea4a8) vendored mcore is **line-identical** to
the on-box 0e0b65a6 pin for `token_dispatcher.py`/`moe_utils.py` (mcore
d3932e757c vs 57efae08b — diff empty), so the forensic line refs
(preprocess:558 all_gather, :959 event sync) map exactly onto the Mac files.

Note: in mcore 0.19.0 the dispatcher has **no per-expert nonzero loops** —
permute is argsort-based or TE-fused (`moe_permute_fusion=True` in ship config);
split bookkeeping is already device-compute + side-stream async D2H + deferred
event sync. Mission candidate (a) as stated does not exist in this version.

## 1. Candidate fixes (priority order by measured cost)

### FIX A — DSA backward `nonzero(topk_length > 0)` → precomputed async emptiness flag (THE elephant, ~19.2 s CPU)

Site: `_run_sparse_attention_backward` (`dsa_cudnn_kernels.py:2130`), reached
from `FusedSparseAttentionFunc.backward` (the trainer's path:
`run_fused_absorbed_sparse_attention`; trace-confirmed). Once per layer-mb
(156/step) it does a syncing `torch.nonzero` to compact away rows with zero
valid top-k entries before the cuDNN bwd call.

Design (env-gated `BT_FUSED_DISPATCH_SPLITS=1` or a dedicated gate — see open
questions):
- In `FusedSparseAttentionFunc.forward`, right after `topk_length_flat` is
  produced, kick `(topk_length > 0).all()` as an **async D2H** (1 byte, pinned
  buffer, side stream + event — mirror the dispatcher's own
  `_maybe_dtoh_and_synchronize` pattern). Save `(event, host_flag)` on `ctx`.
- In `backward`: `event.synchronize()` (µs — copy completed during fwd) →
  - **all rows nonempty** (dominant case: empty rows ⟺ THD padding rows; the
    uniform benches have none): do **not** call the fast-path branch
    (`all_rows_nonempty=True` changes the cuDNN batch shape N→N+1 → possible
    ULP-level dkv reduction-order diff). Instead keep the *existing* compaction
    dataflow but substitute `valid_row_indices = torch.arange(N)` — **bitwise
    identical**, since `nonzero(topk_length>0)` *is* `arange(N)` exactly when
    all rows are nonempty. Same cat/index_select/wrapper calls, zero sync.
  - **some rows empty** (padded batches, real customer data): fall back to the
    original syncing `nonzero` path, unchanged.
- Fallbacks: no CUDA / missing event / None topk_length → original path.

Tradeoffs: +156 tiny async D2H/step in fwd (negligible); bwd event-sync hits an
already-complete event (~µs vs ~123 ms). No numerics change in either branch
(bitwise by construction). Does not require proving layer-independence of
topk_length or host knowledge of padding — the flag is computed from the same
device tensor the nonzero would read. Risk: low. Stream-safety review needed
(side-stream D2H vs. tensor lifetime — laplace's later task).

### FIX B — cache packed-CP layout (query positions + KV reorder) per microbatch (26,520 nonzero + ~150k kernel launches, ~1.3 s CPU)

Site: `dsa.py:1761-1774` (and `:1812-1824` fallback) call
`build_packed_allgather_cp_query_positions_and_key_reorder`, which internally
calls `build_packed_allgather_cp_local_positions` 17× (1 query + 16 CP ranks),
each doing 5 boolean-mask indexes (each = nonzero + D2H sync). Inputs are pure
functions of `(cu_seqlens_q, cu_seqlens_kv, cp_size, cp_rank, sizes)` — all
**per-microbatch constants**, identical across all 78 layers and across the
fwd/recompute replay.

Design: cache on the `packed_seq_params` carrier — the exact pattern the file
already uses for top-k sharing (`_dsa_index_share_topk_holder`,
`dsa.py:1541-1625`). Entry stores the input tensor refs + result; hit iff
stored `cu_seqlens_q is cu_seqlens_q` etc. (holding refs defeats `id()` reuse
after GC). First layer per mb computes (85 nonzeros), other 77 layers + all
replays hit. 26,520 → 170 nonzero/step. **Bitwise identical** (same integer
tensors reused). Must audit downstream for in-place mutation of the returned
tensors (`k.index_select(0, kv_reorder_idx)` is read-only ✓;
`packed_query_positions.contiguous()` returns self if already contiguous —
clone once at store time if any in-place consumer is found).

Tradeoffs: tiny memory (few int64 vectors per mb); carrier lifetime =
microbatch lifetime (already the pattern). Risk: very low. This *is* mission
candidate (c) "reuse metadata across the recompute replay" — the biggest
instance of it is in the DSA layout path, not the dispatcher.

### FIX C — dispatcher replay-metadata reuse (150 replay event-syncs ≈ 2.0 s + 150 replay all_gathers ≈ 1.6 s GPU)

The recompute replay re-runs `dispatch_preprocess` → `preprocess` (16-rank
all_gather over tp_ep, `token_dispatcher.py:558`) + D2H + event sync with
**bitwise-identical routing** (recompute is deterministic). Reusing the fwd
metadata in replay removes the replay's share of the 4.13 s event-sync cost and
the replay all_gathers (collective skip is consistent: all ranks replay the
same layers in the same order).

Keying problem: replay recomputes `routing_map` (new tensor object, same
values) — no cheap identity key; value key needs a sync (self-defeating).
Would need a "in recompute replay" flag threaded from mcore's
`core/recompute.py:custom_forward` + per-(layer, microbatch) anchoring.
Tradeoffs: real win but **medium risk** (relies on replay determinism — true by
design in mcore full recompute, but a silent divergence would corrupt
invisibly; needs the on-box parity test to assert replay-vs-fwd routing_map
bitwise equality first). Recommend doing A+B first, then C only if the A/B
shows the dispatcher residual matters.

### FIX D — batch the dispatcher's per-tensor D2H copies (~0.2–0.5 s, minor)

`_maybe_dtoh_and_synchronize` issues 5–6 separate `cudaMemcpyAsync`
(tokens_per_expert, input_splits, output_splits, output_splits_tp,
num_out_tokens) per MoE layer-mb-pass (~1,500/step; the code even has
`TODO: use MemcpyBatchAsync`). Pack into one staged pinned buffer → one copy.
Does **not** touch the 14 ms event-sync wait (bounded by all_gather
completion, not copy count). Fold into C if C happens; not worth a standalone
patch + A/B slot.

### FIX E — device-side split bookkeeping (mission candidate (a)/(b) as stated)

Mostly already present in 0.19.0 (side-stream D2H + deferred event). The
residual 4.13 s is the genuine host-splits dependency of NCCL alltoallv.
Full device-side splits = DeepEP/fused-EP territory, ruled infeasible on this
fabric on Aug-7. Not actionable here.

## 2. Win accounting (48.8 s step, exp05d trace)

| fix | CPU-sync removed | est. wall win |
|---|---:|---|
| A | 19.2 s (156 full bwd drains) | GPU-starvation-induced fraction — laplace to bound; plausibly several s |
| B | 1.3 s + 26.5k launch/sync round-trips | modest alone, compounds with A (fewer fwd/replay stalls) |
| C | ~2 s event sync + 1.6 s all_gather GPU | ~2–3 s, medium risk |
| D | ~0.2–0.5 s launch overhead | minor |

## 3. PHASE 2 plan (on pascal's go)

Env-gated patch (default OFF) + standalone parity test, Mac-CPU + box-GPU:
- A: structure the branch as a pure helper; CPU test — flag-true ⇒
  `arange == nonzero(...)` bitwise, flag-false ⇒ original path; box GPU test —
  dq/dkv patched vs unpatched on random q/kv/topk with and without empty rows.
- B: random cu_seqlens (incl. empty seqs + padding) — cached vs recomputed
  positions/reorder bitwise; runs on Mac CPU (builder is device-agnostic).
- Loss-canary on-box per HANDOFF rule (drift > 5e-3 = stop).

## 4. Open questions for pascal/laplace

1. Env-gate granularity: one `BT_FUSED_DISPATCH_SPLITS=1` for A+B, or separate
   gates (A is DSA-bwd, B is DSA-layout — neither is literally "dispatch
   splits")? Suggest `BT_DSA_BWD_ASYNC_NONEMPTY` + `BT_DSA_CP_LAYOUT_CACHE`,
   or one umbrella `BT_HOST_SYNC_OPT`. Jack's call via pascal.
2. The megatron-bridge AGENTS.md says "NEVER modify 3rdparty/Megatron-LM" —
   both fix sites live there (this is Baseten's mcore fork with custom DSA
   code; Aug-7 precedent patched vendored mcore on-box and archived the diff).
   Confirm the on-box patch + archive-to-dispatcher_opt/ flow stands.
3. laplace: my parent/grandparent attribution is from slice nesting only (no
   python stacks in this trace — slimmed HTTP-driven format). Suggest laplace
   focuses on the **wall-time upper bound** of the 19.2 s (GPU idle induced by
   the 156 drains) + stream-safety review of FIX A, rather than re-deriving
   the call-site map.
