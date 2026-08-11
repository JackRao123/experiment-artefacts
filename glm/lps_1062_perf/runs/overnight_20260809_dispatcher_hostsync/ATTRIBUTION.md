# ATTRIBUTION: exp05d host-sync trace analysis (laplace, 2026-08-09)

**Trace:** `~/perf_profiles/lps-1062/opt-night/exp05d.pt.trace.json` (1.05 GB kineto, rank 0,
one 48.98s step = 2 microbatches × 262,144 tokens, 78 layers × 2 mb = 156
`CheckpointFunction`/`CheckpointFunctionBackward` pairs, full recompute, EP16/CP16, ship env).
**Method:** perfetto `trace_processor` (python API); CPU-op attribution by `parent_id` ancestor
walks; serialization measured as GPU-union idle (all kernel/memcpy/memset streams) inside
merged host-block windows. Code refs: vendored megatron-bridge @ ef4ea4a8
(`server/vendor/megatron-bridge/3rdparty/Megatron-LM/...`).

## TL;DR — the mission premise needs a correction

1. **The dominant `aten::nonzero` site is NOT the MoE dispatcher.** 93% of the 20.6s nonzero
   CPU time (156 calls, 19.2s) is **one call per layer-backward in the DSA sparse-attention
   backward**: `torch.nonzero(topk_length > 0)` at
   `experimental_attention_variant/dsa_cudnn_kernels.py:2130`. The same 156 calls account for
   19.18s of the 20.0s `cudaStreamSynchronize`.
2. **The 15.0s `cudaMemcpyAsync` CPU time is ~99% blocking pageable-DtoH `tolist()`s in
   `sort_chunks_by_idxs`** (`moe/moe_utils.py:569-573`, via `token_dispatcher.py:768`
   dispatch_postprocess and `:803` combine_preprocess) — this *is* dispatcher split
   bookkeeping, ~2–3 calls/layer-pass, ~30ms each in fwd.
3. **Almost none of it serializes the GPU.** Merged >1ms host-block windows cover 34.3s
   (70%) of the 48.98s wall, but GPU-union idle inside those windows is **0.70s**. During the
   156 big backward-nonzero blocks (19.2s), the GPU runs 14.93s of SendRecv + 4.30s of
   compute — idle 0.41s. Total GPU idle in the step is 4.74s, and the >1ms gaps (0.99s)
   cluster at step-start ramp and the end-of-step optimizer/grad-sync tail, not at syncs.
4. **Realistic step-time win bound if all these host syncs vanish: ~1–2s/step (2–4%)**;
   hard ceiling = total GPU idle 4.74s, and most of that is unrelated (ramp/tail).
   The Aug-7 REPORT.md line "nonzero chain ≈ the other half [of the step], co-dominant" is a
   misread of CPU-block time as critical-path time. The step is comm-bound (SendRecv 24.5s
   GPU time); the host syncs ride in comm slack. **The surgery is still worth doing** — the
   syncs are latent critical path: they become exposed ~1:1 as soon as comm improves, and the
   fixes are cheap (see §5).

## 1. `aten::nonzero` — 26,684 calls, 20.59s CPU

| site | calls | CPU | share |
|---|---:|---:|---:|
| **DSA sparse-attn bwd** — `dsa_cudnn_kernels.py:2130`, inside `FusedSparseAttentionFuncBackward` | **156** | **19.20s** | **93%** |
| DSA CP layout boolean indexing — `dsa_layout.py:159-161,187-188` (via `aten::index`), fwd | 13,260 | 0.68s | 3.3% |
| same, bwd (full-recompute replay) | 13,260 | 0.66s | 3.2% |
| `aten::_index_put_impl_` (step-level + other) | 4 | 0.05s | 0.2% |

- Big-bwd-call duration: p50 120ms, p90 154ms, max 238ms. Runs on the **autograd backward
  thread**; the CPU time is blocked-wait for the compute stream to drain to the nonzero.
- Small-call structure: exactly **85 per layer per pass** = 17 × 5.
  `build_packed_allgather_cp_query_positions_and_key_reorder` (dsa.py:1736–1814) calls
  `build_packed_allgather_cp_local_positions` 1 (query) + cp_size=16 (per-rank key reorder)
  times; each call does 5 boolean-mask indexes (`seq_starts/seq_ends/seq_lens[nonzero]` at
  dsa_layout.py:159-161, `segment_starts/segment_lens[nonempty_segments]` at :187-188).
  Boolean-mask indexing on CUDA dispatches `aten::nonzero` internally → 2 pinned DtoH copies
  + a stream sync each. Individually cheap (~50µs) because they run in the fwd launch phase
  with a shallow queue. 85 × 156 × 2 (fwd + recompute replay) = 26,520 ✓.
- **Dispatcher's own `preprocess`/`_maybe_dtoh_and_synchronize` contributes ~zero nonzero
  calls** — the split math is sum/reshape/gather, no nonzero. The dispatcher's sync cost
  shows up as memcpy/event-sync instead (§2, §3).

## 2. `cudaStreamSynchronize` — 29,131 calls, 19.98s CPU

| site | calls | CPU |
|---|---:|---:|
| inside the 156 DSA-bwd `nonzero` | 156 | **19.18s (96%)** |
| inside the 26.5k layout boolean-index nonzeros (fwd+bwd) | 26,520 | 0.42s |
| `aten::item`/`_local_scalar_dense` (fwd 792 + bwd 792 ≈ 5/layer-pass) | 1,584 | 0.02s |
| step-level `aten::item` (metrics/packing, cf. `packing.py:415`) | 34 | 0.26s |
| misc `copy_`/`_index_put_impl_`/other | ~37 | 0.10s |

## 3. `cudaMemcpyAsync` — 33,630 calls, 15.01s CPU

| site | calls | CPU |
|---|---:|---:|
| **`aten::copy_` blocking pageable-DtoH `tolist()`s — `sort_chunks_by_idxs` (moe_utils.py:569-573)** fwd | 1,896 (308 >1ms) | **9.20s** |
| same, bwd (recompute replay, on autograd thread) | 1,896 (~280 >1ms) | **5.62s** |
| 26.5k nonzero internals (pinned DtoH count+indices, ~4µs each) | ~29.5k GPU-side | ~0.10s |
| `aten::item` DtoH | 1,584 | 0.008s |
| step-level + other | ~140 | 0.08s |

- The blocking ones are ~2–3 per layer-pass, avg ~30ms (fwd) / ~20ms (bwd): the
  `split_sizes.tolist()` / `sorted_idxs.tolist()` calls in `sort_chunks_by_idxs`
  (`torch.split` needs host sizes). `sorted_idxs` (`sort_input_by_local_experts` /
  `restore_output_by_local_experts`) is a GPU argsort result that is **not** included in the
  dispatcher's `_maybe_dtoh_and_synchronize` batch list — so its `tolist()` is a full
  pipeline-drain sync each time. Call sites: `token_dispatcher.py:768` (permutation-2 in
  dispatch_postprocess) and `:803` (combine_preprocess).
- The dispatcher's purpose-built DtoH machinery is already relatively cheap:
  `cudaEventSynchronize` (the `d2h_event.synchronize()` in `_maybe_dtoh_and_synchronize`,
  token_dispatcher.py:959) = 150 calls, 2.11s CPU (~1/layer fwd, ~14ms each), and the
  side-stream pinned DtoH copies are µs-scale CPU each.
- GPU-side, all 33.6k memcpys total 0.19s of transfer time — this is pure host-sync
  overhead, not bandwidth.

## 4. Serialization vs overlap (the win bound)

- Step wall 48.98s; GPU-union busy 44.24s; **idle 4.74s (9.7%)**.
  Idle decomposition: 371k sub-ms inter-kernel bubbles (~3.5s, normal launch granularity);
  >1ms gaps 0.99s, dominated by step-start ramp (t≈0.2–0.3s) and end-of-step tail
  (t≈48.6–48.8s: optimizer / final grad sync) — not host-sync related.
- Merged host-block windows (nonzero>5ms + memcpy>1ms + streamSync>1ms): 34.30s coverage
  (70% of wall). **GPU idle inside them: 0.70s.**
  - Inside the 156 DSA-bwd nonzero windows (19.2s): SendRecv 14.93s, compute 4.30s,
    memcpy 0.04s → GPU ~fully occupied; idle 0.41s.
  - Inside the >1ms tolist-copy windows (14.8s): idle 0.28s.
- SendRecv (comm) stream: 24.48s busy / 24.50s idle over the step. 10.05s of that idle
  falls inside host-block windows — i.e. the comm stream had slack while the CPU was
  blocked; the blocks are not gating comm issue.
- fwd span 31.77s / bwd span 40.37s (interleaved: 78 fwd mb1, then bwd-mb1 ∥ fwd-mb2,
  then bwd mb2). GPU idle in fwd span 2.83s, bwd span 3.13s — mostly compute waiting on
  a2a results, not on the host.

**Bound:** direct measured starvation = 0.70s. Allowing second-order enqueue-delay effects
that never show as full GPU idle, the realistic win is **~1–2s/step (2–4% of 48.8s;
≈ +2–4% tok/s)**. The absolute ceiling (if every GPU-idle second were somehow caused by
syncs) is 4.74s — not credible given the ramp/tail clustering. **It is not a 20–35s lever.**
Note the asymmetry for planning: this bound *shrinks* as comm improves inversely — every
second removed from SendRecv exposes ~1s of these host blocks on the critical path, so the
surgery is a precondition for realizing future comm/DeepEP wins, not an alternative to them.

## 5. Where the surgery should go (priority by unblocked exposure)

1. **`dsa_cudnn_kernels.py:2130`** — the 156× 120ms backward nonzero. `topk_length` is
   computed in fwd and saved; `valid_row_indices` can be computed on-device without a host
   round-trip (compact via cumsum/index_select, pass `all_rows_nonempty` decisions through
   the fwd-saved mask, or cache per-layer indices from the fwd pass — routing/topk is
   identical in the recompute replay). Single biggest block; removes 19.2s of autograd-thread
   blocking that is one comm-improvement away from being critical path.
2. ~~**`moe_utils.py:569-573` tolists**~~ **SUPERSEDED — see errata above.** The 14.8s
   blocking-memcpy class is THD RoPE cu_seqlens host reads (`_apply_rotary_pos_emb_thd`,
   rope_utils.py), not sort_chunks (TE fused permute is active; the tolist lines never
   execute). Fix = ramanujan's 0003 per-microbatch host-copy cache
   (`BT_THD_ROPE_HOST_CACHE`).
3. **`dsa_layout.py` layout builders** — 26.5k small syncs. Inputs (`cu_seqlens_q/kv`,
   cp_size/rank) are identical for every layer of a microbatch and for its recompute replay:
   a per-microbatch cache keyed on the cu_seqlens tensor version eliminates ~26.4k of 26.5k
   calls; the surviving per-call cost is ~50µs. Alternatively build positions with
   device-side ops only (`repeat_interleave` with `output_size` is already used — the boolean
   filtering of empty segments can be done arithmetically).
4. Leave `_maybe_dtoh_and_synchronize` as-is (2.1s event-sync, already batched/side-stream).

Parity note for ramanujan: (1) and (3) change only host/device placement of index math —
bitwise-identical outputs are checkable offline on random routing inputs; (2) same for split
tensors. All three are env-gateable independently.

## ADDENDUM: corrected win accounting + mechanism verdict (2026-08-09, laplace)

Measured on the steady same-shape pair (unpatched-4mb-steady step3 vs
gated-v2-4mb-steady step8, both 4×131,072 = 524,288 tok/step, EP16/CP16;
profiled walls 53.95s → 46.68s; timed steady 634 → ~720 tok/s/GPU, **+11–14%**):

| metric | unpatched 4mb | gated v2 (A+B+F) | verdict |
|---|---:|---:|---|
| SendRecv total / calls | 25.77s / 2,700 | 26.03s / 2,700 | **flat** |
| SendRecv per-call p50 / p90 | 9.19 / 17.96ms | 9.35 / 17.24ms | **flat** |
| GPU-union idle | 8.52s | **1.82s** | **−6.7s ≈ the wall win** |
| SendRecv inter-call gaps | 27.09s (p90 41ms) | 19.59s (p90 30ms) | compressed |
| comm-stream idle | 28.19s (8.36 inside blocks) | 20.65s (0.45 inside) | decompressed |
| host-block coverage (>1ms) | 32.17s = 60% of wall | 0.64s = 1% | gone |
| dispatcher eventSync CPU | 2.22s (600 calls) | 33.56s (600 calls) | inflated by runahead |
| FIX A flag eventSync CPU | — | 0.46s (312 calls, p50 5.5µs, 12 >1ms) | µs-scale as designed |

**1. Mechanism verdict: launch-pipeline decompression, NOT comm de-staggering.**
SendRecv time is flat in total and per-call — the all-to-all did not get faster;
the pre-registered de-staggering signature (shrinking SendRecv totals / falling
per-call p50/p90) is absent. The win is entirely GPU-feeding: GPU-union idle fell
6.7s against a 7.3s profiled wall win. At 4mb the host blocks (RoPE tolist
copies 13.9s CPU + 53k layout-builder nonzeros + ~300k extra kernel launches)
starved the *launch pipeline*: 2× the kernels of 2mb with half the per-pass
queue depth, so each host block let the GPU drain to empty. Removing them lets
the host run ~1 layer ahead; idle collapses.

**2. My original win bound (0.7–2s) was shape-specific and did not transfer.**
It was derived at 2mb/256K where the blocks were ~fully overlapped (0.70s idle
inside block windows). The count rows scale exactly with M (validated), and the
host-block CPU *totals* are ~M-invariant (nonzero 20.6→21.4s, memcpy 15.0→13.9s
— per-block wait ∝ per-pass work ∝ 1/M), but the GPU-*starvation* share of
those blocks is shape-dependent and was much larger at 4mb (8.52s idle vs
4.74s). I flagged count transferability but never re-derived the idle/win rows
for 4mb; the +14% exceeded my +1–4% expectation for exactly that reason.

**3. The dispatcher's own eventSync is now the dominant host block — inflated
by the fix.** With the launch pipeline decompressed, the host runs ahead, so
every "wait for the GPU to reach point X" sync lengthens in wall terms:
the dispatcher's `d2h_event.synchronize()` went 2.22s → 33.56s (fwd 9.9s p50
32ms + replay 23.7s p50 77ms). It is benign today (GPU idle 1.82s — the waits
overlap GPU work; the a2a they gate couldn't run earlier anyway), but it is the
next host-sync target: FIX C (replay-metadata reuse) removes the 300 replay
ones (23.7s CPU). Note the feedback loop: each decompression makes the
remaining syncs look bigger in CPU time while helping the GPU.

**4. FIX A worked mechanically and is µs-scale — and buys ~nothing today.**
A-active proof: 312 flag events (78×4, replay-pass kicks), p50 5.5µs, GPU idle
inside their windows 0.012s; the 312 drains are gone (residual >5ms nonzeros:
8, all step-level). Why it buys ~nothing: the dispatcher's replay eventSync
(p50 77ms) fires *upstream* of A's read point in every layer-backward and
throttles the autograd thread back to GPU-lockstep, so both A's event and the
original nonzero were/are cheap at that point regardless. A's exposure is
therefore a *sequela of fixing the dispatcher sync* (FIX C/D), not of comm
wins per se — pascal's "as exposed as the drain" needs that refinement.
Recommendation supported by the traces: **ship B+F, park A default-off, revisit
with FIX C.** (A's −2% marginal under ABF vs BF-only is being pinned by gibbs's
BF-only timed cycle; the trace shows no starvation from A, so the delta, if
real, is kick-overhead/second-order, not the event waits.)

**5. Scaling laws recorded for future derivations** (validated on the
unpatched-4mb capture): op *counts* scale exactly with microbatch count M;
host-block *wait-time totals* are ~M-invariant (per-block wait ∝ queue depth ∝
per-pass work ∝ 1/M); >1ms-threshold counts fall with shape because per-call
waits shorten; launch-bubble GPU idle scales with kernel-launch count, not
wall. Also: window-1 captures are timing-garbage (the A-inert "BF" capture
showed 19.9s dispatcher eventSync vs 2.22s steady) — steady windows only.

Checker state: `baseline-4mb131k` re-seated to the measured unpatched values;
`post-patch-ABF-4mb131k` eventsync rows split into dispatcher (informational,
runahead-inflated) vs A-flag (sharp: 312 calls, ≤1.5s, ≤25 >1ms); gated-v2
capture = ALL PASS.

## Errata (2026-08-09, laplace)

- §3 reported `cudaEventSynchronize` as 150 calls / 2.11s — main thread only. Full count
  is **300 / 4.13s** (150 fwd on main + 150 recompute-replay on the autograd thread);
  ramanujan's figure in CANDIDATE_FIXES is correct.
- **§3 site attribution was WRONG (adjudicated 2026-08-09, ramanujan wins):** the ~590
  blocking pageable-DtoH copies (14.8s CPU) are NOT `sort_chunks_by_idxs` tolists.
  `moe_permute_fusion=True` in the ship config, so the TE fused path runs
  (`te_moe::chunk_sort_fwd` ×600 in the trace) and moe_utils.py:569-573 never executes.
  The real site is **THD RoPE bookkeeping**: `_apply_rotary_pos_emb_thd`
  (`megatron/core/models/common/embeddings/rope_utils.py`, q/k call sites at
  absorbed_mla.py:626/636) — `((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()`
  + `cu_seqlens[-1].item()` + per-seq `cu_seqlens[i].item()`, 2 calls/layer-pass.
  Decisive trace evidence: 619 `aten::to` slices >1ms (14.82s) are direct children of
  CheckpointFunction{,Backward} at 2/layer-mb/pass, and the op motif around every one is
  `slice → sub → floor_divide → blocking to/_to_copy/copy_/memcpyAsync+streamSync →
  split_with_sizes → item×2` (cu_seqlens diff → //cp_size → tolist → torch.split →
  .item()s), followed by cat×3/cos/sin (rotary). My §3 caveat flagged line-level
  uncertainty but the *site* itself was wrong — I read the `fused=` early-return in
  sort_chunks_by_idxs and failed to check whether the ship config sets
  moe_permute_fusion. The op-level counters (587 blocking `cudaMemcpyAsync` >1ms /
  14.74s; 33,630 / 15.01s total) are unaffected — only the owning call site changes.
  Reconciliation with ramanujan's "616": he counted `aten::to` slices >1ms under the
  checkpoints (612 by my count; 619 including 7 step/CE-level) — a different op class
  from the checker's 587 `cudaMemcpyAsync` >1ms (some `aten::to` blocks have their
  memcpy child just under 1ms or block partly in the tolist python loop). Both measure
  the same physical class; the checker's cudaMemcpyAsync row is self-consistent
  (baseline 587 → post-F ≤8, allowing ~2/step cache-miss D2Hs).

## Caveats

- Rank 0 of 16, one step; per-rank skew not measured (SendRecv imbalance exists but doesn't
  change the host-sync attribution).
- Trace has no python stacks (HTTP-driven slimmed capture); call-site identification is by
  op-motif + call-count matching against the ef4ea4a8 source. The two load-bearing matches
  are exact on counts: 85 = 17×5 per layer (layout builders) and 156 = 78×2 (DSA-bwd
  nonzero). The tolist attribution (§3) is ±1 call/layer certain on exact line (569 vs 570/572).
- CPU-time numbers are wall durations of the CPU op slices (blocked-wait included), matching
  the Aug-7 report's 20.6s/20.0s/15.0s figures exactly (26,684 / 29,131 / 33,630 calls).
