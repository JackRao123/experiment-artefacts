# cuDNN DSA kernel audit (GLM-5.2 CP32 path) — 2026-07-28

Follow-up audit to the [dsa_topk_ima](../dsa_topk_ima) incident: after two
data-dependent IMA bugs were found in the DSA indexer top-k path
(cudnn-frontend#407 odd-`top_k` JIT assert; trainers#814 radix top-k OOB-lane
phantom candidates), we audited the **entire** GLM-5.2 production call chain
for more bugs of the same and adjacent classes.

**Scope.** `nvidia-cudnn-frontend 1.26.0` (35fd7b0, matches the pinned wheel):
`indexer_forward/` SM100, `indexer_top_k/` (all six files),
`sparse_attention_backward/` SM100, shared `utils/`; the vendored Megatron glue
(`dsa_cudnn_kernels.py`, `dsa_layout.py`, `dsa_masking.py`, `dsa.py`); trainer
packing (`packing.py`); FlashMLA `flash_mla_sparse_fwd` @ b7643bd boundary;
upstream issues/PRs through 2026-07-28. The two known bugs are **not**
re-reported.

**Method.** Hand-read of the top-k kernels and the backward interface; three
parallel sub-audits (SM100 forward kernel, SM100 backward kernel, Megatron
glue + packing); upstream issue/PR sweep; GPU validation of crash-class
candidates on devbox `tj-31yxx93` (B300, torch 2.11.0+cu128, cudnn-frontend
1.26.0 unpatched venv — the exact pin). Note: compute-sanitizer does **not**
support B300 (L20D) — use a B200 for sanitizer attribution; our proofs are
deterministic hard faults.

---

# 1. Actionables (read this first)

## 1.1 What production will actually hit (GLM-5.2 CP32, current pin + #814)

**Exactly one bug is live with nothing else needing to go wrong:**

| Finding | Why live | Likelihood | Impact per event |
|---|---|---|---|
| **#396 TMEM WAR race** (SM100 bwd dKV drain, head_dim 576/512) | Race on exactly our shape/arch, run every backward step; zero protections | Low per step (upstream needed a 1µs injected delay to force it), but millions of steps × 32 GPUs ⇒ expect occasional hits over a long run | **Silent dkv corruption** — a few wrong gradient values for one microbatch; usually absorbed as noise, tail risk = mystery loss blip or quietly degraded checkpoint. No crash, no log. |

Everything else either needs a second bug/regression to become reachable
(§1.2) or a config change (§1.3).

**Do now:**

1. **Carry the #396 backport** —
   `server/patches/cudnn-frontend-1.26.0-dsa-bwd-tmem-war-race.patch`. Only delta vs
   upstream: `barrier_id=10` (v1.26.0 has no barrier 9; forward-compatible with
   #395). ~+8% backward cost on the 576 path per upstream. GPU-validated
   (§4): happy-path 576/512 backward clean, no deadlock, numerics unchanged.
2. **Land #814** (already in flight); track upstream convergence of NVIDIA #410
   vs our #445 (functionally identical — same three guarded loops).

## 1.2 Latent landmines — one invariant away, cheap to defuse

| Finding | What must break first | Likelihood of that | Impact if hit |
|---|---|---|---|
| **Empty-row backward** (`topk_length==0` → negative-index OOB read + wild gather + pipeline hang; §2.1) | A zero-length row reaches the backward: packing emits `seq_len=0`, or top-k emits zero valid indices | Low today (packing raises on zero-length datums), **but packing is the fastest-moving code** — #801 just rewrote THD-CP packing for DP>1 | Job dead (hang at 100% SM or IMA). **GPU-proven on our pin** |
| **OOB index values in backward** (no upper-bound check; compact mode drops even `>= 0`; §2.2) | A producer bug upstream (exactly what #814 was) **and** failure of the glue's `_topk_in_bounds` filter — two independent layers | Low (double protection) | IMA, job dead. **GPU-proven on our pin** |

**Do now:**

3. **Carry the #439 backport** —
   `server/patches/cudnn-frontend-1.26.0-dsa-bwd-empty-topk-row.patch` (early-exit with
   zero-filled dq for `topk <= 0`; GPU-validated: empty-row case goes IMA →
   clean).
4. **Add a glue-side `topk_length >= 1` assert** wherever
   `all_rows_nonempty=True` is computed (`dsa.py` / `dsa_cudnn_kernels.py`).
   This is a tripwire on the invariant, not a kernel fix: a future packing
   regression fails CI loudly instead of hanging a 32-GPU job at 3am.
5. **Carry + file upstream the index-bounds hardening** —
   `server/patches/cudnn-frontend-1.26.0-dsa-bwd-index-bounds-hardening.patch`.
   GPU-validated: OOB indices now behave like `-1` (zeroed gather, skipped
   scatter); `dq` bitwise-equal to `-1`-masked reference. Converts the whole
   class from "mysterious IMA deep in the backward" into benign behavior.

## 1.3 Re-audit triggers — not reachable at this pin, activate on config change

Don't patch these now; the deliverable is the note so a future config diff
raises a flag:

| Trigger | Bug that activates |
|---|---|
| Unsharded backward with `total_S_q > 65536` (CP off / bigger per-rank shard) | int32 wrap in `sum_OdO` direct gmem addressing (`65536 × 64×512 = 2^31`); also top-k input addressing sits at exactly `2^31−1` elements for unchunked 8192×262144 (glue chunking currently gives 8× headroom) |
| FlashMLA with `s_q > 65535` | combine kernel `gridDim.y` overflow — cherry-pick FlashMLA `9241ae3` (#182) |
| Real attention sinks (≠ all `-inf`) | #421: 576/512 bwd returns `d_sink=0`, dq/dkv ~15% off. Production sink is `full(-inf)` (`dsa_cudnn_kernels.py` L2027) ⇒ inert today |
| `h_kv ≠ 1` indexer | `n_heads_kv` baked at compile, missing from `_interface.py` compile cache key ⇒ wrong-kernel reuse |
| `seqlen_k % 4 ≠ 0` | `indexer_forward` returns a non-contiguous slice (GPU-checked: top-k consumes it correctly today — API surprise, not wrong results) |
| `b > 1` fused DSA at long context | `_local_to_global_flat` int32 `local*b+b` overflow (`skv*b ≥ 2^31`) |
| H200 (SM90) | #385/#373 fixed by #388 (qh16/qh32 fwd/bwd corruption) |
| Unfreezing the indexer (`loss_coeff > 0`) | #426 latent SMEM handoff race in SM100 indexer backward |
| CUDA graph capture / explicit streams around DSA | #354 / #429 stream-ordering and capture hygiene |
| topk not multiple of 512 | forward pads index width to 512, backward consumes un-padded tensor (glue F6) — re-check before enabling |

## 1.4 Triage policy used here (and recommended going forward)

Don't exclude "real but unreachable" bugs — but don't treat them as equal
either. Rank by **number of independent protections × failure loudness × fix
cost**:

- **Zero protections → fix immediately, even if rare** (#396: live race, no
  guard, silent).
- **One protection + cheap fix + deadly failure → fix the kernel AND assert the
  invariant** (#439 + `topk_length ≥ 1` tripwire).
- **Two+ independent protections → document, don't patch** (e.g.
  `scatter_reduce_` clamp gap in `dsa_masking.py`).
- **Requires a different config → re-audit trigger, not a patch** (§1.3).

## 1.5 Landing mechanics (post-#814 vendored-wheel flow)

#814 (merged) replaced install-time patching with a vendored, pre-patched
wheel (`server/vendor/wheels/nvidia_cudnn_frontend-1.26.0+dsatopk1-*.whl`,
sourced via `[tool.uv.sources]`; `server/patches/` is the source of truth;
rebuild recipe in `server/vendor/wheels/README.md`). The follow-up PR lands
the three new patches by: adding them to `server/patches/`, rebuilding the
wheel as `1.26.0+dsatopk2` (pristine PyPI 1.26.0 + all four patches, applied
in the validated order: topk-oob → tmem-war-race → empty-topk-row →
index-bounds), repointing `pyproject.toml`, and relocking. A new
subprocess-style regression test
(`server/tests/unit/dp_worker/test_cudnn_dsa_sparse_backward.py`) covers the
empty-row and OOB-index cases against the vendored wheel.

Note on the actionable-3 tripwire: the "no zero-length rows" invariant is
**already enforced host-side** in trainers' own packing —
`packing.py` raises `ValueError("datum[i] has no tokens")` at both THD-CP
sites (L849, L923). Combined with the #439 kernel fix (zero row → handled),
no additional fork-side assert was added; the residual risk is a top-k kernel
producing zero valid indices for a non-empty row, which the regression tests
cover.

---

# 2. New findings (GPU-proven)

## 2.1 Backward faults on an empty (`topk_length == 0`) row — PROVEN

**Confidence: proven on GPU (deterministic IMA, exact production pin).
Independently found upstream — NVIDIA #439 (merged to develop post-1.26.0)
reproduces a permanent hang for the same root cause.**

**Where** (`sparse_attention_backward/dsa_bwd_sm100.py`, v1.26.0):

- `topk = mTopkLength[token_idx]`; `tile_count = ceil_div(topk, 64)` (~L866-871).
- The `load_KV` **prologue runs unconditionally once** before the
  `while tile_index >= 0` loop (~L1296-1349). With `topk == 0`:
  `tile_index = -1`, `idx = -64 + row ∈ [-64, 0)`; the read guard
  `idx < self.max_topk` is **true for negatives** → OOB read of `mTopkIdxs`
  (before the row; before the tensor base for `token_idx == 0`).
- `full_tiles = (0 % 64) == 0` → `_load_kv_rows(is_first=False)` → compact
  mode **unconditionally** gathers a KV row with the garbage index (§2.2
  mechanism).
- The mma loop runs zero iterations but `dq` is still stored from TMEM no MMA
  wrote (`dq` is `torch.empty`); the zero-tile pipeline deadlocks (upstream's
  symptom: 100% SM hang until SIGKILL).
- SM90 (`dsa_bwd_sm90.py` ~L1179-1265) has the same structure.

**Trigger.** Any call with `topk_length[i] <= 0`. Production today:
unreachable — packing raises on zero-length datums, padding rows get non-empty
bounds, top-k emits `min(seq_len, top_k) >= 1` valid indices per row. The
glue's *other* backward path (`all_rows_nonempty=False`) already filters zero
rows and appends a `topk_length=1` dummy (`dsa_cudnn_kernels.py` L2130-2142) —
the author knew; the production path (`query_valid_rows=None` →
`all_rows_nonempty=True`) just trusts the invariant.

**GPU proof** (`scripts/v1_bwd_empty_row.py`, B300, h=64, d=576/512,
topk=2048, 256 rows, one zero-length row at token 0):

```
control:  COMPLETED OK
zero:     torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

With the #439 backport: both complete.

## 2.2 Backward dereferences top-k index values with no upper-bound check — PROVEN

**Confidence: proven on GPU (deterministic IMA, both directions). No upstream
fix. The missing clamp was clearly intended: `max_seqlen_kv` is plumbed into
`reduce_dKV` (call site ~L1046, parameter ~L2084) and never used.**

**Where (SM100):**

- KV gather: `_copy_kv_row` (~L1175, `mKV[topk_idx, None, (0, batch_idx)]`)
  from `_load_kv_rows` (~L1253-1268). Non-compact checks only
  `topk_idx >= 0`; **compact mode checks nothing on the load side** — full
  tiles unconditional; the partial tile checks `idx < topk` (a position check,
  not a value check).
- dKV fp32 atomic scatter: `reduce_dKV_from_reg` (~L2261),
  `reduce_dKV_64_from_reg` (~L2318), `store_dKV` (~L2371), `store_dKV_64`
  (~L2424) — `>= 0` only, where present.
- An in-range-of-the-tensor but wrong-sequence index never faults: silent
  cross-sequence contamination of `dq` and `dkv`.
- SM90 same class (`_copy_row` ~L1337; `scatter_dkv_atomic` ~L1404
  unconditional in compact mode).
- Note the asymmetry with FlashMLA fwd, whose documented contract
  (`flash_mla_interface.py` L195) tolerates invalid indices set to "-1 **or
  numbers >= s_kv**". Anything valid for the forward but out of range is a
  backward crash.

**Trigger.** Any index `>= total_S_kv` or `< 0` in a read slot (compact: first
`topk_length[i]` slots; non-compact: any slot). At the pin this needs an
upstream producer bug — which is exactly what #814's phantom lanes produced
(uninitialized `torch.empty` output + phantom candidates ⇒ non-negative
garbage indices). Today the glue's `_topk_in_bounds` filter (verified on all
three `_indexer_topk_bshd` branches, L1317-1353) is the only guard.

**GPU proof** (`scripts/v2_bwd_oob_index.py`, compact mode):

```
control:  COMPLETED OK
oob_hi    (index = s_kv + 2^20):            cudaErrorIllegalAddress
oob_neg   (index = -2^20 in compact prefix): cudaErrorIllegalAddress
```

**Fix validation** (`scripts/v4_hardening_check.py`, hardening patch applied):
both variants complete; `dq` bitwise-equal to a `-1`-masked reference run;
`dkv` within fp32 atomic-order noise (3.1e-02, same as ref-vs-ref).

---

# 3. Upstream-fixed-after-1.26.0 (cherry-pick guidance)

v1.26.0 tagged 2026-07-07; `main` has **no** post-tag DSA commits; all fixes
on `develop`/PRs (unreleased, milestoned 1.27.0).

| PR | What | Status for us |
|---|---|---|
| **#396** | TMEM WAR race, SM100 bwd dKV drain, hd576: `tmem_dKV2/3` alias `dKV0/1` (v1.26.0 L113-115); MMA overwrites before reduce T2R. Upstream repro: dkv relL2 → 1.0 under 1µs delay | **Backport carried** (`server/patches/…tmem-war-race.patch`), GPU-validated stacked |
| **#439** | Zero/negative `topk_length` rows: zero-tile pipeline deadlock + garbage dq; fix early-exits with zero dq | **Backport carried** (`server/patches/…empty-topk-row.patch`), GPU-validated |
| **#421 / #405** | `same_hdim_kv` wrongly guards sink folding + `sum_dSink` launch → 576/512 `d_sink=0`, dq/dkv ~15% off with sinks | Inert at pin (sink = `-inf`); carry if sinks enabled |
| **#410** | NVIDIA's OOB-lane top-k fix — same three loops as our #445 | Watch which lands; either is fine |
| #395 | Latent SM100 bwd sync bugs (staged-store stage count, dealloc_tmem ordering) | Byte-identical cubin at our config; batch-cherry-pick when convenient |
| #354 / #429 | Stream-ordering + CUDA-graph capture hygiene | Only if we capture graphs |
| #426 | Latent SMEM handoff race, SM100 **indexer** backward | We don't run indexer bwd (frozen, `loss_coeff=0`) |
| #385 / #388, #373 | SM90-only fwd/bwd corruption (qh16/qh32) | Not our arch |
| #312, #331, #298, #317 | int32 extra-buffer chunking, convert grid overflow, reduce_dKV validity guard, score-recompute compact codegen | **Already in 1.26.0** ✓ |

FlashMLA pin b7643bd: includes the B200 FP8 accuracy fix (5aa668c) and #173;
lacks `9241ae3` (#182, combine gridDim.y overflow, `s_q > 65535` — §1.3).
Watch #158 (hardcoded device 0 / CUDAGuard, sparse path) for multi-GPU.

---

# 4. Validation log

Devbox `tj-31yxx93` (B300, 8×275GB, idle), venv
`/root/.cache/user_artifacts/trainers_main/server/.venv` (torch 2.11.0+cu128,
cudnn-frontend 1.26.0 **unpatched**). Scripts in `scripts/`; run with
`CUDA_VISIBLE_DEVICES=0`, kill switch `timeout -s KILL`.

| Test | Script | Result |
|---|---|---|
| bwd control (all rows len 65) | `v1_bwd_empty_row.py control` | OK |
| bwd one `topk_length=0` row | `v1_bwd_empty_row.py zero` | **IMA (deterministic)** → OK after #439 backport |
| bwd control | `v2_bwd_oob_index.py control` | OK |
| bwd index `s_kv + 2^20` (compact) | `v2_bwd_oob_index.py oob_hi` | **IMA (deterministic)** |
| bwd index `-2^20` in compact prefix | `v2_bwd_oob_index.py oob_neg` | **IMA (deterministic)** |
| fwd `sk=1029` (`sk%4≠0`) ⇒ non-contig out | `v3_fwd_noncontig.py` | confirmed non-contiguous; top-k results identical to contiguous clone (0/64 row mismatches) |
| hardening patch: oob_hi / oob_neg | `v4_hardening_check.py` | complete; `dq` bitwise = `-1`-masked ref; `dkv` diff = ref-vs-ref atomic noise |
| stacked: #396 + #439 + hardening | v1/v2/v4 suite | all pass; 576/512 path exercises #396's new barrier without deadlock |
| patch stack dry-run | `patch -p1` on pristine 1.26.0 | #814 → #396 → #439 → hardening all apply cleanly, syntax OK |

# 5. Static findings detail (latent; none production-reachable at pin)

### 5.1 Backward interface validation gaps (`sparse_attention_backward/`)
- `topk_length` dtype never checked on SM100 (SM90 asserts int32); compile key
  has `has_topk_length` but not dtype → same-shape int64 reuses an
  int32-compiled kernel. Glue always passes int32 ✓.
- No `topk_idxs.shape[0] == q.shape[0]` check; no `head_dim ∈ {512,576}` check
  on SM100 (SM90 asserts); `head_dim_v` hardcoded from `head_dim == 576`
  (`_interface_sm100.py` L59); `num_head % 64 != 0` breaks LSE tiling silently.
- Reduce-side index-tensor preload loops clamp to `topk` (row length), never to
  `max_topk` (tensor width): corrupt `topk_length[i] > max_topk` reads into
  following rows. Needs a corrupt length tensor — defense-in-depth.
- Dormant: `batch_idx` clobbered by `threadIdx.z` in several kernels; harmless
  while `batch_size` is hardcoded 1.

### 5.2 Indexer forward (`indexer_forward/`)
- Non-contiguous return when `seqlen_k % 4 != 0` and `out=None`
  (`_interface.py` L130-141, 229-232): TMA-pad path returns a sliced view.
  GPU-checked (V3): the top-k wrapper consumes it correctly (stride honored) —
  API surprise, not wrong results. Production pads `seqlen_k` to 64 ⇒ not taken.
- `n_heads_kv` baked at compile, missing from module-level compile cache key
  (`_interface.py` L150-164) → wrong-kernel reuse if `h_kv` varies.
  Production: `h_kv = 1` always.
- Kernel only writes computed tiles; the wrapper's `out.fill_(-inf)`
  (L207-209) guarantees defined values everywhere — confirmed present and
  stream-ordered. Real `-inf` floods in the top-k input resolve correctly
  post-#814 (traced through all four fp32 refinement rounds; candidate counts
  bounded by `num_cols` ⇒ no buffer overflow).

### 5.3 Indexer top-k (`indexer_top_k/`)
- `output_indices` is `torch.empty` (decode_varlen L684) while `api.py` L51-54
  documents unwritten rows as "initial (-1) state". In the single-CTA
  production path every row's CTA always writes its full output row (verified:
  `length <= top_k` branch writes `-1` tail; `length > top_k` writes exactly
  `top_k` entries) — only non-production dynamic-multi-CTA early-exit paths
  leave uninitialized output. Doc/contract bug; fix docstring or `fill_(-1)`.
- **CORRECTION (2026-07-30, LPS-1003):** the bullet above under-called this.
  The packed CP-THD chunked top-k IS the production path, and prod evidence
  (LPS-1003 Issue 2: boot/rebuild-window partition-tail destruction, healing
  with allocator history, poison-null signatures) indicates the production
  static path DOES under-write in some edge — the "writes exactly top_k
  entries" trace of the `length > top_k` radix branch has a gap, suspected in
  the `large_occupancy` (num_rows>148) + gmem-spill write-out. Dispatch is
  static one-CTA-per-row (multi-CTA variants compiled out on our entry path),
  so "non-production dynamic-multi-CTA only" is not where the risk ends.
  `fill_(-1)` is a required fix, not a docstring nicety. Full analysis:
  ../lps1003_loss_spikes/CODE_AUDIT_TOPK.md.
- `IndexerTopK.check_support` can't validate `seq_lens[i] <= num_cols` (device
  values); violating caller gets OOB input reads. Glue clamps at all four
  construction sites ✓.
- `+NaN` scores map to coarse bins 0-3 via `to_coarse_key`/`to_ordered` and are
  selected with top priority (torch.topk also picks NaNs first ⇒ consistent for
  `+NaN`; `-NaN` diverges — sorts below `-inf`). Footgun only with NaN scores;
  not memory-unsafe post-#814.
- `ComputeDynamicCTAOffsets.MAX_NUM_ROWS = 512` unasserted vs `num_rows`
  (dynamic-multi-CTA path only; not production).
- Compile-variant divergence (`num_rows > 148` ⇒ large_occupancy) reviewed:
  smem/thread selection consistent; `radix=256 ≤ threads` (or merge-blocks cap
  keeps `radix % threads == 0`) in all variants.

### 5.4 int32 headroom — verified safe at pin, zero margin unsharded
- Top-k input/extra-buffer: extra_buffer got explicit 64-bit row slicing (#312,
  in our pin). Input element offsets reach exactly `2^31−1` at maximal
  unsharded envelope (8192 × 262144) — glue chunking caps calls at
  `_TOPK_WRAPPER_MAX_SCRATCH_BYTES` = 2 GiB (~1024 rows × 262144 = 2^28), so
  production has ~8× headroom.
- Backward `sum_OdO` direct gmem addressing wraps at `total_S_q > 65536`
  (M-stride 64×512 = 32768; `65536 × 32768 = 2^31`). Production per-rank rows
  ≤ 8192 ✓.

### 5.5 Megatron glue / packing (vendored) — all non-production at pin
- `_local_to_global_flat` (`dsa_cudnn_kernels.py` L1007-1031): int32
  `local * b + batch` overflows when `skv * b ≥ 2^31` (comment states the
  precondition; unenforced). b>1 fused path only.
- Multi-packed CP fast path (L818-826) over-generates indices (beyond-causal,
  `>= sk`) when `topk >= max_segment_k`; **contained** because every
  `_indexer_topk_bshd` branch post-filters before compaction. Recommend
  self-clamping in the fast path.
- `scatter_topk_into_index_mask` (`dsa_masking.py` L277-299): `clamp_min(0)`
  only, no upper clamp before `scatter_reduce_` (all callers pre-filter).
- `build_packed_allgather_cp_local_positions` (`dsa_layout.py` L167-204):
  divisibility guard CPU-only; CUDA silently truncates if padding contract
  changes (packing.py enforces `2*cp_size` today).
- `get_packed_qk_cu_seqlens` (L295-317): per-stream fallback can mix padded-q
  with unpadded-kv coordinates if only one `_padded` field set (trainer sets
  both).
- Forward pads topk width to 512; backward consumes the un-padded tensor
  (L952-957 vs L2055-2059) — benign at topk=2048.
- `packing.py::_split_batch` (L491-500): `atomic_row_group_size` can round a
  microbatch above the token budget (OOM-class, bshd grouped-loss path only).
- Verified clean: padding rows never exceed the gathered KV extent; every row
  `ends - starts ≥ 1`; all `seq_lens ≤ num_cols`; doc-local→global offsetting
  applied exactly once; empty rows filtered in the `all_rows_nonempty=False`
  backward; dummy backward row contributes zero gradient.

# 6. Areas audited and cleared (no bug)

- Radix top-k invariants (post-#814): stage-1 histogram coverage of
  `[0, length)` exact (prologue + aligned vector tiles + leftover); strict
  inequality in the threshold search guarantees a threshold bin every round
  (inductive — the `count(==) == remaining` exact-tie case cannot stall it);
  `s_indices` (2048) and ping-pong `s_num_input`/`g_num_input`/`buffer[2]`
  accounting exact; fp16/bf16 single-round and fp32 four-round shift schedules
  consistent; ±inf bin placement monotonic; real `-inf` floods resolve through
  refinement without overflowing candidate buffers.
- `block_scan.py` named-barrier discipline (no full-CTA barrier inside
  partial-CTA scans; warp-scan masks exact).
- `compactify.py`: coverage exact, positions bounded; note it propagates
  garbage faithfully (no upper-bound check) — see §2.2.
- `local_to_global_dsl.py`: int64 offset math; binary search correct with empty
  batches; no upper-bound check on locals (propagates to §2.2).
- SM100 indexer forward: causal n-block skip proof holds (3 rightmost blocks
  masked, conservative by one for qhpkv ∈ {32, 64}); partial-K tiles always in
  the masked region; TMA store bounds-clipped by hardware; `sW`
  double-buffering race-free (Q-stage release ordered after the epilogue's
  tile-t sW read); head-reduction layout exact.
- Backward (besides §2.1/§2.2): TMEM column budget exact; pipeline arrive/wait
  counts balance; dKV cross-CTA accumulation (zeroed fp32 workspace + atomics +
  separate convert kernel) correct; `_copy_kv_row` covers 576 columns exactly
  once; cp.async 16B alignment holds for any int32 index (row stride 1152B =
  72×16) — mapping, not alignment, is the §2.2 failure mode.
- `utils/seqlen.py`, `utils/copy.py`, `utils/runtime.py`,
  `utils/tensor_conversion.py`: clean; layouts marked dynamic so
  shape-insensitive compile caches are sound.

---

## Note on the trainer-sampler mismatch question

Asked whether #396 explains trainer-sampler (logprob/KL) mismatch: **no, not
directly.** #396 is backward-only; a trainer-vs-sampler comparison is a
forward measurement on shared weights. Gradient corruption would degrade both
engines' weights equally, not create a per-token forward divergence. If the
mismatch predates #814, the top-k phantom-index bug is the prime suspect
(trainer forward selected wrong KV; sampler's separate indexer never had it).
Post-#814, look at forward numerics between the cudnn DSA stack and the
sampler's sparse path (tie-breaking, `-inf` handling, pad-row masking,
FP8/bf16, `attention_backend`), then weight-sync skew.
