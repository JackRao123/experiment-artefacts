# KERNEL DEFECT — PRIOR-ART MAP (THD varlen + CP attention corruption)

Author: serre, 2026-08-13 ~03:0x CDT, for cauchy (Jack order, fix mode; gauss
hunts the code, this is the prior-art angle). No GPUs touched.

**RESOLVED 2026-08-13 (cauchy, reproducer gate): the wheel gap was CAUSAL.**
PP1/CP8 ×3 on cudnn-frontend 1.27.0: mean 12.30481 (matches golden to 5e-5),
run-to-run spread 0.00077 vs 0.019 on the stale 1.26.0+dsatopk1 wheel. The
fix was the version bump this document's headline proposed — already pinned
in the tree since Aug 3 (commit 6717eaf0); the box's venv was simply stale.
The #543/#538 upstream-engagement contingency
(`UPSTREAM_ENGAGEMENT_CUDNN_543_538.md`) stays PARKED (do-not-send condition
held). Everything below is the pre-resolution map, kept as the defect-hunt
closure record.

**Signature being matched** (cauchy): cuDNN fused attention / FlashMLA under
THD varlen + context parallelism; multi-invocation forward with varying
packed-segment layouts per call; corruption front partway into
larger-than-predecessor invocations, escalating tailward; nondeterministic,
uninit-read class; suspected execution-plan/workspace cache keyed too coarsely
(layout not in key).

## TL;DR — the headline is a version mismatch, and the bump is already pinned

**The box w56lorq runs the RETIRED cudnn-frontend patch build
(`1.26.0+dsatopk1`), not the wheel the tree pins.** The tree has pinned
`nvidia-cudnn-frontend 1.27.0.dev20260803+git7478516` since Aug 3
(commit 6717eaf0, "retire dsatopk patch stack"), and that pin's commit message
carries direct A/B evidence on a B300 devbox: *"pristine 1.26.0 fires 12/12 at
4 GiB out with the erasure signature; new wheel 0/12 everywhere"* — i.e. the
bump is already proven to kill a nondeterministic DSA corruption class on our
hardware. The pinned snapshot carries the entire LPS-1003 race-fix series that
1.26.0+dsatopk1 lacks: the DSA wrapper stream-race root fix (#354), the SM100
DSA backward race/sync fixes (#395/#396/#426/#429/#439, incl. the TMEM WAR
race at *exactly GLM-5.2's head_dim 576/512* → silent dkv corruption), and
#421. If tonight's defect is in that series (the signature fits), the fix is
"run the pinned wheel," not new code.

## What the box runs vs what the tree pins (verified on-box)

| component | box venv (running tonight) | tree pin (server/pyproject.toml) |
|---|---|---|
| nvidia-cudnn-frontend | **1.26.0+dsatopk1** (dist-info verified) — the retired patch stack: 1.26.0 + the top-k OOB patch only (inferred from commit lineage: aa5d05e0/#814, Jul 28) | **1.27.0.dev20260803+git7478516** (vendored from-source wheel; override list, :330-334) |
| nvidia-cudnn-cu13 (backend) | **9.19.0.56** (dist-info verified) | **9.23.2.1** (:245) — "the backend ABI that matches the cudnn-frontend DSA namespace; 9.23.0.39 leaves libcudnn_cnn symbols undefined" (:240-244) |
| flash_mla | 1.0.0+b7643bd (nv_dev branch) | same pin (matches) |
| flash_attn (FA4) | 4.0.0b16 | — |
| transformer-engine | 2.16.0 | 2.16.0 (matches; `pad_between_seqs` present in the box's TE package — the #3331 fix is engaged) |
| torch | 2.11.0+cu130 | — |

Trainer-side code on the box IS current (73c24b00 includes #875, the
LPS-1003 dedicated-stream pinning for the indexer forward — verified
`git merge-base --is-ancestor` on-box). The staleness is wheel-level only.

## Ranked known issues matching the signature

### Tier 1 — direct matches (nondeterministic corruption / cache-key class)

1. **cudnn-frontend PR #354** (merged Jul 9; ships in 1.27.0; **MISSING on the
   box**). Three DSA fixes; the load-bearing one: `torch.cuda.ExternalStream(0)`
   does not preserve the legacy default stream, so PyTorch setup/copy ops in
   the DSA wrappers could run on a different stream than the CuTe kernel —
   **a genuine race, nondeterministic corruption class**, called "the LPS-1003
   root fix" in our pin commit. (Also: SM90 q_causal-offset alignment —
   per-packed-segment slicing — and a CUDA-graph `.item()` fix.) No env
   switch; the fix is the version. Partial local workaround existed as the
   #875 trainer-side stream pinning ("covers only indexer_fwd" — the box has
   this trainer-side piece, not the wheel-level root fix).
2. **cudnn-frontend PRs #395/#396/#426/#429/#439 + #421** (merged post-1.26;
   all in 1.27.0; **MISSING on the box**). SM100 DSA backward race/sync
   series. #396 is the standout: latent TMEM WAR race in the SM100 backward
   dKV drain for head_dim 576/512 — *exactly GLM-5.2's shape* — MMA warp
   overwrites dKV columns before the reduce warps' reads complete → **silent
   dkv corruption** (our dsatopk6 backport note, commit ddfbe228). #439: a
   topk_length≤0 row ran the KV-load prologue with tile_index=-1 (OOB read +
   garbage gather + deadlock). Backward-side, so not the step-0 forward gap —
   but mandatory for any training run on this wheel.
3. **cudnn-frontend PR #543** (OPEN, unmerged as of Aug 13 — **not in any
   build, including the pinned snapshot**). Two THD execute-path defects:
   (a) host prep (metadata build, D2H reads, H2D upload, descriptor/dummy
   buffers) enqueued on torch's *current* stream while the kernel launches on
   the handle's stream → race when they differ (pre-existing since the FROST
   engines landed, #476); (b) THD compile keyed on packed token totals →
   `lru_cache` degenerates into per-step recompiles under varying layouts.
   This is the literal "plan cache keyed on runtime layout" issue. Caveat for
   us: (a)'s documented exposure is direct graph-API users with an explicit
   handle stream; our DSA namespace path launches on torch's current stream —
   match the call path before weighing it.
4. **cudnn-frontend issue #538** (OPEN): the `cu_seq_len` (cu_ragged) form
   derives offsets assuming packed token stride `h*d`; wider-strided views are
   mis-addressed → corruption class. Check our THD tensors' strides against
   this before ruling out.
5. **TE #3331** (Jack's issue; closed Aug 8; **fix ENGAGED on the box** via
   mcore 57efae08b explicit `pad_between_seqs` + TE 2.16.0 carrying the kwarg):
   THD + CP (p2p) tail-padding auto-detect blindness → nondeterministic
   forward, dropped real keys, exact-zero real rows, nondeterministic
   gradients. Tonight's signature is the *zero-padding* case, so this specific
   one is excluded — but it is the same family, and it proves the family is
   real on our exact stack.

### Tier 2 — adjacent (same class, neighboring paths)

6. **cudnn-frontend #552** (OPEN, Aug 11): FROST THD execute does 2× `.tolist()`
   D2H syncs + host cumsum + H2D upload + per-total `compile()` keyed on
   packed totals. Perf-class on its face; it is the upstream acknowledgment
   that the THD plan/cache layer keys on runtime layout.
7. **cudnn-frontend #510** (OPEN): SDPA DSL torch glue ops run on torch's
   current stream, not the handle stream — race class.
8. **cudnn-frontend #514** (OPEN): torch-native SM80 SDPA adapters → scratch
   workspace carving conversion — the workspace under-validation class.
9. **cudnn-frontend #539** (referenced from #552): THD backward engines must
   adopt the declared-stride contract — backward-side stride assumptions.
10. **TE #3333** (OPEN, Aug 9): FusedAttention (cuDNN) THD backward produces
    wrong gradients on SM120; forward correct. Different SM (we are SM103) —
    listed as proof the THD-backward-wrongness class is live upstream.
11. **TE #3249** (OPEN): THD + learnable-softmax backward IMA (cuDNN 700) on
    Hopper. **TE #2892** (OPEN): fused_attn_fwd SIGSEGV when
    `cu_seqlens_q != cu_seqlens_q_padded` under FP8 blockwise — padded-vs-real
    varlen mismatch.

### Tier 3 — crash-class, already fixed (version windows; box status)

12. **cudnn-frontend #446 / PR #410** (Jack's production incident e3m916q;
    fixed in 1.27.0; **box HAS it** — dsatopk1 *is* this patch): varlen
    indexer top-k counted OOB -inf-filled lanes as candidates → per-row buffer
    overflow → IMA (Xid 43), or phantom lanes selected as winners → silent
    out-of-range indices. Data-dependent; triggered when scores collapse into
    the fp16 -inf bin. Our PR #445 merged as NVIDIA's duplicate.
13. **cudnn-frontend #406 / PR #407** (ours; fixed in 1.27.0): odd top_k >
    threads-per-CTA JIT compile failure / scalar-store fallback.
14. **FlashMLA #161** (OPEN): varlen option derivation, input validation,
    workspace handling improvements (CUTLASS internals). **#171** (closed):
    SM100 CUTLASS internal error at 1M seqlen. **#192** (OPEN): notes a B200
    sparse-decode *accuracy fix* (5aa668c) whose cost scales with topk —
    decode-side; our use is `flash_mla_sparse_fwd` (prefill sparse), pin
    1.0.0+b7643bd (nv_dev).

### Workarounds / env switches documented upstream

- None of the Tier-1 corruption fixes have env-switch escapes; they are
  version fixes. The documented interim measures: #552 lists "bucket the
  compile key" and caller-provided host totals as half-measures; #543 is the
  real fix and is unmerged.
- Our own LPS-1003 stopgaps exist in the tree as tripwires/canaries
  (907acd7e score-contract tripwire, 94896b53 score canary) — detection, not
  correction.
- Cross-reference for the F3 lane: cudnn-frontend #287 documents
  cuMemCreate NOT_PERMITTED errors interacting with PYTORCH_CUDA_ALLOC_CONF —
  adjacent to the expandable-segments/cuMem angle in
  `results/F3_PP1_NCCL_DIAGNOSIS.md` H2.

## The actionable

1. **Close the version gap on the box** (or whoever runs the repro next):
   the venv should carry the pinned `1.27.0.dev20260803+git7478516` wheel +
   `nvidia-cudnn-cu12==9.23.2.1` backend. The box's `1.26.0+dsatopk1` +
   `9.19.0.56` predates every Tier-1 fix. Why the box venv predates the Aug-3
   pin (image-baked venv vs skipped rebuild) is a devbox-provisioning
   question — worth one line in the morning report either way.
2. **Re-run the failing config on the pinned wheel** before any new code is
   written: the pin commit's A/B (12/12 corruption on pristine 1.26.0 → 0/12
   on the new wheel, B300, with an erasure signature) is exactly this
   defect's family.
3. Honest caveat for the Gate-1 decomposition: a race produces run-to-run
   variance; a layout-keyed cache bug produces deterministic-per-layout
   wrongness. The ~0.19 gap's stability across legs discriminates between
   them — gauss's code read owns that call; this map says both classes exist
   upstream and which versions carry which fixes.
4. If the defect survives the wheel bump: the open upstream items to engage
   are #543/#552 (plan-cache keying) and #538 (packed-stride assumption) —
   both open, both NVIDIA-engineering-labeled, both exactly our signature.

## Search coverage

- cudnn-frontend issues/PRs: varlen, workspace, plan-cache, nondeterministic,
  DSA, THD sweeps; Jack's author footprint (#406, #446); the pinned-fix pair
  (#354, #410) read in full; #543/#552/#538/#539/#510/#514/#520/#548 skimmed
  for relevance.
- TransformerEngine issues: varlen/THD/nondeterministic sweep (open+closed);
  Jack's footprint (#3331).
- FlashMLA issues: varlen/sparse/nondeterministic/cache sweep.
- NVIDIA forums: JS-walled search, no yield — the effective vendor channel is
  the GitHub trackers (NVIDIA collaborators engage directly on our issues;
  several fixes above are labeled orig-nv-eng).
- Our own repo: the full dsatopk1→dsatopk6→1.27.0-dev pin lineage from git
  history (commits aa5d05e0, 7826f264, ddfbe228, 6717eaf0, 8bac180f, #875).
