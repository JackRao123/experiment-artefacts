# LPS-1062 optimization night — notebook

> **2026-08-10 night — new direction: PP2/CP8/EP8 @131k, fresh start from
> tip of main.** Jack's call: prior artifacts here are largely slop; restart
> clean. That effort logs in **`pp2cp8ep8/NOTEBOOK.md`** (goal in
> `pp2cp8ep8/GOAL.md`); this file stays as historical record.

> **2026-08-09 PM — MFU convention change (Jack-commissioned).** Every
> mfu3x/hfu cell in this file has been converted in place to LoRA-corrected
> accounting (frozen base = dgrad-only backward + audit-constant fixes):
> method in the rewritten `runs/overnight_20260809_mfu_sweep/mfu.py`; per-label old↔new mapping and
> ×-factors (mfu3x ×0.686–0.736, hfu ×0.756–0.793 by L) in
> `runs/overnight_20260809_mfu_sweep/mfu_lora_correction.md`. Raw tok/s/GPU, step times, memory,
> losses are method-independent and unchanged. Pre-correction values:
> the mapping table, `runs/overnight_20260809_mfu_sweep/mfu_pre_lora_20260809.py`, and the results
> JSONs (old-convention MFU fields through the v2 cycle; LoRA-corrected
> fields from the BF-only cycle onward). Prose milestones below keep their
> original figures as historical record where marked.

---

## 2026-08-09 (day) — MoE dispatcher host-sync elimination (pascal orchestrating; laplace=attribution, ramanujan=implementation, gibbs=validation)

**Goal (Jack-approved):** eliminate the 26,684 `aten::nonzero` (~20.6s CPU) +
~29k `cudaStreamSynchronize` (20.0s) + 33.6k `cudaMemcpyAsync` (15.0s) per
524K-token step in the mcore alltoall dispatcher's host-side split
bookkeeping — co-dominant with SendRecv (24.5s/50%) since the Aug-7 NCCL fix.
Plan: runs/overnight_20260809_dispatcher_hostsync/HANDOFF.md. Correctness rule: env-gated patch
(`BT_FUSED_DISPATCH_SPLITS=1`, default OFF), standalone bitwise parity test,
on-box A/B with loss canary (>5e-3 drift = stop).

**Milestones:** (1) laplace: trace attribution of the sync time by call
site/phase from exp05d.pt.trace.json → go/no-go + win bound; (2) ramanujan:
patch + parity test green (Mac checkout, diffs via pascal); (3) gibbs: fresh
2-node B300, A131-131k-d4 baseline repro (620/steady ~645), then
same-kit A/B; (4) diff + writeup archived to runs/overnight_20260809_dispatcher_hostsync/.

**Numbers to beat (steady tok/s/GPU, 524K tok/step, ship env + TF32):**
golden@131K 645 · CP8/DP2@131K 745 · customer-shape 16k-d32 734.

**Log:**
- ~08:0x — pascal took handoff from feynman; operators briefed (fresh
  sessions, no prior context). gibbs provisioning; laplace on the 1.05 GB
  exp05d kineto trace; ramanujan code-reading + candidate enumeration,
  design held for attribution.
- ~09:0x — laplace ATTRIBUTION.md + ramanujan CANDIDATE_FIXES landed,
  **independently convergent on call sites** (no python stacks in trace;
  both matched op-motifs/counts exactly: 156 = 78L×2mb, 85 = 17×5/layer).
  → **F6 premise correction** (below). pascal ruling: GO on FIX A + FIX B
  (+F if cheap), C deferred (medium risk), D folded, E n/a. Separate
  default-OFF env gates per fix; Aug-7 on-box patch-and-archive flow
  confirmed for the vendored 3rdparty tree.

- ~10:0x — laplace REVIEW_FIXA.md: FIX A stream-safe with 8 binding reqs
  (key: attach at `_run_sparse_attention_forward:2037` — the 156 syncs are
  UNCONDITIONAL, `FusedSparseAttentionFunc.backward` never passes
  `all_rows_nonempty`; per-call pinned buf on ctx; `is_grad_enabled` gate
  for the recompute no-grad pass; graph-capture guard; forced-fallback
  parity case). Fallback frequency ≈ 0 on bench AND customer THD data
  (packer pads inside `cu_seqlens_padded` → every row keeps a non-empty
  causal range; must ship a per-step fallback counter to verify). A/B
  acceptance = counter table (REVIEW_FIXA §2c: 26,684 nonzero → ≤200,
  >5ms slices → 0, etc.) + canary ≤5e-3 + throughput ≥ −2% (expect
  +1–4%; >+6% = artifact). MAX_CONNECTIONS confirmed unset in trainer.
  ATTRIBUTION errata: eventSync 300/4.13s (both threads). pascal signed
  off; ramanujan proceeding to diffs. gibbs box qr4ggv3 booting.
- ~10:3x — laplace `runs/overnight_20260809_dispatcher_hostsync/check_acceptance.py` validated:
  baseline-exp05d profile reproduces the §2c baseline column 16/16 on the
  exp05d trace; negative control (post-patch profile on unpatched trace)
  fails 11/16 with the must-not-change rows passing — discriminates.
  Version note: trace_processor v57+ keeps 5,405 overlapping kineto events
  v56.1 drops (kernel_count 361,032 vs 355,627; checker-authoritative =
  v56.1, ±2%). Per-gate bisection profiles (post-patch-A/-B/-AB/-ABF)
  added + validated for internal consistency (each gate profile fails
  exactly its moved rows on the unpatched trace). Encoding notes: post-A
  gt5ms bound is ≤4, not 0 (4 of the 160 big nonzeros are step-level
  `_index_put_impl_`, out of scope); A adds 156 tiny pinned flag copies
  (dtoh bound 30,000 A-only / 3,200 AB).
- ~10:5x — gibbs box qr4ggv3 UP + healthy (2×8 B300 ali; 0e0b65a6 dirty =
  Aug-7 patches; TF32 grep=1; kit md5 ✓; HTTP on slurm node0
  tj-qr4ggv3-1, gotcha confirmed; /proc env ✓ full prod warmup).
  Baseline A131-131k-d4 r3 running.
- ~11:1x — ramanujan PHASE 2 Mac-CPU-green: 3 patches archived
  (`runs/overnight_20260809_dispatcher_hostsync/patches/0001-fix-a-dsa-bwd-async-nonempty`,
  `0002-fix-b-dsa-cp-layout-cache`, `0003-fix-f-thd-rope-host-cache` +
  PATCH_NOTES + tests). REVIEW_FIXA reqs implemented as binding; parity A
  19 PASS (incl. forced-fallback), B 191 PASS, F PASS; patches
  `git apply --check` clean vs on-box mcore pin d3932e757c (touched files
  byte-identical to Mac). **SITE CORRECTION claimed for the 14.8s memcpy
  class:** NOT sort_chunks tolists (TE fused chunk_sort active,
  te_moe::chunk_sort_fwd ×600) but THD RoPE bookkeeping in
  `_apply_rotary_pos_emb_thd` (rope_utils.py:222-239 via
  absorbed_mla.py:626/636; 616 `aten::to` >1ms = 14.8s, 2/layer-mb/pass).
  FIX F regated `BT_THD_ROPE_HOST_CACHE`. Conflicts with ATTRIBUTION §3 →
  sent to laplace for adjudication + diff review (B aliasing, F cache
  keying — tensor-identity reuse hazard, A compliance). On-box
  application HELD until verdict.
- ~11:4x — laplace VERDICT: **ramanujan's RoPE site correction upheld**
  (te_moe::chunk_sort_fwd ×600 → sort_chunks tolists are dead code under
  moe_permute_fusion; 619 `aten::to` >1ms/14.8s = `_apply_rotary_pos_emb_thd`
  motif exact; ATTRIBUTION §3 errata posted; 587-vs-616 reconciled as
  different op classes). Diff reviews: 0001 all 8 reqs verified SIGN OFF;
  0002 aliasing audit clean SIGN OFF; 0003 SIGN OFF conditional on one
  hardening — add `cu_seqlens._version` to the cache key (module-level
  run-lifetime cache; future in-place mutation would silently serve stale
  seqlens; _version keying → cache miss instead). Checker: post-F
  memcpy_gt1ms bound 0→≤8 (cache-miss D2Hs), re-validated. ramanujan
  applying the one-liner + mutation parity case; gibbs GO follows.
- ~12:0x — 0003 regenerated with _version keying + mutation-miss parity
  case (miss → fresh copy → output == unpatched ref; sensitivity
  asserted). Mac-CPU final: A 19 / B 191 / F 167 PASS; apply-check clean
  vs d3932e757c. **GO issued to gibbs**: canary-rerun anchor → apply
  0001..0003 → on-box GPU parity → gated boot (3 gates + ship env) →
  timed A/B → profiled capture → check_acceptance post-patch-ABF →
  fallback counter.
- ~12:4x — baseline REPRODUCED with valid canary (row A131-repro2-w2:
  630/634 steady, dloss ≤1.1e-3, gn exact). Patches 0001..0003 applied
  clean on-box (archived `runs/overnight_20260809_dispatcher_hostsync/patches/box-applied-qr4ggv3.patch`, md5
  e9670205, cross-node grep-verified); on-box GPU parity ALL GREEN
  (A 22/22 incl. forced-fallback telemetry; B, F incl. mutation-miss).
  One test-only harness fix by gibbs (Generator device on GPU leg,
  2 lines, Mac copy synced). Gated reboot in progress
  (BT_DSA_BWD_ASYNC_NONEMPTY=1 BT_DSA_CP_LAYOUT_CACHE=1
  BT_THD_ROPE_HOST_CACHE=1 + ship env).
- ~13:0x — capture-shape adjudication (gibbs): exp05d counter bounds are
  2-mb-calibrated; bench shape is 4×131072 (4 mb). Ruling: capture at the
  bench shape (profile the workload we accept). laplace pushed
  `post-patch-ABF-4mb131k` profile, all mb-scaling rows re-derived +
  documented (drains 312, layout 340/≤450, eventsync ~912 exp with
  gt1ms ≤650 as the A-replay-check, F misses 4/step, dtoh ≤5,500,
  kernel ≤430k; sendrecv ±10% and idle ≤6.5s flagged loose); validated
  against both traces. Timed A/B bench running.
- ~14:0x — **PROVENANCE TWIST (laplace, from the patched capture): FIX A
  was INACTIVE in the capture** — all 312 FSA-bwd drains present, zero
  flag events/probe kernels — while **B and F verified working** (layout
  nonzeros → 348 [87/mb, +2/mb secondary path vs derivation]; blocking
  memcpys 587 → 4 = exactly the predicted per-mb cache misses; memcpy CPU
  15.0 → 0.16s; kernel count 188.5k). If A's env var proves present on
  the capture boot, A was likely inert on the timed boots too → **the
  +14% is B+F alone and A is unmeasured** (0 fallback WARNINGs is not
  activation evidence — inactive A prints nothing; patch-quality gap:
  inertness is silent). SendRecv in the BF-only trace: p50 9.07ms ≈ exact
  size-halving vs exp05d, total 27.16s (+11%) → NO de-staggering
  signature from B/F; front-runner mechanism = launch-pipeline
  decompression. Diagnosis running (gibbs: /proc dumps + capture window;
  ramanujan: 0001 inertness audit + definitive activation check).
  Capture sequence re-planned: BF-only (in hand) / unpatched-4mb /
  A-active-4mb three-way.
- ~14:3x — diagnosis: A's gate char-exact in worker /proc on BOTH boots,
  patch confirmed applied on-box → not a launch-env problem. BUT the
  capture was the FIRST user step on a fresh boot (59.2s wall = first
  window + profiler), and FIX A no-ops under is_current_stream_capturing;
  boot logs show cuda_graph_modules activity → **inert-A may be a
  window-1 artifact, not a code bug**. Existing patched trace: timing
  rows contaminated, B/F count rows valid. Re-plan: steady-window capture
  pair at matched shape (unpatched-4mb-steady on the gates-off boot →
  gated-4mb-steady after reboot). ramanujan re-scoped: (a) loud
  activation telemetry for all 3 gates (armed/disabled boot line +
  counter at a level that reaches trainer_srun.log) — REQUIRED in final
  patches; silent inertness cost a capture cycle; (b) desk-verdict on
  whether zero-probes-at-window-1 is even consistent with the guard
  placement. Protocol addition: no first-step captures.
- ~15:0x — **ROOT CAUSE (ramanujan, window-1 theory REFUTED 3 ways):
  FIX A's `torch.is_grad_enabled()` gate (review req §5) can never be
  True inside `autograd.Function.forward` — torch runs Function.forward
  under internal no-grad regardless of outer context (empirically
  verified on-box torch 2.11; behavior since ~1.10). Probe disabled
  unconditionally → A inert in ALL windows, BOTH boots.** Refutation of
  window-1: (i) drains visible as eager aten ops ⇒ not graph-captured;
  (ii) replay kicks run on the autograd thread which never captures;
  (iii) gate bug is total. ⇒ **the timed +14% steady (724) is a
  confirmed B+F-ONLY result — accidental clean bisection; A's win is
  still unmeasured and potentially additive** (312 drains remain).
  Corrected 0001 shipping: gate removed (discarded-pass kick provably
  harmless — ctx dropped), + warning-level armed/disabled telemetry on
  all 3 gates + per-window counter reaching trainer_srun.log; inert v1
  archived for provenance; laplace delta-review before on-box; parity
  test to be fixed to exercise the REAL call context (it bypassed
  Function.forward — the gap that let inertness through). Lesson for
  synthesis: activation telemetry is mandatory for env-gated patches;
  parity green ≠ active.
- ~15:3x — 0001-v2 shipped (gate removed, WARNING-level armed/counter
  telemetry on all 3, silent-fallback→counted, checkpoint-replay parity
  test that fails on v1 + source guard). laplace delta-review: SIGN-OFF
  (ordering byte-identical to reviewed v1; discarded-pass kicks ~15ms/step
  negligible; GC-safe via CachingHostAllocator stream-recording; zero new
  hot-path syncs; probes_missing=0 = the inertness detector). GO issued:
  swap v1→v2 (+0002/0003 telemetry versions), parity, gated boot with
  **armed-line grep as hard gate**, timed r3 vs anchor, steady capture,
  checker. Open question: A's marginal win on top of B+F's 724.
- ~16:0x — unpatched-4mb-steady capture (2.08 GB, step 3→4, wall 53.95s;
  gates-off prep bench 643 = patched-checkout-ungated runs at baseline →
  env-gating inertness empirically validated). laplace derivation
  validation: **counts exact** (all 7 count rows dead-on incl. kernels
  704,681), **wait-times all ~2× high → corrected scaling law:
  per-block wait ∝ compute-queue depth ∝ 1/M at fixed tokens, so total
  host-block CPU is M-INVARIANT** (nonzero 20.6→21.4s, memcpy
  15.0→13.9s conserved across shapes); >1ms counts fall via shorter
  per-call waits; gpu_idle scales with launch count not wall (8.52s
  measured; post-ABF bound recalibrated 6.5→7.5s, was guaranteed
  false-FAIL). Profiles now measured-calibrated, re-validated ALL PASS.
  Mechanism discriminators PRE-REGISTERED for the gated-steady read:
  SendRecv total/p50 flat + idle/gap rows fall ⇒ launch-pipeline
  decompression; SendRecv total/p50 shrink ⇒ de-staggering. Unpatched
  4mb refs: SendRecv 2,700/25.77s p50 9.19ms; block coverage 32.17s
  (60%); comm-idle 28.19s (8.36 in-window); gaps 27.09s.
- ~16:2x — swap v1→v2 clean on-box (combined diff re-archived as
  `box-applied-qr4ggv3-v2.patch`, armed markers sibling-verified). On-box
  GPU parity v2 ALL GREEN incl. _checkpoint_replay_test (probe kicks in
  both passes, probes_missing 0, bitwise grads) + v1-regression guard.
  Second test-only harness fix by gibbs (lse device on cuda leg of the
  replay test; Mac synced, md5 550d147a). Gated boot next; armed-grep is
  the hard gate before benching.
- ~17:0x — laplace final analysis (ATTRIBUTION addendum written): (1) the
  34.02s eventsync FAIL decomposed — **A's 312 flag events = 0.459s
  total, p50 5.5µs: the replay argument HOLDS, A works as designed**; the
  inflation was the dispatcher's 600 pre-existing events (2.22→33.56s,
  replay p50 77ms) because B+F's decompression lets the host run ~1 layer
  ahead, lengthening every wait-for-GPU sync in wall terms — benign (GPU
  idle 1.82s). (2) **MECHANISM VERDICT: launch-pipeline decompression
  CONFIRMED, de-staggering REJECTED** — SendRecv flat, idle −6.7s ≈ wall
  win, host-block coverage 60%→1%. 2mb bound failed to transfer because
  starvation SHARE is shape-dependent though block CPU totals are
  M-invariant. (3) **A-park refinement: A is free today because the
  dispatcher's replay eventSync (77ms p50) throttles the autograd thread
  upstream of A's read — A's exposure is a sequela of FIX C (dispatcher
  sync removal, quantified target: 300 replay syncs = 23.7s CPU), not of
  comm wins.** Ship B+F, park A default-off, revisit with FIX C.
  (4) Checker: eventsync split into dispatcher-informational vs A-flag
  sharp rows; gated-v2 = ALL PASS; gpu_idle tightened 3.5s.
### SYNTHESIS (pascal, 2026-08-09 ~15:0x PDT) — dispatcher_opt project CLOSED

**Outcome: ship FIX B (`BT_DSA_CP_LAYOUT_CACHE`) + FIX F
(`BT_THD_ROPE_HOST_CACHE`) — +11–13% steady at 131k×d4 (715 vs 633–645),
+18.5% at the customer-dominant 16k×d32 shape (726 vs 612), loss-parity
proven at both shapes (131k ≤2e-3; 16k matched-step ≤5e-4), memory flat,
DP-safe golden mesh (no F2 exposure). FIX A parked default-OFF (isolated
effect ≈ 0; unblocks only after FIX C). Full story, evidence and tickets:
`runs/overnight_20260809_dispatcher_hostsync/REPORT.md`.** Key science: premise correction F6 (sites =
DSA bwd + DSA layout + THD RoPE, not the dispatcher; mechanism =
launch-pipeline decompression, SendRecv flat, GPU idle 8.5→1.8s;
host-block CPU is M-invariant, starvation share is shape-dependent).
Key process lessons: activation telemetry is mandatory for env-gated
patches (v1 shipped inert behind an always-False `is_grad_enabled()`
inside `Function.forward` and passed all parity); parity tests must
exercise the production call context; no first-step profiled captures;
pre-register mechanism discriminators before the decisive trace.
Box qr4ggv3 torn down at close. Patches remain applied (gates
default-OFF = inert) in the shared checkout; diffs archived in
`runs/overnight_20260809_dispatcher_hostsync/patches/`.

- ~13:1x — **MFU method change in progress (Jack-commissioned, session
  fibonacci):** mfu.py rewritten to LoRA-corrected FLOP accounting
  (frozen base skips wgrad: useful = 2·F_matmul + 3·F_attn(L) +
  3·F_lora(r); executed = 3/4/4; + audit corrections). Prior mfu3x
  ~×1.30–1.39 high, hfu ~×1.21–1.27 high; **raw tok/s/GPU unaffected.**
  Process settled after a hold: results JSONs stay untouched (derived
  correction table instead), NOTEBOOK edits remain single-writer
  (pascal; MFU cells in this section to be converted from fibonacci's
  table at final-writeup time), REPORT/READMEs = fibonacci.
  Pre-change tree snapshot:
  `~/perf_profiles/lps-1062/backups/lps_1062_perf_snapshot_20260809-1315.tgz`.
  On-box bench kit stays frozen on the old driver for run-to-run
  consistency; conversion at write-time. [Superseded ~13:2x: a re-stage
  race put the new driver on-box BEFORE any bench ran on it — pascal
  blessed it for the final benches (raw numbers driver-independent,
  --lora-rank 32 explicit, MFU labeled new-formula). fibonacci's
  correction table landed (`runs/overnight_20260809_mfu_sweep/mfu_lora_correction.md`, incl. a
  C-D-131k-d4 transcription-slip catch) and this file's MFU cells are now
  CONVERTED per the top-of-file note.]

**F6 — mission premise corrected: the 26,684 aten::nonzero are NOT MoE
dispatcher splits, and they are ~fully comm-overlapped today.**
(a) 93% of the 20.6s nonzero CPU = 156 calls (78 layers × 2 mb) of
`torch.nonzero(topk_length>0)` in **DSA sparse-attn backward** compaction
(`dsa_cudnn_kernels.py:2130`), ~123ms full pipeline drain each on the
autograd thread. (b) 26,520 calls = boolean-mask indexing in the DSA CP
packed-layout builders (`dsa_layout.py:159-161,187-188` via `dsa.py:1761`),
85/layer-mb/pass, identical inputs per microbatch, replayed by full
recompute — cacheable. (c) The 15.0s cudaMemcpyAsync = blocking pageable
`tolist()`s in `sort_chunks_by_idxs` (`moe_utils.py:569-573` via
`token_dispatcher.py:768/803`) — the only genuinely-dispatcher item;
`sorted_idxs` isn't in the `_maybe_dtoh_and_synchronize` batch. (d) mcore
0.19.0's dispatcher has NO per-expert nonzero loops (Aug-7 attribution
stale); its own D2H is already side-stream+event (2.1–4.1s). (e) **Win
bound now: GPU-union idle inside all >1ms host-block windows = 0.70s of
the 48.8s step** (the blocks ride inside SendRecv slack) → realistic
immediate win ~1–2s/step (+2–4% tok/s), hard ceiling 4.74s. The Aug-7
"co-dominant" read conflated CPU-block time with critical-path time.
**Strategic value stands: every second shaved off comm exposes ~1s of
these blocks, so the surgery is a precondition for future comm/DeepEP
wins — proceeding with corrected expectations.** Evidence:
`runs/overnight_20260809_dispatcher_hostsync/ATTRIBUTION.md`, `runs/overnight_20260809_dispatcher_hostsync/CANDIDATE_FIXES_ramanujan.md`.

| # | config | tok/s/GPU | step (s) | mfu3x | hfu | max GPU mem (MiB) | loss canary | notes |
|---|---|---:|---:|---:|---:|---|---|---|
| A131-repro1 | expA131 EP16/CP16 2×8 qr4ggv3 (131K boot, full warmup), 131k×d4 r3 | **623** (609/617/645; steady 645) | 52.6 | 5.7% | 8.2% | 201,275/194,831 | throughput/mem ✓ vs ref (620/645/52.8); loss INVALID — warmup-datums=1 vs ref w2 (rng-stream rule; gibbs self-caught) | baseline repro on the validation box; canary rerun with --warmup-datums 2 on fresh reboot → becomes the A/B anchor |
| A131-repro2-w2 | same, fresh reboot, --warmup-datums 2, 131k×d4 r3 | **630** (623/634/634; steady 634) | 52.0 | 5.7% | 8.3% | 201,711/194,831 | **✓ dloss ≤1.1e-3 vs last-night w2 ref, gn main2 0.6645 exact** | **BASELINE REPRODUCED — canonical A/B anchor** (`runs/overnight_20260809_mfu_sweep/results/A131-131k-d4-repro2-w2.json`); patches being applied |
| A131-patched-ABF | same boot-shape, gated boot (A+B+F gates + ship env, /proc-verified), 131k×d4 r3 + same-boot re-measure r3 | **687** (624/723/725; steady ~724) · re-measure **721** (721/721/720) | 47.7 / 45.5 | 6.3% | 9.1% | 201,439/194,735 | **✓ parity: dloss ≤1.5e-3 all windows vs anchor**; 0 fallback WARNINGs (8 windows) | **+14% steady vs anchor (724 vs 634), +12% vs last-night 645; ~6.2s/step** — later RE-ATTRIBUTED: FIX A inert (gate bug) ⇒ this row = **B+F only** |
| A131-patched-ABF-v2 | v2 patches (A ACTIVE: armed-lines ×16 ranks, probes_missing=0), same protocol, r3 + same-boot re-measure | **672** (631/706/683) · re-measure **707** (713/701/707; steady ~707) | 48.8 / 46.3 | 6.1% | 8.9% | ≈flat | **✓ parity: dloss ≤0.9e-3 all windows** | **A's marginal win over B+F ≈ 0 (707 vs 723, ~−2%)**: A works exactly as designed (312 flag events 0.46s, p50 5.5µs — replay argument holds) but buys nothing TODAY — the drains it removes were already throttled cheap by the dispatcher's upstream replay eventSync (77ms p50); the 34.02s eventsync FAIL = that dispatcher class runahead-inflated by B+F (benign, GPU idle 1.82s). Steady capture: eventsync 912 EXACT (600+312), nonzero 356/0.11s, gpu_idle 8.52→**1.82s**, SendRecv 26.03 FLAT ⇒ launch-decompression mechanism per pre-registered discriminators; checker ALL PASS after eventsync-row split |
| A131-patched-BF | BF-only gated boot (A DISABLED ×16 verified), same protocol, 131k×d4 r3 | **701** (674/723/707; steady ~715) | 46.8 | 6.4% | 9.3% | 201,915/194,975 | **✓ dloss ≤2.0e-3 vs anchor** | **A isolated same-code gate-off/on: 715 vs 707 ≈ −1% = ZERO within noise (telemetry confound killed). TOTAL PATCH WIN vs unpatched 634–645: +11–13%, ALL from B+F** |
| A131-16k-d32-patched-BF | same boot, 16k×d32 r2 (customer shape, golden mesh) | **726** (726/725) | 45.2 | 5.7% | 8.4% | 201,959/194,975 | 12.2618(w)/12.2559/12.2400, gn 0.617/0.579/0.544; cross-config vs expB-16k refs +35e-3 — CONFOUNDED (step-count mismatch ~0.02 + mesh reduction order); deconfound run ordered | **customer shape FULL SPEED on patched golden mesh (726 ≈ B's 734 −1%)**; becomes the (A131,16k,patched) canary ref; matched-step unpatched 16k reference = final boot cycle |
| A131-131k-d4-unpatched-rep3 | gates-off boot (0 gates in /proc), exact BF-boot sequence replicated, 131k×d4 r3 | 616 (586/629/637; steady ~633) | 53.2 | 5.6% | 8.1% | 201,415/194,591 | ✓ dloss ≤1.7e-3 vs anchor | third unpatched replicate — baseline band 633–645, rock solid |
| A131-16k-d32-unpatched | same boot, 16k×d32 r2 (matched steps + content vs patched-BF 16k) | **612** (607/617) | 53.6 | 4.8% | 7.1% | — | 12.2617(w)/12.2565/12.2399 gn 0.604/0.582/0.544 | the missing (A131,16k,unpatched) ref. **DECONFOUND VERDICT: patched-vs-unpatched dloss ≤5e-4 all windows — BENIGN; the +35e-3 vs expB was mesh+step confound, NOT patch-induced. BONUS: 16k patch win = 726 vs 612 = +18.5%, the biggest — customer-dominant shape benefits most (32 docs/step ⇒ most host bookkeeping eliminated)** |

---

## 2026-08-09 overnight — parallelism/MFU sweep for the customer regime (feynman orchestrating; gibbs=A, ramanujan=B, laplace=C operating)

**Goal:** customer trains GLM-5.2 (CP THD) on ~608K samples — 70% ≤32K,
28.5% 32–64K, 1.4% 64K–131,072 (max 131K). Which parallelism config + simple
levers maximize MFU at packed datums of 32K–262K tokens? (Prior nights
optimized 256K–524K operating points.)

**Code:** trainers_main @ ef4ea4a8 (or later at provision) + TF32 LM-head
cherry-pick `e42e843c` (`BT_TF32_LM_HEAD=1`). **Ship env on all runs:**
`NCCL_IB_QPS_PER_CONNECTION=8 NCCL_IB_SPLIT_DATA_ON_QPS=1
NCCL_NCHANNELS_PER_NET_PEER=8`.

**MFU method (updated, `runs/overnight_20260809_mfu_sweep/mfu.py`):** FLOPs/token now
length-dependent — matmul 84.3 GF + DSA attn 10.5 GF (const) + indexer
23.6 GF × (L/262144). Reproduces the old constant (118.3 GF) at L=262,144
exactly; old numbers in this notebook remain valid. `mfu3x` = 3×fwd (useful
FLOPs, notebook convention), `hfu` = 4× (full recompute). Peak 2.5 PF/GPU.
[2026-08-09 PM: this method superseded by the LoRA-corrected mfu.py; all
table cells converted — see note at top of file.]

**Protocol:** `runs/overnight_20260809_mfu_sweep/PROTOCOL.md`; driver `runs/overnight_20260809_mfu_sweep/bench_driver2.py`
(seed 0xB300, warmup + N main windows, per-GPU mem polled). Canary = first
run at each seq len per stack; later runs drift-checked (≤2e-3 noise).

Boxes (ali, B300 275,040 MiB/GPU): A = 2-node (gibbs), B = 2-node running
trainer `--num-nodes 1` (ramanujan), C = 4-node (subagent).

**Orchestration log (feynman):** 00:21 — the 00:06 boxes (w79ypo3/qe5d22q/q9v7p53)
found STOPPED; operators re-provisioning fresh boxes themselves. PROTOCOL.md
rewritten to v2 (feynman orchestrates, single-writer notebook, per-box queues,
DP-semantics gate on box B). Config fixes: `expB` max_seq_len 262144→**131072**
(CP8@262K ≈300 GiB predicted — boot warmup would OOM); added
`expF-ep32cp8dp4.json` (stretch). Every bench standardized at ~524,288
tokens/step (seq_len × datums) for direct MFU comparability with Aug-7.
MFU method: kept `runs/overnight_20260809_mfu_sweep/mfu.py` (length-dependent FWD(L), reproduces the
old 118.3 GF/token constant at 262K); independent arithmetic audit running →
`runs/overnight_20260809_mfu_sweep/mfu_audit.md`. Correctness rule: only proven loss-neutral env
(ship NCCL + BT_TF32_LM_HEAD=1); every config×L canary-checked, >5e-3 stops.

**MFU audit result (00:35, `runs/overnight_20260809_mfu_sweep/mfu_audit.md`):** mfu.py FWD(L) is
**~4% high, uniformly across L** (indexer params counted in all 78 layers vs
the 21 `indexer_types` full layers; wq_b input dim; FULL_IDX_LAYERS 22→21).
Corrected FWD: 93.9/96.7/102.3/113.6 GF/tok at 32K/64K/131K/262K (script:
97.7/100.6/106.5/118.3). All other dims exact vs HF config. **Convention kept
for tonight** — table stays comparable to every prior LPS-1062 row; multiply
mfu3x/hfu by ×0.96 for absolute numbers. Also noted: 2.5 PF/GPU peak is our
convention (published HGX B300 dense bf16 ≈ 2.25 PF), and mfu3x's 3× slightly
overstates useful FLOPs for LoRA (frozen-base bwd ≈ 1× fwd) — both
convention-level, not errors.

**DP-semantics gate (00:47, ramanujan, code-level @ ef4ea4a8): CLEARED.**
`/forward_backward` datums are **sharded** across DP replicas (contiguous
`ceil(n/dp)` slices, `dp_shard` training_runner.py:79-99; dp_size = pure DP,
CP peers share dp_rank; THD-CP packing requires identical local_real across
the CP group). Total-token TPS accounting in bench_driver2 is correct. Loss =
token-weighted **global** mean (CP-reduce, then one pure-DP all-reduce of
(loss_sum, loss_tokens)) → canaries directly comparable across DP factorings
modulo reduction order.

**Stack resolution (01:00, gibbs):** the trainer runs from the SHARED
checkout = **0e0b65a6 + Aug-7 dirty-tree patches (exact Aug-7 exp06 stack)**;
devbox-up never touches it ('shared clone exists, not pulling'). The
ef4ea4a8 sighting was the per-box clone at /root/trainers. Decision: stay on
0e0b65a6+dirty all night (same-commit anchor reproduction, no fleet-wide
mutation). TF32 grep=1 ✓, kit md5-verified on-box (no CPFS NULs) ✓.
ramanujan re-verifying the DP gate against the shared checkout before
benching.

Queues: A (gibbs, 2-node, expA golden EP16/CP16): anchor-262k-d2 r3,
131k-d4, 65k-d8, 32k-d16, stretch 131k-d2. B (ramanujan, 2-node, expB
EP16/CP8/DP2): DP-gate, then 131k-d4 r3, 65k-d8, 32k-d16, stretch 131k-d2.
C (laplace, 4-node): expD EP32/CP32 (262k-d2 r3, 131k-d4) → expE
EP32/CP16/DP2 (131k-d4 r3, 262k-d2, 65k-d8) → stretch expF EP32/CP8/DP4
(131k-d4). Headline A/B: B-131k-d4 vs A-131k-d4.

All rows: B300, ship NCCL env + BT_TF32_LM_HEAD=1, trainers 0e0b65a6+Aug-7
patches, full recompute, attention-only LoRA r32, ~524,288 tokens/step
unless noted. mfu3x/hfu per mfu.py convention (×0.96 for audited absolute).
[2026-08-09 PM: mfu3x/hfu cells below converted to LoRA-corrected values —
see note at top of file; also note the "attention-only LoRA" shorthand is
wrong per lora_targets.py — targets include dense-MLP/shared-expert/LM-head
adapters, folded into F_lora(r).]

| # | config | tok/s/GPU | step (s) | mfu3x | hfu | max GPU mem (MiB) | loss canary | notes |
|---|---|---:|---:|---:|---:|---|---|---|
| A-anchor-262k | expA EP16/CP16 2×8, 262k×d2 r3 | **630** (587/647/663) | 52.0 | 6.6% | 9.5% | 263,753/224,433 | dloss ≤9e-4 vs Aug-7 exp06 ✓ | **exact reproduction of exp06** (629/52.1s); night's baseline anchored |
| C-D-262k-d2 | expD EP32/CP32 4×8, 262k×d2 r3 | 300 (254/318/346, climbing) | 54.5 | 3.1% | 4.5% | 173,131/150,271 | dloss ≤3e-4 vs Aug-7 ✓ | 2× GPUs, same step wall as golden → latency-bound at fixed 524k/step; buys **−90 GiB/GPU floor** (the enabler for DP variants) |

| A-131k-d4 | expA (262K boot), 131k×d4 r2 | **54** (49/59, systematic) | 609.7 | 0.5% | 0.7% | 274,105/272,597 (pinned at cap) | 131k ref: 12.2908(w)/12.2696/12.2522, gn 0.665/0.636/0.581 | **F1 thrash**: 131K shapes under 262K-warmup pool → reserved to ceiling; see A131 rerun for the config-vs-boot deconfound |
| C-D-131k-d4 | expD EP32/CP32 (262K boot), 131k×d4 r2 | main0 137 / **main1 353 (clean)** | 46.4 (main1) | 3.2%*† | 4.7%*† | 263,611/190,831 (96% cap, spread 72.8 GiB) | 12.2876/12.2697/12.2511 (131k traj.) | F1 reproduced in milder form (EP32 headroom); main1 = usable steady state (*mfu from main1; †published 5.0/6.7 was a transcription slip even under the old convention — 353 t/s ⇒ 4.5/6.0 old; corrected from tps=353) |

| A131-131k-d4 | **expA131** EP16/CP16 2×8 (**131K boot**, full prod warmup), 131k×d4 r3 | **620** (576/644/646; steady ~645) | 52.8 | 5.6% | 8.2% | 201,195/194,611 (74 GiB headroom) | 131k ref (w2): 12.3178/12.3095/12.2947, gn 0.905/0.680/0.664 | **F1 deconfound: 54→620 (11.5×) purely from boot-shape match** — thrash was the 262K-boot artifact; golden@131K healthy, slightly under golden@262K (630) |
| B-131k-d4 | expB EP16/**CP8/DP2** 2×8 (131K boot, BT_SKIP_WARMUP), 131k×d4 r3 | **691** (54.3/43.9/44.0; steady ~745) | 47.4 | 6.3% | 9.1% | 229,371/217,611 | **tight vs A131: dloss ≤4e-4 across all 3 mains ✓** | **HEADLINE: +11% vs golden@131K (620), steady +15% (745 vs 645)** at identical per-rank tokens; CP8 comm halving + ~free LoRA DP grad-AR; dp=2 verified in /status + step-time sharding proof |

| A131-65k-d8 | expA131 EP16/CP16 2×8, 65k×d8 r2 | 632 (626/639) | 51.8 | 5.3% | 7.8% | 234,575/203,091 | 65k ref (w2): 12.2626/12.2434, gn 0.617/0.556 | healthy; golden is ~flat across 65K–262K buffer sizes under matched boots (632/620/630) — TPS length-insensitive, MFU falls with FWD(L) |

| A131-32k-d16 | expA131 EP16/CP16 2×8, 32k×d16 r2 | 641 (639/642) | 51.1 | 5.2% | 7.6% | 234,875/203,091 | 32k ref (w2): 12.2089/12.1938, gn 0.473/0.442 | healthy; **box A queue done** — golden curve flat 620–641 t/s across 32K–262K buffers (matched boots) |

| C-E-131k-d4 | expE EP32/CP16/DP2 4×8 (max 262K cfg, skip-warmup boot), 131k×d4 | **ABORTED** — 41 t/s sustained (398s/step ×2 mains, no recovery) | — | — | — | 272,265 max (99% cap), node spread 97–144 GiB | 12.3172/12.3090, gn 0.886/0.663 (sane) | **F1 refinement needed: thrash WITHOUT boot warmup** (skip-warmup boot; box B healthy on identical bench) → suspect config max_seq_len-scaled allocations or DP2×4-node effect; discriminator = expE131 reboot (max 131K) |

| A131-131k-d2 | expA131 EP16/CP16 2×8, 131k×d2 r2 (**half step: 262,144 tok**) | 649 (651/647) | 25.2 | 5.9% | 8.6% | 234,875/203,091 | dloss −0.14 vs d4 = weight evolution over 10 optim steps on one boot (~0.01/step, gn smooth 0.90→0.39), not drift | **half-step is FREE on golden** (649 vs 645 steady, +0.6%); optim ~0.05s; global-batch flexibility costs nothing on EP16/CP16 |

| C-E131-131k-d4 | **expE131** EP32/CP16/DP2 4×8 (**max 131K cfg**, skip-warmup), 131k×d4 r3 | 395 (356/421/413; steady ~417) | 41.5 | 3.6% | 5.2% | 168,773/149,493 (tight spread) | tight vs A131/B: dloss ≤8e-4 ✓ | **discriminator: healthy after only max_seq_len 262144→131072** → F1 trigger is the config value, not warmup. But 4-node economics poor: 417×32 ≈ 13.3K tok/s aggregate vs B's 745×16 ≈ 11.9K (+12% for 2× hardware) |

| C-E131-65k-d8 | expE131 EP32/CP16/DP2 4×8, 65k×d8 r2 | 320 (259/**419 steady**) | 51.1 | 2.7% | 3.9% | 196,457/157,013 (no thrash) | 12.2627/12.2434, gn 0.62/0.56 (matches A131-65k to ≤1e-4) | expE131 also length-flat (~417–419 steady at 65k & 131k); step-time invariance confirms latency-bound regime on EP32 |

| C-G-131k-d4 | **expG EP16/CP8/DP4 4×8** (max 131K, skip-warmup), 131k×d4 r3 | **665** (572/727/723; **steady ~725**) | 24.6 | 6.1% | 8.8% | 227,707/214,267 (spread 13 GiB, no thrash) | loose (w4 stream): band + gn sane ✓ | **4-node scale-out PROVEN: per-GPU lands in B's range → 32×725 ≈ 23.2K tok/s aggregate ≈ 2.0× the 2-node winner.** Principle: EP16 (a2a locality) + CP8 (min for 131K) + scale by DP; ep16/cp8/dp4 verified in /status |

| B-65k-d8 | expB EP16/CP8/DP2 2×8 (fresh boot), 65k×d8 r2 | **116** (sustained thrash) | 283.1 | 1.0% | 1.4% | 274,057 (at cap; 176↔274 GiB churn) | 12.3081/12.2986, gn 0.744/0.725 sane | **F5: pathological point, reproduced on 2 independent pools** (2 docs/131K-partition) |
| B-32k-d16 | expB EP16/CP8/DP2 2×8, 32k×d16 r2 | 501 (379/**738 steady**) | 65.4 | 4.1% | 5.9% | 273,441 transient, settles | 12.2713/12.2561, gn 0.614/0.597 | 4 docs/partition RECOVERS → F5 non-monotonic, 65k-specific; steady ≈ 131k's 745 |
| C-G-65k-d8 | expG EP16/CP8/DP4 4×8, 65k×d8 r2 | **108** (96/123, sustained) | 151.5 | 0.9% | 1.3% | 274,099 (99.7% cap; churn median 113 GiB/GPU) | sane | **F5 confirmed fleet-wide: CP8+multi-doc, NOT DP2/2-node-specific** (same boot ran 131k-d4 clean at 665) |
| B-16k-d32 | expB EP16/CP8/DP2 2×8, 16k×d32 r2 (**customer-shape: 8 docs/131K partition**) | **734** (44.3/44.9s, both steady) | 44.6 | 5.8% | 8.5% | 271,481/249,601 | 12.2214/12.2046, gn 0.493/0.459 sane | **customer's dominant regime runs FULL SPEED on CP8/DP2** — F5's 65k point is singular, not the regime |

| C-G-16k-d32 | expG EP16/CP8/DP4 4×8, 16k×d32 r2 (customer shape) | 339 (252/**517**) | 48.4 | 2.7% | 3.9% | 273,881 (99.6% cap — residual) | 12.2094/12.1934, gn 0.47/0.44 sane | **contaminated: ran on the pool the 65k thrash left at cap**; main1 517 = recovering, fresh-pool likely ≈B's 734 but unproven on this arm (accepted caveat, no re-boot at 05:30) |

| B-custmix | expB EP16/CP8/DP2 2×8, 20 heterogeneous datums (10,240–63,488 tok, customer histogram, 530,432 tok/step) | **DEADLOCK (F2)** — warmup hung >12 min | — | — | — | ~273 GiB, both nodes NCCL-spin | n/a | **F2 empirically confirmed on customer-realistic input**: replica0 got 294,912 tok→3 partitions, replica1 235,520→2 → replica0 stuck in partition-3 EP dispatch (token_dispatcher.py:959), replica1 already past the loop (packing.py:415). Forensics → runs/overnight_20260809_mfu_sweep/results/B_custmix_f2/ |

### Findings so far (02:0x)

**F1 — allocator thrash when bench shapes ≪ boot max_seq_len (box A) —
CONFIRMED via A131 deconfound (54 → 620 tok/s/GPU, 11.5×, only change =
max_seq_len 262144→131072 at boot). Customer rule: set max_seq_len = actual
packed-buffer size.**
A-131k-d4 under expA's 262K-warmup boot: main0 fb=662.9s = **49 tok/s/GPU**
(~13× slower than anchor; warmup 1×131K alone was fine at 16.2s). Symptoms:
reserved touched 272,279 MiB (≈ceiling), per-GPU spread oscillating
23→128→75 GiB, 1 rank at 0% / 15 spinning (NCCL waits). Read: 262K-shaped
reserved pool (~264 GiB) + new 131K-shaped allocations → reserved grows to
ceiling → context-sweep flush-and-realloc regime. Customer-relevant rule if
confirmed by the expA131 rerun: **set max_seq_len = actual packed-buffer
size** (don't run buffers far below max_seq_len near the memory ceiling).
Redirect: A skips 65k/32k under this boot; reboots `expA131` (golden at
max_seq_len=131072, prod-like full warmup) → A131-131k-d4 r3, 65k-d8,
32k-d16.

**F2 status upgrade (05:4x): EMPIRICALLY CONFIRMED on customer-realistic
input** — B-custmix (heterogeneous lengths from the customer histogram)
deadlocked mid-warmup exactly as the code reading predicted (3 vs 2
partitions across replicas). DP>1 is unshippable on real THD data at this
stack rev; every DP>1 number in this table is contingent on the
partition-count-equalization fix.

**F2 — DP2 boot DEADLOCK in THD-CP + EP (box B), plausibly novel.** expB
(EP16/CP8/DP2) boot hung ~35 min in warmup pass-2: node-0 ranks in MoE
`dispatch_preprocess` all_gather over tp_ep_group (16 ranks — **EP group
spans both DP replicas**), node-1 ranks idle in DP grad-sync wait; GPUs 0%,
no NCCL abort (warmup lifts pg timeout to 120 min). Mechanism (ramanujan,
code-level): THD-CP packs **data-dependent per-replica partition counts**
(controller comment assumes no cross-DP collective in the per-partition
loop), but per-partition MoE EP collectives ARE cross-DP → count mismatch
deadlocks the first MoE layer. **CONFIRMED (02:2x, py-spy line-level):**
pass-1 startup warmup (megatron_controller.py:2411) sends **1 datum** →
dp_rank1's shard is empty (`data[1:1]`) → 0 partitions → skips forward →
dp_rank0 blocks in EP16 all_gather, dp_rank1 blocks in DP grad-reduce.
Deterministic at DP>1; the :2471 'no phantom padding required' invariant is
violated whenever EP spans DP replicas and partition counts differ. Prod
implication: DP>1 THD-CP GLM configs deadlock at boot 100%, and on real
(uneven) data any step can deadlock — ticket + real fix needed (all-reduce
max partition count across DP + phantom partitions). Tonight's mitigation
(no code change; patch plan considered then cancelled): **`BT_SKIP_WARMUP=1`**
(megatron_controller.py:2377 — gates BOTH warmup passes; note
`BT_SKIP_FULL_WARMUP` alone is pass-2-only and would still deadlock) on all
DP>1 boots + bench driver `--warmup-datums <DP>` (driver warmup absorbs
kernel compile), datum counts multiples of DP at equal lengths. Shared
checkout stays pristine. Forensics →
`runs/overnight_20260809_mfu_sweep/results/B_dp2_boot_deadlock/`.

**F5 — multi-doc-partition allocator thrash on CP8 meshes (box B; GATES the
recommendation).** B-65k-d8 (2 docs per 131K partition, CP8/DP2) thrashes on
a FRESH skip-warmup pool: 116 t/s, pool at cap, 176↔274 GiB churn within
seconds (deconfounded from the earlier orphaned-op run — deterministic
losses match). Same per-rank tokens/step as the clean B-131k-d4 (1
doc/partition, stable 229 GiB); box A's CP16/DP1 65k-d8 (2 docs/partition)
was clean at 234.6 GiB → confound CP8-vs-DP2 under test via C-G-65k-d8
(CP8/DP4). **Customer impact: real data is THD-packed ~4–9 docs per 131K
buffer — the single-doc-per-datum headline benches under-model exactly this.**
Scaling-law probe result: **non-monotonic** — B-32k-d16 (4 docs/partition)
settles to **738 steady** after one transient window (main0 379→main1 738;
near-cap transient only), while 65k thrashed sustained on TWO independent
pools (as first-bench-on-fresh-boot AND as later bench) → 65k@CP8/DP2 is a
shape-specific allocator pathology (allocation-size/segment-bin resonance),
not a docs/partition law. Customer-shape probes: B-16k-d32 and C-G-16k-d32
(8 docs/partition): **B-16k-d32 = 734 steady → the customer's dominant
regime is clean; 65k is a singular point.** Mitigation dead-end:
`expandable_segments:True` is ALREADY the default (launch.sh:62, verified in
live trainer env) — the pathology exists despite it. Root cause open →
ticket: in-trainer `memory._record_memory_history` snapshot during a 65k
thrash. Practical customer guidance: corner case; a packer rule avoiding
~2×65k-homogeneous partitions sidesteps it entirely. Final probe:
B-custmix (heterogeneous lengths drawn from the customer histogram) — also
doubles as the empirical F2 test on realistic data (per-replica partition
counts may differ → mid-step deadlock).

**F4 — shared `.devbox_up/trainer_srun.log` is cross-box contaminated**:
every box's start_trainer.sh truncates/rewrites it (the 'pass-2 262144'
banner B saw was box A's trainer). Don't trust it for multi-box nights; use
per-rank/per-job logs.

**F3 — shared env.sh hazard: `BT_WARMUP_FULL_TOKENS=262144`** was present in
the shared env.sh early tonight (another provisioner's write) but **transient**
— gibbs's 00:39 provision rewrote env.sh without it, and B's relaunched
trainer env was clean (`/proc` verified). The '262K pass-2' banner B saw was
F4 log contamination, not the leak. Standing rule kept: log
`env | grep BT_` from `/proc/<pid>/environ` at every launch — the shared
env.sh is mutable by any box's provisioner at any time.

---

### SYNTHESIS (feynman, 2026-08-09 ~06:00) — customer recommendation

**Question:** fastest way to train GLM-5.2 (CP THD) on ~608K samples
(~15–17B tok; 70% ≤32K / 28.5% 32–64K / 1.4% 64–131K, max 131K).

**Steady tok/s/GPU at ~524K tok/step, all with ship NCCL env + TF32 head
(both re-verified loss-neutral tonight, drift ≤9e-4):**

| mesh | nodes | 131K bufs | 32K bufs | 16K bufs (customer shape) | DP-safe today? |
|---|---|---:|---:|---:|---|
| EP16/CP16 (golden) | 2 | **645** | 641 | — | **YES (DP1)** |
| EP16/CP8/DP2 | 2 | **745** | 738 | 734 | no (F2) |
| EP32/CP32 | 4 | 353 | — | — | yes, but pointless (≈2-node agg) |
| EP32/CP16/DP2 | 4 | 417–419 | — | — | no (F2) |
| EP16/CP8/DP4 | 4 | **725** | — | (517 contaminated; ≈734 expected) | no (F2) |

**Recommendation, in order:**
1. **Ship env + TF32 head unconditionally** (the Aug-7 +51%; correctness
   re-verified across 4 boots and 3 meshes tonight).
2. **Pack to 131K buffers and set max_seq_len = 131,072 exactly** (F1: a
   mismatched max_seq_len cost 11.5× in the worst case; confirmed twice,
   trigger is the config value itself, warmup or not).
3. **Today, without code changes: golden EP16/CP16 on 2 nodes** — ~645
   steady tok/s/GPU ≈ 10.3K agg → **~17–19 days/epoch**. Throughput is flat
   in buffer size and step size (half-step free), so global-batch choice is
   unconstrained. There is no useful DP1 path to 4 nodes (EP32/CP32 adds
   nothing over 2-node).
4. **Fix F2 (partition-count equalization across DP — all-reduce max count +
   phantom partitions), then EP16/CP8/DP4 on 4 nodes** — ~725 steady ≈ 23.2K
   agg → **~7.5–8.5 days/epoch (2.2× golden)**. 2-node CP8/DP2 (+15%) is the
   intermediate. F2 is small, localized, fully root-caused (code + py-spy on
   two shapes incl. customer-realistic input) — highest-leverage engineering
   item by far.
5. Post-fix caveat **F5**: CP8 meshes have a singular allocator pathology at
   ~2×65K-homogeneous partitions (sustained ~10× thrash; reproduced on 3
   pools, 2 meshes, DP2+DP4; NOT fixed by expandable_segments, which was on
   the whole time). Customer's dominant shapes are clean (734 @ 8×16K/
   partition; 738 @ 32K). A packer rule avoiding ~65K-homogeneous partitions
   sidesteps it; root-cause ticket below.

**MFU method:** mfu.py convention kept all night (length-dependent FWD(L);
comparable with every prior LPS-1062 row). Independent audit
(`runs/overnight_20260809_mfu_sweep/mfu_audit.md`): absolute values ~4% high (×0.96 to correct);
2.5 PF/GPU peak is a convention (datasheet dense bf16 ≈ 2.25 PF). Raw
tok/s/GPU is method-independent.
[2026-08-09 PM: superseded — the ×0.96 covered only the audit constants;
the LoRA pass-structure correction is larger (published mfu3x ~×1.30–1.39
high overall). All cells above now LoRA-corrected; steady refs: exp06
660 ⇒ mfu3x 6.9/hfu 9.9; B 745 ⇒ 6.8/9.8; C-G 725 ⇒ 6.6/9.6. See
`runs/overnight_20260809_mfu_sweep/mfu_lora_correction.md`.]

**Tickets to file:**
1. **F2 fix** — DP>1 THD-CP deadlock: per-replica partition counts must be
   equalized (all-reduce max + phantom partitions); boot warmup pass-1
   (1 datum) deadlocks 100% at DP>1; BT_SKIP_WARMUP=1 is the stopgap.
   Evidence: `runs/overnight_20260809_mfu_sweep/results/B_dp2_boot_deadlock/`, `B_custmix_f2/`.
2. **F5 root-cause** — CP8 ~65K-homogeneous allocator thrash despite
   expandable_segments (launch.sh default): capture
   `memory._record_memory_history` snapshot mid-thrash; consider packer
   avoidance rule meanwhile. Evidence: 3 reproductions + mem-poller csvs.
3. F1 hardening — warn (or auto-set) when max_seq_len ≫ observed packed
   buffer size; the failure mode is silent 10×.
4. (carried from Aug-7) MoE dispatcher host syncs (26.7K nonzero/step) —
   still the deepest throughput lever; and default the ship NCCL env in
   prod ali-B300 trainer pods.

**Ops notes:** devbox-up defaults to B200/hyd (2 near-miss provisions —
pass `b300`); one silent mid-bootstrap death (papercut filed by laplace);
shared-FS hazards again (stale config staged → would-have-OOM'd, caught by
md5 protocol; trainer_srun.log cross-box truncation = F4; env.sh mutable by
any provisioner). Never kill the bench driver mid-op (orphaned server-side
op poisons the next optim_step token accounting — restart the trainer).
Boxes: A=wxlgezw, B=wlxm57w, C=wpr25e3 — all torn down by ~06:00, verified
STOPPED. 17 result jsons + 4 forensics/mem bundles in `runs/overnight_20260809_mfu_sweep/results/`.

---

## 2026-08-07 (original session below)

**Date:** 2026-08-07 (overnight session) · **Goal:** maximize GLM-5.2 256k training throughput on 2×8 B300 without OOM or correctness regressions.
[2026-08-09 PM: mfu3x/hfu cells in this section's table converted to
LoRA-corrected values — see note at top of file.]

Baseline (2026-08-06, devbox q480z53, trainers_main @ 0e0b65a6, golden config TP1/PP1/EP16/CP16, full recompute, alltoall dispatcher):
**446 tok/s/GPU · 73.5 s / 524k-token step · MFU(4×fwd, w/ indexer, 2.5PF) 8.4% [LoRA-corrected: 6.7%].**
Bottlenecks (from `runs/overnight_20260807_baseline_shipconfig/glm52-b300-s256k/REPORT.md`):
NCCL 65% of step (EP a2a SendRecv 59%), 26.7k `aten::nonzero` GPU syncs/step, 8 FP32 SIMT vocab GEMMs (1.85 s), allocator reserved 260/275 GB.

## Measurement protocol (apples-to-apples)

- `runs/overnight_20260807_baseline_shipconfig/bench_driver.py` (this folder): synthetic random tokens seed `0xB300`, **identical rng consumption order to the baseline profile run**: 1×262k-token warmup window, then 2 main windows of 2×262k datums (524,288 tokens/step). Metrics = mean of the 2 main windows.
- Report per iteration: **tok/s/GPU, step time (s), MFU** (two conventions: `mfu3x` = model FLOPs 3×fwd — rewards removing recompute; `hfu` = hardware passes actually run), plus loss/grad_norm per window as the **correctness canary** — must stay ≈ baseline (12.356/0.940 warmup, 12.339/0.941, 12.310/0.693) modulo small reduction-order drift. Fwd FLOPs/token = 118.3 GF (84.3 matmul + 10.5 DSA + 23.6 indexer), see `runs/overnight_20260807_baseline_shipconfig/glm52-b300-s256k/mfu_calc.py`.
- No kineto/memory profiler in timed runs (profilers only for diagnosis, marked as such).
- **Per-GPU max memory**: every bench runs under `runs/overnight_20260807_baseline_shipconfig/run_bench.sh`, which starts `tools/poll_gpu_mem.sh` (nvidia-smi, 2 s cadence) on every node via srun and folds per-GPU max used MiB into the result json (`aggregates.per_gpu_max_used_mib`). nvidia-smi reports allocator-reserved memory, which is the OOM-relevant number; 2 s sampling can miss sub-second transients.
- Oversized artifacts (traces, pickles) → `~/perf_profiles/lps-1062/opt-night/`; everything else here.

## Lever map (from code reading, trainers @ 5191b710)

| Lever | Config | Attacks | Risk |
|---|---|---|---|
| DeepEP fused dispatch | `moe_token_dispatcher: "flex"` (wheel vendored, cu13) | a2a 59% + nonzero syncs | RoCE/NVSHMEM bring-up; fallback envs: `NVSHMEM_IB_ENABLE_IBGDA=1`, GID 3, `NVSHMEM_HCA_LIST` |
| EP a2a↔compute overlap | `comm_overlap.overlap_moe_expert_parallel_comm: true` (+ optional `delay_wgrad_compute`) | hide a2a latency | **requires `recompute_granularity != "full"`** (bridge validator, comm_overlap.py:493); THD-CP loop forbids only `overlap_grad_reduce` |
| Selective recompute w/ module list | `recompute: {granularity: "selective", modules: [...]}`; allowed: core_attn, moe_act, layernorm, mla_up_proj, mlp, moe, shared_experts | recompute pass replays fwd a2a under "full"; module list can approximate full-recompute memory | OOM at 256k if saved set too big |
| CE/vocab FP32 GEMMs | code path TBD (8×115 ms, grid [1210,32], bias_relu epilogue) | 2.4% of step | needs patch; loss parity check |
| CUDA_DEVICE_MAX_CONNECTIONS | launch env (=1 today) | serialized launch queue kills overlap | was the *mask* for LPS-1003 DSA stream race (fixed in PR#875 wheel +dsatopk5); only touch with loss parity validation |
| moe_expert_capacity pad | fixed-shape routing | nonzero syncs | pads inflate a2a volume — likely net loss; only if syncs persist after flex |

Notes: EP8 (intra-node experts) infeasible — +91 GB/rank expert weights at bf16. `moe_permute_fusion` already on, `moe_grouped_gemm` on, aux-loss off, `moe_router_fusion` hardcoded off.

## Experiment queue (revise as results come in)

1. `exp00-baseline` — golden config rerun on fresh box + current main (commit moved 0e0b65a6 → 5191b710): re-anchor.
2. `exp01-flex` — dispatcher flex, all else baseline.
3. `exp02-selective` — recompute selective `["core_attn","moe","layernorm","mla_up_proj"]` (memory probe; a2a still replayed via "moe").
4. `exp03-overlap` — flex + selective + `overlap_moe_expert_parallel_comm` (+ delay_wgrad if needed).
5. `exp04+` — CE path fix, recompute-list tuning (drop `core_attn` if TE fused attn makes it unnecessary — mcore warns it may be), NCCL tunables, env experiments. Data-driven.

## Iterations

| # | config | tok/s/GPU | step (s) | mfu3x | hfu | max GPU mem (MiB) | loss canary | notes |
|---|---|---:|---:|---:|---:|---:|---|---|
| exp00 | golden baseline (alltoall, full recompute) | **416.5** | 78.7 (82.1/75.2) | 4.4% | 6.3% | (poller missed — srun queued; see exp00b) | 12.3578/12.3401/12.3106 ≈ baseline ✓ | **tonight's anchor.** ~7% below q480z53's 446 — box/fabric variance; windows spread ±5% |
| exp00b | same, hot trainer, 1 window (mem probe) | 445 | 73.6 | 4.7% | 6.7% | **263,009 / 223,889** (max/min) | informational (3 optim steps applied) | worst-GPU headroom only ~12 GiB |
| exp01 | flex/DeepEP | — | — | — | — | — | — | **infeasible on fabric** (see below) |
| exp03 | EP a2a overlap (+selective recompute) | — | — | — | — | — | — | **infeasible: validator × memory** (see below) |
| exp04 | TF32 CE head (v1 patch) | 417 | 78.7 | 4.4% | 6.3% | 263,609/223,649 | drift ≤5e-4 ✓ | no-op — head isn't LoRA-wrapped (attn-only LoRA), unpatched branch ran |
| exp04b | TF32 CE head (v2, verified active) | **431** | 76.0 (80.1/71.9) | 4.5% | 6.5% | 263,989/223,729 | drift ≤1.4e-3 ✓ | **+3.5% vs anchor**, both windows faster; keep on |
| exp05a | + `NCCL_IB_QPS_PER_CONNECTION=4` `NCCL_IB_SPLIT_DATA_ON_QPS=1` | **464** | 70.6 (72.8/68.3) | 4.9% | 7.0% | 262,849/223,549 | drift ≤1.5e-3 ✓ | **+11.4% vs anchor.** LAG bonds hash flows per-QP — multiple QPs spread each peer connection across bond slaves. GDR confirmed enabled; 6×400 Gb bonds/node; NCCL 2.28.9 |
| exp05b | + QPS=8, `NCCL_NCHANNELS_PER_NET_PEER=4` | **559** | 58.6 (63.3/53.9) | 5.8% | 8.4% | 263,297/224,237 | drift ≤1.4e-3 (mains) ✓ | **+34% vs anchor** — fabric lever is rich; probing ceiling |
| exp05c | + `NCCL_NCHANNELS_PER_NET_PEER=8` | **583** | 56.2 (61.5/50.8) | 6.1% | 8.8% | 262,553/224,513 | drift ≤8e-4 ✓ | **+40% vs anchor**; channel scaling flattening (+4% for 4→8) |
| exp05d | + chan=16, `NCCL_MAX_NCHANNELS=64` `NCCL_MIN_NCHANNELS=32` | **597** | 54.9 (61.6/48.3) | 6.2% | 9.0% | **266,123**/226,423 | drift ≤3.1e-3 (w0), ≤1.4e-3 mains ✓ | **+43%**, but +3.5 GiB peak (NCCL buffers) → 8.9 GiB headroom; escalation stopped, kineto capture on this config |
| exp05e | exp05d but `NCCL_IB_SPLIT_DATA_ON_QPS=0` | 603 | 54.4 (59.3/49.4) | 6.3% | 9.1% | 266,883/226,683 | drift ≤2e-3 ✓ | tied with exp05d — split knob immaterial at these sizes |
| **exp06** | **SHIP: exp05c env, 3-window soak** | **629** (57.0/49.6/49.6 → steady ~660) | **52.1** | **6.6%** | 9.4% | 263,813/224,333 | drift ≤2e-3 ✓ | **+51% vs anchor (+58% steady-state)**; headroom-first pick: 16-chan buys ~2.5% but costs ~3.5 GiB of OOM margin |

### exp01 — flex/DeepEP dispatcher (attempts, 2026-08-07 ~10:40–11:30 PDT)

- **Attempt 1** (no NVSHMEM env): crash at NVSHMEM IBGDA init — `mlx5dv_devx_obj_modify INIT2RTR_QP syndrome 1ffea3`, `ibgda_rc_init2rtr failed` on every RC; also `cudaHostRegister IoMemory error=800` + `ibgda_alloc_and_map_qp_uar` GPU-handler failures. Root cause: NVSHMEM defaulted to GID 0 (RoCEv1 link-local); fabric needs GID 3 (RoCEv2, routable — same as NCCL_IB_GID_INDEX=3). Fabric GID table confirmed via sysfs.
- **Attempt 2** (`NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER=cpu NVSHMEM_IB_GID_INDEX=3 NVSHMEM_HCA_LIST=mlx5_bond_0:1,mlx5_bond_1:1`): QPs connect, boot reaches first MoE forward in warmup, then **`DeepEP error: timeout (dispatch CPU)` in `internode_dispatch`** + illegal-memory-access cascade on peer ranks. Transport connects but the IBGDA data path over the LAG-bonded RoCE devices (`mlx5_bond_*`) doesn't move data.
- **Attempt 3** (`NVSHMEM_IBGDA_NIC_HANDLER=gpu` + `NVSHMEM_DEBUG=INFO`): same dispatch timeouts (17 DeepEP errors). NVSHMEM logs `IBGDA: device used mlx5_bond_0, data_direct support: 0`.

**Verdict: flex/DeepEP is infeasible on ali B300 (LAG-bonded RoCE) tonight.** QPs connect with GID 3 but IBGDA never moves data over the bond devices, both CPU and GPU NIC handlers. This is an infra/qualification gap (DeepEP+NVSHMEM IBGDA over RDMA LAG bonds), not a trainer-config problem. Follow-up ticket material: qualify DeepEP on the ali fabric or expose non-bonded physical ports to NVSHMEM.

Fallback prepared: `exp03a-overlap-alltoall.json` (selective recompute + `overlap_moe_expert_parallel_comm` — the bridge validator accepts dispatcher `alltoall` too), `exp03b` adds `delay_wgrad_compute`.

### exp03 — EP a2a overlap: structurally infeasible at 256k (2026-08-07 ~11:45)

Two boots, two mcore validator walls, then arithmetic kills it:
1. `disable moe_shared_expert_overlap when enabling overlap_moe_expert_parallel_comm` — GLM provider defaults it on; only the flex branch cleared it. **Patched** on-box (megatron_controller.py `_configure_moe_provider`: clear it under the overlap flag too) — keep for PR regardless.
2. `disable moe in recompute_modules when enabling overlap_moe_expert_parallel_comm` — the overlap schedule can't run under a checkpointed MoE replay. But *not* recomputing MoE means saving the expanded-token activations: ~262k×topk8/EP16 rows × 6144 h × 2 B ≈ 1.6 GB dispatch output + ~2.7 GB expert GEMM in/out per layer per rank ⇒ ~4-6 GB × 75 layers ≈ **300+ GB/rank** vs a 12 GiB worst-GPU headroom. Same arithmetic kills every selective-recompute variant that leaves MoE (or even just layernorms, ~31 GB) resident.

**Verdict: EP-overlap and all lighter-recompute configs are memory-infeasible at 256k on 275 GB parts with EP16/CP16. Full recompute stands.** Re-ranked queue: TF32 CE head (exp04), NCCL alltoall transport tuning (exp05).

### Diagnostic kineto trace of exp05d config (2026-08-07 ~21:10Z)

One profiled 524k-token step (48.8 s wall), rank 0; trace at `~/perf_profiles/lps-1062/opt-night/exp05d.pt.trace.json` (1.05 GB). vs baseline trace:
- **SendRecv 24.5 s (50% of wall, was 44.9 s/59%)**; per-call p50 17.8 ms (was 34 ms), p90 32.6 ms — QP/channel spreading ~halved a2a time. AllGather 0.93 s, ReduceScatter 0.61 s, AllReduce 0.01 s.
- **`aten::nonzero` unchanged: 26,684 calls, 20.6 s CPU** (+29k `cudaStreamSynchronize` 20.0 s, 33.6k `cudaMemcpyAsync` 15.0 s). With comm halved, the MoE-dispatch GPU-sync round-trips are now co-dominant — the reason step time didn't fall in proportion to per-call a2a. **Top remaining lever; needs mcore dispatcher surgery (device-side split bookkeeping / batched D2H), out of scope tonight — follow-up ticket.**
- **FP32 SIMT vocab GEMMs: gone** (TF32 head verified in trace; nvjet tensor-core GEMMs only).
- Compute kernels ≈ 19.6 s busy; DSA bwd 3.0 s, indexer fwd 1.65 s + topk 0.44 s, sparse fwd 1.16 s, GEMM ~3.7 s; idle 6.6%.

### Ops incident — phantom trainer start (2026-08-07 20:21Z)

A trainer srun using MY `lps1062/ctl/run_trainer_node.sh` was submitted at 20:21:45Z by an unknown caller (not my launch tasks; coincided with a shared-FS `env.sh` regeneration at 20:19:26Z by another session's 4-node provisioning). Its env (NCCL knobs) would have been whatever the caller had — untrusted. Mitigations: caller-audit logging + `BT_LPS1062_LAUNCH=1` guard added to `ctl/start_trainer.sh`; job killed, exp05d relaunched deliberately. Result integrity: exp05b/c scored 559/583 vs 464 for correct-env exp05a and ~430 for no-NCCL-env — a default-env trainer can't produce those numbers, so their env was intact (pending sruns keep dispatch-time env). No benched result came from the phantom trainer.

### exp00 — baseline re-anchor (2026-08-07 ~01:45)

Box tj-qzlr0o3 inherited the baseline session's shared-FS state: trainers_main @ 0e0b65a6 + LPS-1003 full-footprint-warmup patch, fabric-aware run_trainer_node.sh, GLM-5.2-FP8 HF cache. Trainer boot ~13 min. Loss canaries match the q480z53 baseline to ≤2e-3 → correctness anchor holds. Rank-0 reserved peak 260.2 GB (matches baseline 260.2). Mem-poller srun queued behind the trainer job (fresh srun ≠ --jobid attach) — fixed in runs/overnight_20260807_baseline_shipconfig/run_bench.sh by attaching to the devbox_trainer allocation; exp00b-memprobe (warmup + 1 main window on the hot trainer) captures per-GPU peaks for the baseline config.
