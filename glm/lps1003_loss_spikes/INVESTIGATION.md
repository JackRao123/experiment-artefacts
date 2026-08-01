# LPS-1003 Issue 2 investigation log (session 2026-07-29/30)

Devbox: tj-3y0n54q (2×8 B300 ali, ex-prod GLM nodes). Stack: trainers_main @
0e0b65a (= patched prod image trainer-cuda13-sm103-0e0b65a), server venv cu130,
cudnn-frontend 1.26.0+dsatopk1. Trainer: golden GLM-5.2-FP8 B300 leaf
(tp1/pp1/cp16/ep16, maxlen 262144, flash, lora r32 a32, seed 1234), weights HF
snapshot ba978f7d (bit-identical to prod /b10/loaded_weights: config.json,
index.json, shard-1 sha256 all match — verified vs live pod 232l69w).

## Established (all verified, chronological)

1. **Batch reconstruction is bit-exact.** Batches 0-5 aggregates
   (n/total/min/max) match prod ClickHouse `submit: op=forward_backward` lines
   for BOTH 4w5y6r3 and 8w6y12q exactly. Same shuffle, same data file.
2. **Phase A (frozen base weights, 22 prod batches ×3 identical /forward):**
   - All batches in 0.72–0.78 band; bump-labeled batches NOT harder than quiet.
   - Batch-level cross-repeat wobble ≤ 0.011.
   - Token-level wobble is discrete and large: ~47% of supervised tokens move
     >0.01 nats between identical runs; ~250 tokens/batch move >1 nat; max 5.5
     nats. Same structure in quiet and bump batches. (Discrete path flips —
     DSA top-k and/or MoE routing under FP8/atomics noise; amplitude too small
     and symmetric at batch level to explain prod bumps alone.)
3. **Phase F arm 0 (real training replay, steps 0-31, prod recipe):** clean.
   Quiet band 0.425–0.516, NO bumps at any prod bump step (12-14/16-17/23/
   26-29). Arm 1 (identical fresh boot) tracking arm 0 to ~2e-4 mid-run.
4. **Prod bumps are real server-side** (8w6y12q, no crashes): datum_mean =
   train_mean_nll × num_loss_tokens / 32 reproduces the W&B series exactly
   (formula verified end-to-end on our own run: 0.003037×8052/32 = 0.7642 =
   client value). Bumps at 1-indexed steps 13-15/17-18/24/27-28/30 (0.84–1.19)
   vs quiet 0.45–0.81. No foreign ops, no re-inits, no retries in window —
   bumps occur inside ordinary 32-datum ops.
5. **Prod starts perturbed; devbox does not.** Prod batch-0 forward (step 0,
   LoRA B=0 → must equal base forward) = 1.4617 (4w5y6r3) / 1.5335 (8w6y12q)
   server-side AND client-side. Devbox same op = 0.7642 / 0.7645 (two fresh
   boots) and 0.72–0.78 across 22 batches. Prod steps 1-5 elevated 1.5–2.0,
   settling to the same 0.45-0.55 quiet band as devbox by ~step 7.
6. **Ruled out for the prod anomaly:** hard data (2); recipe/data/stack replay
   (3); client-metric artifact (4); foreign/injected ops (4); weights mismatch
   (hashes); moe_router_dtype config staleness (GLM5 bridge hardcodes
   provider.moe_router_dtype="fp32"); moe_expert_capacity injection (prod
   runtime config.json read from live pod: null); dispatcher-only layer (it's
   the standard in-process op serializer, devbox runs it too); numerics env
   deltas (pod env clean except PYTORCH_CUDA_ALLOC_CONF=expandable_segments,
   NVTE_CUDA_ARCHS=103a); warmup grad leak (code scraps grads + tracker, no
   optim in warmup).
7. **Regression archaeology:** GLM-on-B300 training exists only since 07-23
   (no clean B300 baseline ever). Mudith's "clean run" = full-FT on H100 via
   custom script (not this stack; 0.59% spike rate; his H100 LoRA attempts all
   crashed early). At the first spiky tag B200/B300 golden configs were
   identical except image → cuda13/sm103 toolchain bring-up (cuBLAS 13.6.0.2,
   custom TE wheel, cutlass-dsl cu13, FA4 b16, 07-22/23) was the whole
   B300-only delta; PR #722 (07-20) fp32 GLM output head + dummy-loss rewrite
   is the only GLM-only forward change in window.

## Phase F complete (arm0 + arm1)

Two byte-identical fresh-boot training replays (steps 0-31): batch-NLL tracks
within 0.015 throughout (growing slowly from ~0.003), max per-datum divergence
0.088, ZERO datums >0.3, zero bumps in either arm. Training-amplified
nondeterminism is real but ~30x too small for prod bumps.

## Corrected comparator matrix (4q9ex6w archaeology)

- 4q9ex6w = Mudith's clean completed run: SAME OE-grader recipe/data, SAME
  trainers-server stack, LoRA r32 — on B200 ×32 (hyd), driven from workstation
  wlemv13 ("full50k" = dataset size, not full-FT). 1529 steps, 0.59% spikes.
- BUT: deployment e3m97kq lineage = 6wg87jq (cold boot, died step 18 on a B200
  DSA IMA — the IMA family is NOT B300-only) → 4w5jk73 (fresh reinit, 58
  steps) → 4q9ex6w (warm continuation).
- 4w5jk73 (fresh-init, B200): first-10 datum-means oscillate 1.81/0.62/0.58/
  1.41/0.50/1.28/0.50/0.47/0.49/1.28 — elevated-start ratio 2.09x. SAME
  signature as cold B300 sessions, sparser.
- 4q9ex6w (warm): 9 spikes at steps 4,28,49,56,71,96,109,249,883 — 8 of 9
  inside first ~110 steps.
- Window-matched spike rates: B200 early ≈ 7%, B300 early ≈ 11-17% — the
  "B300 vs B200" gap mostly reflects that the B300 runs never left the early
  window (died/stopped at 55-70 steps). Hardware amplitude difference is
  secondary, not primary.

## Current hypothesis

Fresh-adapter PLATFORM sessions (any hardware) start with a perturbed
forward state that decays over ~50-150 steps and re-excites on some batches
(the "bumps"). Devbox sessions with identical code/config/weights/data/seed
do not have it (clean ×3 boots). Since LoRA B=0 makes the adapter output
exactly zero at step 0, the perturbation is NOT in the adapter weights — the
step-0 forward itself differs. Remaining candidate state: FP8/TE activation
scaling or other module-level mutable state seeded differently at platform
boot (BT_WARMUP_SEQ=128 vs devbox 64), allocator env, or the
init_trainer_server path. Also ruled out mid-run: client retries (zero
"transient step error" lines in nspvxlhu client log), foreign ops, checkpoint
ops.

## Open question (the crux)

Why does prod's step-0 forward differ 2× from devbox on identical
weights/code/config/data/seed? Remaining prod-vs-devbox boot deltas:
  a. BT_WARMUP_SEQ=128 (prod) vs 64 (devbox default)
  b. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (prod only)
  c. /init_trainer_server op fired after READY (prod; hit "skip rebuild" path)
  d. per-node hardware (weak: two different prod pairs gave 1.46/1.53)

## Prod-mimic boot experiment (04:11) — CLEAN

Fresh boot with BT_WARMUP_SEQ=128 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments
(prod pod env) + POST /init_trainer_server lora_rank=32 (done; skip path like
prod) → /forward batch 0 ×3 = 0.7622/0.7672/0.7631. Still clean. ALL
boot-surface deltas exonerated: warmup seq, allocator, init op, dispatcher,
weights (hash-verified), config (read from live prod pod), env, seed, data
(bit-verified), client metric (formula-verified both sides), retries (none),
FP8 state (compute is bf16 — bridge dequantizes on load), adapter init (B=0
⇒ exactly zero at step 0). Five devbox boots agree at 0.762-0.765.

Conclusion: the prod anomaly is not reproducible through the trainer HTTP
surface with matched inputs. The perturbation arises only in real platform
sessions (both B200 and B300), decays over ~50-150 steps, re-excites
per-op probabilistically (can hit the very first forward: 4w5jk73 step-1 =
1.81 on a fresh in-process LoRA rebuild). Next discriminator requires
observing a live platform session (see below).

## SMOKING GUN (live session 2329m1w / trainer nwxgj6q, ~04:55-05:10)

Launched a real loops session (Jack-approved) with Mudith's exact recipe.
Signature reproduced immediately: step0=1.4451, step1=0.657, step2=1.4906
(clean refs: 0.764/0.597/0.598).

In-pod probe (batch-0 /forward payload direct to localhost:8000, bypassing
gateway), two reps ~3 min apart while the client trained:
  rep0: datum_mean=1.0084 — 30/32 datums NORMAL (0.46-0.85);
        datums 4,5 DESTROYED (6.33, 5.51 nats); 13,14 mildly high (1.32,1.24)
  rep1: datum_mean=0.5354 — ALL 32 normal; datums 4,5 = 0.43/0.48

⇒ The "loss bump" = individual (adjacent-in-execution-order) datums getting a
catastrophically wrong forward, intermittently, healing within minutes. Not
batch-wide, not persistent state, not client-path (direct probe sees it).
Adjacency ≠ THD partition boundaries (partition 0 = datums 0-6; only 4,5
destroyed) → fault unit ≈ consecutive per-document kernel work.

Leading mechanism: the incident's marginal DSA top-k condition in SILENT form
— #814 fixed the OOB-write crash; marginal docs may still take in-bounds
garbage top-k selections, flipping on/off with tiny weight/numeric changes
(same signature as incident ops passing byte-identical retries). Being tested
via live re-probes ×8 (recurrence pattern) + devbox Phase-D margin capture on
batch-0 docs 4/5.

## Re-init discriminator (live trainer nwxgj6q, ~05:40) — INIT-LINKED, NOT WARMTH

After 14 clean client steps + 8 clean probes, killed the client and fired
/init_trainer_server (REBUILD path — state accumulated) on the warm process,
then probed batch-0 ×6 immediately:
  rep0: datum_mean=1.7056 — datums 4,5,6 DESTROYED (5.86/8.49/11.16!),
        22/29/30 elevated (2.7-3.2)
  rep1-5: healing without ANY weight update (0.82/0.766/0.795/0.797/0.808),
        occasional mild flickers (one datum 1.6 in rep5)
Datums 4,5 destroyed in BOTH observed events (+ different extras each time)
→ data-anchored proneness + stochastic trigger.

Implications:
- Process warmth (JIT/autotune) ruled out (hour-warm process, caches hot).
- Weights ruled out as the healing state (healing occurred across read-only
  /forward ops with B=0 adapter — the forward mathematically equals base).
- ⇒ The healing state is MEMORY: alloc churn at (re)build leaves fresh/stale
  pages that some kernel reads (uninitialized/stale-workspace read) →
  garbage forward for docs whose shapes hit the bad read pattern → pages get
  overwritten by subsequent work → decay. Explains: decay-with-ops, per-doc
  proneness, node randomness, B300>B200 amplitude, prod-vs-devbox
  (expandable_segments + no warmup on the rebuild path), and is the SILENT
  READ twin of the #814 OOB-WRITE incident class. Pervasive top-k tie
  degeneracy (margin capture: exact ties every call, ~400/1401 rows <1e-3)
  marks the DSA indexer as the prime suspect kernel family.

## PARTITION-TAIL SIGNATURE (~06:15)

Batch-0 THD partitions (order-preserving greedy fill, cap 262144):
p0=datums 0-6, p1=7-14, p2=15-22, p3=23-30, p4=31.
Every corrupted datum across all three destruction events sits at a
partition TAIL:
  event1 (boot):        4,5 destroyed + 13,14 mild  → tails of p0, p1
  event2 (manual reinit): 4,5,6 + 22 + 29,30        → tails of p0, p2, p3
  event3 (hammer c0):   4,5,6 + 30                  → tails of p0, p3
⇒ corruption lives at the END of packed THD rows — where the pad/candidate
region sits. Consistent with a kernel consuming uninitialized memory beyond
valid keys when freshly allocated (garbage fp16 scores → candidate-flood →
silent garbage top-k selections post-#814). Margin capture already shows the
top-k boundary is tie-degenerate everywhere, so tail docs are maximally
vulnerable.

Prod rebuild→first-probe destruction rate so far: 3/3.

NEW BUG (secondary): the third in-process init_trainer_server REBUILD
deadlocked the trainer (per-rank LoRA-init lines then silence, GPUs idle,
op never completes) — rebuilds are not reliably re-entrant. Wedged pod
deleted; StatefulSet recreating (fresh boot = fresh destruction window).

## Rotation test (queued for the fresh boot window)

probe_batch0_rot8.json = batch 0 rotated by 8 (order 8..31,0..7).
Predicted partitions: [8-15],[16-22],[23-30],[31,0-5],[6,7] → predicted
destroyed-if-position-anchored: tails 15, 22, 30, 5, 7 (previously-clean 15
and 7 becoming victims = strong confirmation; previously-destroyed 13,14 now
mid-partition = should stay clean).

## ROTATION TEST (fresh prod boot, ~07:0x) — POSITION-ANCHORING PROVEN

rot8 probe rep0 (boot window): datum_mean=2.82. Destroyed (orig idx):
11,12,13,14,15 (tail run of partition [8..15]) + 18,19,20,21,22 (tail run of
[16..22]) + 29,30 destroyed / 27,28 mild (tail run of [23..30]). Partitions
[31,0..5] and [6,7] untouched this event. Previously-3x-destroyed docs 4,5,6:
CLEAN (now early/mid-partition). Previously-never-destroyed docs 15,22:
DESTROYED (now tails). rep1: residual (22 → 2.48) then healing; straight
probe right after: fully clean (window decayed).

⇒ Corruption = TAIL RUNS of packed THD partitions during the post-boot/
rebuild window; depth 1-5 docs per partition per event; per-event random
subset of partitions; decays over ops. Victims adjoin the packed row's pad
region → uninitialized pad/tail memory feeding attention on fresh
allocations; corruption depth couples to CP zigzag chunk geometry.
Combined prod evidence: 4/4 fresh-window events fired (3 straight + 1 rot).

## Poison test (devbox, ~16:14 UTC 07-30) — NEGATIVE

Filled all 16 GPUs with 0xFF (fp16 garbage) on both devbox nodes, then booted
the trainer and probed batch-0 ×4 immediately: 0.7615/0.7646/0.7657/0.7684 —
clean. Caveats: the 756GB weight load + bridge conversion + warmup rewrite
most memory between poison and first op, and the devbox allocation order may
never map the vulnerable buffer onto poisoned pages. Kills the naive
"any garbage memory reproduces anywhere" form; does NOT touch the prod-side
positional/window evidence (4/4 on e02-sg pod nodes). Environmental trigger
(what supplies garbage on prod pods but not devbox: node pool, container
memory path, driver page handling) remains open — buffer identification needs
a repro on prod-pool hardware or in-pod instrumentation.

Devbox totals: 0/6 fresh windows fired (5 rebuild + 1 poisoned boot).

## Next experiments

- E-prod-mimic: fresh boot with (a)+(b), then POST /init_trainer_server
  {lora_rank:32} (skip path), then /forward batch 0 ×3. 1.5 → cornered;
  0.76 → escalate to rank-mapping / entrypoint diff.
- Also available: init-REBUILD path test (post-arm1, one op + one forward).
- Phase D margin capture + flash/fused A/B still queued (mechanism detail for
  the token-level flip machinery; bump amplifier now looks prod-boot-specific).

## Prod anomaly shape (for reference)

8w6y12q datum-mean by step (1-indexed): 1.53 2.01 1.80 1.51 1.71 | 0.70 0.61
0.51 0.51 0.53 0.71 0.81 | 1.09 0.90 0.90 | 0.49 | 0.86 0.88 | 0.49 0.51 0.58
0.52 0.53 | 0.98 | 0.66 0.57 | 0.84 1.19 | 0.50 | 1.18 | 0.67 0.47 ...
Devbox arm0 same batches: 0.76 0.60 0.60 0.52 0.50 | 0.50 0.50 0.48 0.49 0.48
0.44 0.47 | 0.46 0.45 0.44 | 0.43 | 0.44 0.41 | 0.43 0.45 0.45 0.43 0.47 |
0.46 | 0.45 0.45 | 0.43 0.44 | 0.43 | 0.45 | 0.46 0.44


> Note (2026-07-30): investigation W&B runs moved out of oe-grader-sft to baseten-training/jackrao-lps1003 (nspvxlhu→itcqft6p, ykpzwhyn/liveprobe→94z3lu2o); old run ids/links are dead.

## 2026-07-30 evening — code-verification session (source-level, no GPU)

- **Read-past-scores hypothesis killed in source** (was candidate #3; also
  the premise of the original experiment_handoff.md, since rewritten): four
  walls verified in the vendored wheel — unconditional -inf prefill of the
  scores output (_interface.py:207-209), n-block loop clamped to per-segment
  seqlen_k (indexer_fwd_sm100.py:1017-1028), boundary tiles masked -inf
  pre-store (:1168-1177), K input fully materialized via index_select
  (dsa_cudnn_kernels.py:796-797). Every score top-k reads is a real q·k or
  -inf.
- **Under-write mechanism narrowed**: multi-CTA/dynamic-multi-CTA variants
  are compiled OUT on our entry path (defaults False, static one-CTA-per-row
  grid, decode_varlen.py:534-541/:655-663); trivial branch full-writes incl.
  -1 padding (varlen_util.py:468-487) → defect must be in the long-row RADIX
  write-out of the `large_occupancy` (num_rows>148, compile-cache key :611)
  variant — smem candidate buffer shrunk + gmem spill enabled (:157-178),
  prod-only shapes, never compiled by small tests. Exact edge still unproven.
- Context: decode_varlen is the only top-k impl the wheel ships
  (api.py:18-19); our usage is NVIDIA's announced GLM-5.2 CP-training recipe
  (Megatron-Bridge discussion 4957) — strengthens the upstream report.
- Docs updated accordingly: CODE_AUDIT_TOPK.md, HANDOFF.md, VERDICT.md
  (evening addendum), EVIDENCE_INDEX.md (evening table), FOLLOWUP_TICKETS.md
  item 4 re-cut, experiment_handoff.md rewritten.

## 2026-07-30 night — E1/E2 output-indices under-write experiments (devbox tj-3y0gjkq)

Ran the rewritten experiment_handoff.md program on a fresh 2×8 B300 devbox
(GLM-5.2-FP8, CP16/EP16 THD, flash, LoRA r32, max_seq_len 262144, stack
trainer-cuda13-sm103 @ 0e0b65a, wheel 1.26.0+dsatopk1). Tooling in `probe2/`
(harness `probe2/sitecustomize.py`, analyzer `probe2/audit_summary.py`, boot
wrapper `probe2/boot_probe2.sh`, local kernel drivers `probe2/test_*.py`).
Raw: `probe2/runs/{audit,stage}/SUMMARY.txt` + probe jsonl/log,
`probe2/results_{flood_sweep,prodgeom}.txt`.

**Result: the under-write does not reproduce. Every metric is zero under a
verified detector, in-situ and locally, at real production geometries.**

### Method note — targeted allocator poison replaces the whole-GPU poison

The spec's E2 (whole-GPU int32 poison via poison_gpus.py) is a blunt
instrument: it hopes hostile bytes land in the one buffer that matters. Instead
the harness poisons *exactly* that buffer. The wheel allocates
`output_indices_torch = torch.empty(num_rows, top_k, int32)`
(indexer_top_k_decode_varlen.py:684) and it is the FIRST real allocation after
the compile-cache lookup, so filling an (rows, k) int32 block with a sentinel
and freeing it immediately before the call hands the kernel a known-poisoned
output buffer. Any slot still carrying the sentinel afterwards is an unwritten
slot. Sentinel 200003: > every observed sk (so unambiguous) and < 262144 (so it
also mimics the in-range garbage the whole-GPU poison would have produced).

Crucially this carries its own **positive control**: per shape, stage → free →
`torch.empty` of the wheel's exact shape → measure sentinel fraction. Without
it a zero count is ambiguous (correct kernel vs block never handed over). The
control returned **1.0000 on all 91 shapes in-situ and all 48 local cases**,
including under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, so the
detector is proven live rather than assumed.

### In-situ staged boot (the load-bearing run)

Real trainer, 16 ranks, real scores/allocator history, boot warmup + 4×
batch-0 /forward reps:

| metric | value |
|---|---|
| top-k calls / rows audited | 42,336 / 190,506,624 |
| rows through the radix branch (the suspect) | 178,805,760 |
| shapes staged / failing control | 91 / **0** |
| **unwritten slots (under-writes)** | **0** |
| duplicate indices within a row | 0 |
| selected indices ≥ row window | 0 |
| negative (-1) sentinels | 11,848,113,984 (all from window<k rows) |
| batch-0 mean_nll, 4 reps | 0.7643 / 0.7651 / 0.7647 / 0.7659 |
| datums with NLL > 2.0 | 0 |

Baseline gate held (band 0.760–0.771 over five prior clean boots), so the
GPU→CPU syncs the harness adds are not masking or creating the effect.

### Local geometry hunt (48 cases × 2 reps, seconds each, same detector)

`probe2/test_flood.py` + `probe2/test_prodgeom.py` drive the wheel kernel
directly, so geometry space is searchable without 20-minute boots:

- **Candidate-flood/spill edge — REFUTED, in source and by test.** For fp32 the
  smem candidate buffer holds `min(max_smem_input_size, max_num_cols)` entries
  (varlen_util.py:40-95). Ties at the threshold coarse bin are stored
  `if pos < indexer_topk_smem_input_size`, and the refine pass clamps
  `num_input = min(s_num_input[r_idx], smem_input_size)` (:805) — which reads
  like a silent drop. It is not: when `enable_gmem_store` is true the `else`
  branch spills the overflow to gmem (`buffer[0, buffer_pos] = idx`, :666-672),
  and when it is false capacity == `max_num_cols` (bucketed to a power of two,
  so ≥ the actual column count) and `pos` cannot reach it. Both branches are
  safe by construction. Tested anyway across the capacity boundary (cols
  32768/40960/49152/60192/65536/98304 × relu_flood/all_tied, i.e. up to 98,240
  ties against a 16,384-entry buffer): zero unwritten slots everywhere.
- **Real prod geometries — clean.** 9 measured (rows, sk, window) triples from
  the audit above × {arange, interleaved} windows × {relu_flood, all_tied},
  covering odd/unaligned sk (39501, 48906, 52668, 58311 — the `_fill_oob`
  aligned-tail path, the #814 family), per-row varying windows, launches mixing
  window<k and window>k rows, and rows up to 15,360: zero unwritten, zero
  duplicates, zero out-of-window, zero short-row deficits.

### E0 branch/shape audit (both boots)

- Score production is **entirely** the cuDNN path: `cudnn_fw` = 17,883 calls vs
  `torch_scores` = **0**. All 6,720 `_indexer_topk_from_score_chunks` calls
  arrive with `bottom_right_key_start` set and starts/ends/score_seq_lens None
  → the line-644 dense-with-q_causal_offsets branch. The torch scorer that
  -inf-masks in Python never runs in this configuration.
- Launches are prod-shaped: rows 2–15,360, so `num_rows > 148` holds for all
  large calls → the `large_occupancy` variant with the gmem spill path is what
  was instrumented, not the small-shape variant unit tests compile.
- K is even everywhere (2048 on large calls, 1576, plus small 2–64); zero
  odd-K calls, so the fully-initialized torch fallback never fired and never
  masked anything.

### Scores-path refutation independently confirmed at runtime

Boot 1 audited the full invalid region (columns ≥ each row's seq_len) as
*delivered* to top-k: **1,601,212,824,960 elements, 100.000% -inf** — zero
finite, zero NaN, zero +inf. That is runtime confirmation of read-side walls #1
and #3 from the evening source read. The sampled regression watch in the staged
boot agrees (841 calls, 0 finite). Note this also makes the spec's `neginf`
control analytically a no-op — writing -inf where -inf already is — so that arm
was skipped rather than run.

### What this does and does not settle

Settles: on this devbox, at real production shapes and with adversarial tie
content, the cuDNN DSA top-k **full-writes `min(K, window)` slots on every
row**, and the two code paths that could plausibly drop candidates are safe by
construction. "torch.empty + unwritten rows" is not reproducible on demand, so
the last unproven link in VERDICT.md's chain remains unproven — and now has a
strong, well-controlled negative against it.

Does not settle: prod fires 4/4 fresh windows and the devbox has now fired 0/8
(6 prior + these 2). Everything here is devbox; the remaining discriminator is
the spec's own fallback — the E1 audit harness on a **fresh prod session
window**, where the events actually occur. `probe2/` is prod-ready for that:
audit-only mode adds no allocation, and `stage` mode is what turns a silent
under-write into a counter if one exists there.

## 2026-07-31 — BISECTION: is it data? is it the script? (devbox tj-3y0gjkq)

Zoomed out at Jack's direction, to test the two claims VERDICT.md asserts but
never measured ("not data", "not recipe/script") before digging further into the
trainer. Tooling: `probe2/data_forensics.py` (offline packing/position/content),
`probe2/worst_tokens.py` (per-token top-k with decoded context — the
`debug_worst_tokens` idea from basetenlabs/subprime-rl, reimplemented, see
`probe2/README.md`), `probe2/packing_determinism2.py`. Raw:
`probe2/runs/{window,sweep}/`, `probe2/results_*.txt`.

**Both claims survive, now with numbers. Neither data nor script explains it.**

### DATA — ruled out, exhaustively

- **All 1024 documents probed** (32 batches × 32 datums, one op each, frozen base
  weights, `runs/sweep/`): **zero datums above 2.0 nats, zero above 5.0**. Worst
  single document in the entire corpus = **1.153 nats**. Prod destroys documents
  at **5-11 nats**. No document deterministically reproduces destruction.
- Batch means span 0.7196-0.7954 (spread 0.076, ±5%). Prod bumps are 1.5-3×.
- Prod-label correlation is noise: spike/bump-labelled datums score **+0.0113
  nats** vs quiet (Welch t=+1.25, n=704). Content explains ~1-4% of the effect.
- No duplicate documents; no datum under 2048 tokens (all engage the DSA indexer).

### SCRIPT / PACKING / POSITION — ruled out on three independent grounds

1. **Position does nothing.** Partition map reconstructed by greedy packing
   (reproduces the documented batch-0 map `[0-6][7-14][15-22][23-30][31]`
   exactly). Tail-of-partition documents — the slots prod destroys — score
   **-0.0070 nats** vs mid/head at batch level (t=-0.62, n=704) and **-0.0027
   nats** at TOKEN level across **500,679 supervised tokens** (t=-0.39). Slot
   profile is flat (0.7298-0.7880, no trend). A deterministic packing or masking
   bug at partition boundaries would show a large positive delta here.
2. **Packing is deterministic.** The 4 byte-identical `/forward` reps produced
   exactly 4-periodic top-k launch fingerprints (rows, sk, k, sl_min, sl_max) on
   every rank checked (`runs/stage/PACKING_DETERMINISM.txt`). Data and script
   bugs are deterministic functions of the input, so neither can produce the
   observed heal-on-replay (prod event1 datum 4: **6.328 → 0.427 nats** on the
   identical payload).
3. **Masking/alignment is correct.** The first supervised target is the single
   constant token `{"` (id 4913) in all 2112 datum-dumps, sitting exactly after
   the prompt tail `...<|assistant|><think></think>`. An off-by-one would put a
   different token there. `prefix_len` well-formed in all 704 datums; one
   identical 3-token prefix across every datum.

### TOKENIZER — healthy

Decoded with the GLM-5.2-FP8 tokenizer on-box. The worst tokens are all
irreducibly unpredictable *content*, decoding sensibly: `'C'` in "NCCN guideline
(Citation 1)" (20.5 nats), `' Mak'` in "Cochrane Makrgeorgou et al.", `'/J'` in
"Flythe/JAMA review", `' ALT'` in "Strong ALT thresholds". Same positions every
rep, wobbling 1-2 nats (one token: 20.48 / 18.77 / 18.04). No corruption
signature, no template leakage into the supervised region beyond a rare
`'/system'` (30 occurrences, mean 9.3 — worth a glance but immaterial).

### Loss concentration — NOT a finding in itself, but the arithmetic is load-bearing

**Calibration (Jack, 2026-07-31): the distribution shape is unremarkable.**
Heavy-tailed / near-bimodal per-token loss is normal for any LM, and especially
for SFT on templated JSON completions: scaffolding tokens cost ~0, content tokens
cost a lot. Do not read the histogram as evidence of anything by itself. The one
part that *is* load-bearing is the arithmetic in the third bullet below.

Per-token supervised NLL over 500,679 tokens: **p50 = 0.0106**, p90 = 2.56,
p99 = 7.82, max = 20.5, mean = 0.746. Half of all supervised tokens are
essentially free; the datum mean is dominated by ~10% of its ~237 tokens. Two
consequences:
- It explains the *small* run-to-run wobble: 1-2 nats of numeric noise on a
  handful of 20-nat tokens moves the datum mean visibly.
- It rules content OUT for *destruction*: taking a datum from 0.76 to 6.0 nats
  needs ~1250 extra nats, i.e. wholesale corruption of the document, not a few
  bad tokens. Consistent with "the document attends a garbage key set".
- Recipe note for the customer: supervised fraction is **0.0089** — 99% of the
  forward compute is unsupervised prefix (median 25.8k tokens carrying ~237
  supervised).

### Fresh-window boot with the mitigation OFF — still clean (removes my own confound)

Both earlier boots this session ran with the uncommitted full-footprint warmup
patch active — the mitigation designed to burn the window. Re-ran with
`BT_SKIP_FULL_WARMUP=1` (prod-equivalent: warmup 85.9s vs 252.6s patched), probe
as the FIRST real op, 6 reps: **0.7591 / 0.7669 / 0.7642 / 0.7631 / 0.7610 /
0.7656**, max datum 1.006, zero duplicates, zero out-of-window
(`runs/window/`). So the devbox's failure to reproduce is NOT explained by the
mitigation. **Devbox now 0/9 fresh windows vs prod 4/4.**

### Where that leaves the search space

Exonerated to date (mine + prior sessions): data (bit-verified batches + all
1024 docs), client/script metric formula, packing/position, masking/alignment,
tokenizer, weights (sha256), config (read from a live prod pod), env, seed,
allocator, warmup sequence, adapter init (B=0 ⇒ exactly zero at step 0), FP8
(compute is bf16), forward-only vs full training (`trainF` replays ran real
`forward_backward`+`optim_step` from step 0 in a fresh boot: step0 = 0.7642 vs
prod step0 = 1.45-1.53), entrypoint (devbox uses the identical
`server/scripts/launch.sh`), DSA top-k output-indices under-write (staged
detector, 190.5M rows), DSA scores read-past (source + runtime).

**The residual is the prod-vs-devbox BUILD/RUNTIME, not the data, the script, or
the trainer's logic.** Prod runs the prebuilt image `trainer-cuda13-sm103-0e0b65a`;
the devbox runs a venv compiled from the same commit — same code, different
built artifacts. Devbox manifest captured for the diff (`runs/devbox_venv_freeze`
attempt returned empty; versions read directly instead): torch 2.11.0+cu130,
nvidia-cudnn-frontend 1.26.0+dsatopk1, transformer-engine 2.16.0, triton 3.6.0,
nvidia-cudnn-cu13 9.19.0.56, nvidia-cublas 13.6.0.2, transformers 5.8.1,
nvidia-cutlass-dsl 4.5.2.

Recommended next step — and explicitly NOT more devbox kernel archaeology, which
is the rabbit hole: run the `probe2` harness in `audit` mode on a fresh **prod**
session, where events fire 4/4. Audit mode is prod-safe (no extra allocation,
GPU→CPU syncs only) and yields the spec's E1 dump metrics.

**PARKED by Jack (2026-07-31): the image-vs-venv diff.** Real candidate (see the
`LD_LIBRARY_PATH=""` evidence below) but it is a shot in the dark until something
narrows it — do not spend a session on it without a reason to.

### Memory-freshness hypothesis — TESTED, and it kills the whole-GPU poison plan

Jack asked whether prod destroys documents because its GPU memory is "fresh"
while the devbox is "warmed", and whether random-initialising surrounding memory
would reproduce it. Tested directly (`scratchpad/fresh_mem_test.py`): wrote 8 GiB
of the in-range sentinel 200003, exited the process, and had a fresh process
allocate 8 GiB uninitialised and search it → `nonzero=0`, sentinel hits 0, all
zeros. **The driver scrubs device pages between processes.**

Consequences:
1. "Prod fresh vs devbox warm" cannot be a CROSS-process effect. Every trainer
   process starts from zeroed pages on this platform, prod and devbox alike.
2. **The spec's E2 (whole-GPU in-range int32 poison via `poison_gpus.py`) is
   futile by construction** — the poison is scrubbed before the trainer starts.
   The docs attributed the earlier 0xFF poison null to "0xFF decodes to -1, the
   legitimate sentinel"; that may also be true, but it is not the binding reason,
   and choosing in-range values would NOT have rescued the experiment.
3. Note the inversion this exposes: the freshest memory is *zeroed*, and zeros are
   filtered downstream by the `>= starts` bounds check. Garbage requires
   RECYCLED blocks still holding a previous tensor's values — i.e. the warmed
   state, not the fresh one. The "fresh window" framing in VERDICT.md is in
   tension with this.
4. The only viable uninit source is within-process caching-allocator reuse. That
   was tested precisely for the top-k output buffer (staged block, control
   1.0000, 190.5M rows, 0 unwritten). **Not yet tested broadly:** a free-list
   poisoner that fills every cached block across size classes with an in-range
   sentinel and frees them without `empty_cache()` before an op, so ANY
   uninitialised buffer in that forward (top-k scratch `buffer_torch`, attention
   workspace, MoE dispatch, KV) sees garbage. That is the remaining devbox-side
   experiment worth doing, and it is the correct form of Jack's suggestion.

### Concrete prod-vs-devbox delta found (not yet pursued — parked)

`run_trainer_node.sh:39` does `export LD_LIBRARY_PATH=""`, so on the devbox all
CUDA libs resolve to the pip wheels — verified from `/proc/self/maps`:
`.venv/.../nvidia/cudnn/lib/libcudnn.so.9`, `nvidia/cu13/lib/libcublas.so.13`,
`nvidia/nccl/lib/libnccl.so.2`. But `server/scripts/launch.sh` states the image
deliberately PRESERVES `/usr/local/cuda/lib64` in `LD_LIBRARY_PATH`, and the ali
B300 node's system CUDA is **12.8** while the build is cu130. So prod may load a
different `libcudnn`/`libcublas` than the devbox does. Same Python, different
kernels underneath — the right shape for a bug that fires 4/4 in prod and 0/9 on
devbox. Also relevant: anything compiled at install time (transformer-engine) and
the locally-rebuilt `+dsatopk1` cuDNN-frontend wheel can differ binary-wise at
the same version string.

## 2026-07-31 overnight — prod-vs-devbox parity night: SOLVED

Full running log: parity/NOTEBOOK.md (authoritative for this session). Summary:

1. **Simultaneous A/B**: fresh loops session 8w6k4y3 / trainer 5wolkzw (image
   0e0b65a, verified) + devbox boot in the same minutes, identical batch-0
   payload, full per-token logprob dumps both sides. Prod boot window fired
   7/8 + 5/8 reps (13 destroyed datums, up to 18.3 nats); devbox 0/14 clean.
   Prod lifetime windows 6/6, devbox 0/10 — under STOCK envs.
2. **Fingerprint diff killed the environment mysteries**: same hardware pool
   (both b300-1-*, L20D, driver 580.105.08), same kernel, byte-identical
   cudnn/cublas/nccl/torch wheels (60/73 mapped .so sha-identical;
   1.26.0+dsatopk1 on both), same allocator conf, same topology (torchrun
   2x8). Residual deltas: 6 devbox-only env vars + NVTE_CUDA_ARCHS + glibc
   point release + nvshmem .so size (same version).
3. **Env-mimic boot fired on the devbox on the first attempt** (prod-exact
   env), and single-var bisection named the toggle:
   `CUDA_DEVICE_MAX_CONNECTIONS=1` — present (devbox stock) 0/10 windows;
   absent (prod default) 3/3 windows with near-identical victim sets
   {4,5,6}. gloo/arch/nvte arms clean (see NOTEBOOK for final counts).
4. **Prod causal confirmation**: LWS env patch conn=1 → 8/8 clean window
   reps + second fresh boot also clean, on the same deployment that fired
   13/16 without it.
5. **Layer bisection** (activation-tracer sitecustomize on all 16 ranks, LWS
   initContainer injection on prod): corruption enters at
   decoder.layers.1.self_attention.core_attention (DSAttention), ONLY in
   second-zigzag-chunk bins; embedding/L0/indexer-input-linears bit-clean
   during events; output plausible-scale wrong-content (wrong key set).
   Positional law: only docs with tokens in the second half of a long packed
   row are destroyed (verified against exact packing offsets, 13 event reps).
6. **Top-k selection digests inconclusive by design**: selections churn
   (sha1 mismatch 91-97%, |Δidx-sum| up to 2e11) between CLEAN reps — tie
   plateaus reshuffle below-cutoff candidates. Selection-vs-gather split
   still open; needs a weight-aware instrument.
7. Conclusion AS OF OVERNIGHT: missing stream/event dependency inside the
   DSA core path, masked by single-HW-queue serialization. Mitigation
   (conn=1) validated on prod.
   **CORRECTED 2026-07-31 day session (7 single-variable boots; see
   NOTEBOOK "DAY SESSION 2026-07-31"): every specific consumption-path
   stream-dependency reading was falsified** — top-k TVM-FFI env-stream pin
   changes nothing (both streams are 0 in-situ; the unpinned launch is a
   real latent defect, branch jackrao/dsa-topk-stream-pin, but not this
   bug); full per-gather device syncs still fire (NCCL delivery
   exonerated); no-expandable-segments fires; torch.topk-for-radix fires
   (selection kernel exonerated). Standing conclusion: corruption
   originates INSIDE the fused DSA score/attention kernel path under
   ambient GPU concurrency (suspects: cuDNN indexer fwd CLC dynamic
   scheduler; FlashMLA). conn=1 masking + prod mitigation validation are
   unaffected. Root fix owed; next discriminator = arm H double-exec.

Prior sessions' "prod fresh-window memory" framing is retired: the window =
cold-boot timing regime in which the race actually loses, decaying as clocks/
caches warm — explains heal-on-replay, per-boot randomness, rebuild windows,
and why byte-identical inputs flip.

## 2026-08-01 — Minimization: the bug needs cp>1 (cp1 minimal config clean 5/5)

Jack asked for the smallest f: no CP, synthetic data, few fwds. One devbox
boot on tj-3y0gjkq (same stack as the 10/10 cp16 repro), tp1/pp8/ep2/cp1
(dp2) @ 64k — pp8 on 78 layers is legal via the explicit uneven layout
`_glm52_dsa.py::_GLM52_DSA_PIPELINE_LAYOUTS[(78, 8)]` (a (78,16) layout also
exists; pp16 needs 32 ranks). Trigger conditions identical to the proven
repro: prodenv (CUDA_DEVICE_MAX_CONNECTIONS unset) + BT_SKIP_FULL_WARMUP=1,
5 /forward reps of an identical payload at the /health window. Payload = 2
synthetic 50k-token datums (one "ABCDEFEDCBA" palindrome-text repeat; one
random 256-token chunk tiled — position-sensitive on purpose, since pure
repetition can mask wrong-KV reads).

Result: **clean 5/5**. Per-datum NLL: palindrome 0.000115→0.000131
(max |Δ| vs rep0 = 1.6e-5); random-tile 0.16747→0.16192 (max |Δ| = 0.0056 —
three orders below the 5–11-nat signature). rep0 is not an outlier: every
rep pair diverges >1 nat at 15–165 tokens (rep2-vs-rep3 = 15, rep0-vs-rep4 =
165), all concentrated at two fixed intra-chunk offsets (213 and 102 mod
256) — near-tie tokens under ambient nondeterminism. Means drift
monotonically down with reps 2–4 agreeing tightest → warm-up settling, not a
boot window.

Structural read (why this was predicted): at cp=1 the trainer bypasses the
packed-THD multi-doc path entirely — THD packing is gated on
context_parallel_size > 1 in megatron_controller.py; cp1 routes through
padded-BSH `_pack_batch`. The proven corruption lives exclusively in
second-zigzag-chunk bins, which exist only under CP. So this run empirically
pins the minimization boundary rather than exonerating anything new.

Consequences for the minimal repro: **cp2 is the next candidate** — the
smallest zigzag-preserving config (single node feasible: tp1/pp1/cp2 + ep to
fit, multi-doc packed batch). Synthetic data has not yet been tested under
cp>1; if cp2+synthetic fires, hardware and data minimize together.
Artifacts: `min64k/` (README, config, builder, exact payload, driver, full
per-position logprobs for all 5 reps + positional_deltas.json.gz; run
min64k_0731_234442). Devbox torn down and verified idle.

## 2026-08-01 — Minimization step 2: cp16 @ 32k with 2×30k docs is ALSO clean (3/3)

Same trigger as the 10/10 repro (prodenv conn unset, BT_SKIP_FULL_WARMUP=1,
window /forwards), golden cp16/ep16 leaf with only max_seq_len dropped to
32768, payload = [random-256-tile 30k synthetic, longest real bundle datum
≤30k = batch 21 idx 13 (29,985 tok, prefix 29,682)]. Both docs are
single-doc partitions (two 30k docs can't share a 32k row). Full-position
logprobs all reps.

Clean: max per-datum |ΔNLL| vs rep0 = 0.021 (mudith; and rep0 is the LOWEST
rep). No rep0-outlier per-token structure — pairwise >1-nat counts
comparable for all pairs (random-tile 34/8/31, every hit at intra-chunk
offset 175 mod 256, max 4.99 nats; mudith 5/4/6, all in the supervised
band). Ambient cp16 wobble only. CORRECTION (same night): the "prompt
region quiet" part of this read was an ARTIFACT — /forward zero-fills
logprobs at weight-0 positions, so the masked 29,681 prompt entries were
never measured (see next section).

So {conn unset + boot window + cp16 zigzag + real-data doc + row-tail
supervision} at 32k single-doc rows does NOT fire. Combined with the cp1
run, the minimization ladder now points at prod-scale ROW GEOMETRY as the
missing ingredient: 64–131k packed rows (per-CP-rank local ~8k vs ~1.9k
here) and/or multi-doc partitions (destroyed docs were partition TAILS with
neighbours ahead of them). Next rung: cp16 @ 64k+, many docs packed per row,
synthetic first. Artifacts: `min32k/` (README, exact payload + meta, full
logprobs run min32k_0801_003030). Devbox torn down, verified idle.


## 2026-08-01 (later) — 3×10k multi-doc window run + the masked-logprob artifact

Experiment 3 (Jack's spec, uniform LM supervision everywhere — no masks):
same cp16@32k config fresh boot, payload = [palindrome 10k, random-tile 10k,
mudith b0 i0 first-10k-tokens] = 30k in ONE multi-doc row (first run in the
ladder with a partition-TAIL doc with neighbours), 3 window reps, then
same-boot follow-ups (steady): mudith solo, mudith at row head, and the
b21i13 30k payload with uniform weights.

**No LPS-1003 fire.** Window reps: palindrome 0.00313/0.00307/0.00309;
random-tile 0.3587/0.3567/0.3554; mudith-tail 2.6864/2.7243/2.7017 (rep0
LOWEST). The `destroyed={2: 2.7}` flags on every rep are just this doc's
natural full-sequence LM level crossing the 2.0 threshold — stable, no
fire-and-heal.

**Real finding 1 — the masked-logprob artifact:** /forward returns logprob
**0.0 at every weight-0 position**. Proven arithmetically: boot-1 b21i13
"full-seq NLL" 0.0090 = 0.895 × 303/29,984 exactly; direct check = 29,681
zeros, 0 nonzero. Consequences: (a) every masked-payload dump in this
investigation only measures supervised bands (the historical probes knew
this — probe_nll slices lp[p-1:] — but the min32k boot-1 "prompt quiet"
claim did not); (b) an apparent "boot-dependent churn explosion" (4-6 vs
5,470 churning tokens for the same doc across boots) was zeros-vs-real, NOT
boot state. Uniform weights are now the standing probe policy (Jack's
directive) — which also makes full-position recording actually true.

**Real finding 2 — ambient per-token churn on real text is LARGE:** between
identical no-optim forwards (all rep pairs equally, rep0 not special),
real-text docs churn 12–18% of positions by >1 nat (max 15–19 nats!):
mudith b0i0-10k ~1,150–1,200/9,999 in EVERY arrangement (row tail / row
head / solo — position and packing exonerated for this effect); b21i13-30k
~5,470/29,984. Synthetic docs barely churn (palindrome 0, random-tile ≤3 at
one intra-chunk offset) because their logits are saturated. Datum means
stay stable (±0.04) → the churn is symmetric. Consistent with the known
clean-rep behavior "top-k index digests churn 90%+ between reps" (near-tie
score cutoffs reshuffle selected KV sets; uncertain real-text tokens are
sensitive to which keys get picked). NOT the destruction signature (that is
an asymmetric +5–11-nat MEAN shift on whole docs, rep0-only, healing) — but
it sets the noise floor any future detector must clear: per-token deltas
below ~19 nats on isolated uncertain tokens are ambient here.

Ladder state after 3 rungs: cp1@64k clean, cp16@32k single-doc clean,
cp16@32k multi-doc (tail doc, real data) clean. Still missing vs the 10/10
repro: prod-scale rows (64–131k packed, per-rank local ~8k vs ~1.9k).
Next rung: cp16 @ 64k+ with many real docs packed per row (approaching
batch-0, which at 262144 IS the proven repro).

Artifacts: `min32k/x3x10k_0801_011540/` (window reps + mud_solo + mud_first
+ b21_boot2 uniform resend, all full-position logprob dumps),
`payload_3x10k.json` + `.meta.json`, `payload_mud10k_solo.json`,
`payload_3x10k_mudfirst.json`, `payload_min32k_uniform.json`. Trainer left
RUNNING (warm, window spent) pending next experiment.

## 2026-08-01 — POSITIVE CONTROL FIRED: partition 1 alone, uniform loss; cutoff mapped at chunk 17/32

Jack's design: batch-0 docs 0-6 ONLY (the historically-destroyed [0-6]
partition), exact ids/order, but uniform supervision everywhere → validates
the day's negatives AND maps the destruction boundary in one boot.

FIRED 1/1: rep0 doc4 4.78 / doc5 8.36 / doc6 8.60 vs healed ~3.4-4.0
(docs 0-3 flat across reps). Chunk-aligned profile (7,968-token zigzag
chunks, mean rep0−rep1 logprob): chunks 0-12 clean (|Δ|≤0.10); chunks 13-16
= flicker band (mixed signs −0.45/+1.03/+1.45/+2.68 — reps 1-2 also
unstable there); chunks 17-31 hard-destroyed (−2.0 to −6.2 sustained). So
the second-half law resolves to: hard onset ~row 135k (one chunk past the
127.5k midpoint) with a 4-chunk flicker skirt straddling the midpoint.
Bonus findings via uniform loss: shared ~2k template prefix is immune in
every doc; healed reps churn ~2× higher in the row's second half (0.25-1.6
vs 1.3-3.3 mean/chunk) — the min32k "churn is position-independent" note
was a 30k-row statement only.

Consequences: (1) all three 08-01 negatives are REAL (box verified firing
same-day, same stack/wheel +dsatopk1); (2) single-partition ops fire —
partition count/sequencing exonerated; (3) f shrinks to ONE 254.5k row of 7
real docs (fires) vs 30k rows (clean ×3) → row-scale threshold between 30k
and 254.5k, unbisected. Next: identical design at ~64k and ~131k.
Artifacts: `ctrl/` (README, payload+meta, full-position dumps run
ctrl_p1_0801_013628). Trainer left up (window spent).

## 2026-08-01 — steady-state soak: second-half instability persists at ~8%/rep (NOT window-only)

146 identical /forwards at ~9.5s each on the warm fired-control trainer
(conn UNSET). Docs 0–3 pinned (total range ±0.06). Docs 4–6 (row second
half) visit a coherent low-NLL state in ~8% of reps (doc5 median 3.984 vs
min 2.161 — a 2-nat coherent swing over 34.6k tokens), rate flat across the
run. Since lower NLL = better prediction, the leading (unconfirmed) reading
is inverted from the window framing: the rare low state may be the CORRECT
computation, with ~92% of steady-state forwards semi-degraded in the row's
second half under prod env. All historical "steady state is stable"
measurements were taken under conn=1. Decisive next experiment: same
config + payload with CUDA_DEVICE_MAX_CONNECTIONS=1 — its level names the
correct state (and if it sits at 2.16 for doc5, prod GLM training loss on
row-tail docs has been systematically inflated on EVERY step, not just
boot windows — gradients included). Artifacts: ctrl/soak_0801_015956/
(soak.jsonl all reps, 17 exemplar full-position dumps), ctrl/soak.py.
