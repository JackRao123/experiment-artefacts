# rearm/ — no-restart rep0 re-arm + exact rep0-vs-rep1 divergence (2026-08-01)

> **RESOLVED 2026-08-01 ~12:30Z — this document's conclusion is SUPERSEDED.**
> The kernel was innocent: the corruption is the -inf output PREFILL racing
> the tvm-ffi kernel launch (default stream + conn unset) and ERASING the
> kernel's stores — erased-rows masks are byte-identical to "clamped
> cu_seqlens" masks, which is what this document chased. Fix = dedicated
> launch stream in the wheel (+dsatopk5 after the 08-02 review),
> PR https://github.com/basetenlabs/trainers/pull/875, full verification
> protocol passed. Authoritative write-up: rearm/NIGHT_0801_FINDINGS.md.


Jack's directive: stop paying a 12-min boot per rep0 observation. (1) Find a
way to re-create the rep0 phenomenon on a warm trainer, by bisecting "cold
start" into its ingredients and re-applying them one at a time. (2) With a
cheap repro, pin exactly where rep0 and rep1 computations diverge.

## Design

**Cold start is a set.** In-process `init_trainer_server` rebuild is the
existing proof that a process restart is NOT required (prod windows were
opened by rebuilds — rebuild_hammer.py). Rebuild redoes: full GPU memory
free+realloc, HF weight reload, model+optimizer object rebuild. Rebuild does
NOT redo: process groups (initialize_runtime=False), CuTe compiled-kernel
caches (module-level `_compile_cache` dicts in
cudnn/deepseek_sparse_attention/*/_interface*.py), python module state, CUDA
context/loaded cubins, cuBLAS handles.

- **Phase A (warm ctrl trainer, zero boot cost):** baseline /forward ×2 →
  forced rebuild (`init_trainer_server lora_rank=16`; rank change defeats the
  pristine-stack fast path; LoRA B zero-init keeps NLL comparable) →
  immediate /forward ×4. Fires ⇒ cold ingredient ⊂ rebuild set (memory /
  weight-load / model-build), compiled-kernel + NCCL state exonerated.
- **Phase B (one boot, then unlimited warm experiments):** harness7 lever
  hook + harness6 double-exec.
  - `harness7/sitecustomize.py`: wraps
    `MegatronBridgeController.execute_forward_backward` on every rank; reads
    `levers.json` (seq-bumped, CPFS-shared) at the start of each /forward op
    and applies actions BEFORE the forward: `empty_cache` (cold allocator),
    `clear_cute_caches` (cold kernels: re-runs cute.compile + module load
    during the next forward), `clear_cublas_ws`, arbitrary `pycode` (hot-swap
    instrumentation without reboot). Chains harness6 via BT_H6_SITE.
  - harness6 (`BT_DSA_DOUBLE=indexer,flashmla`): runs each suspect kernel
    twice per call, bitwise-compares — churn-immune; a self-disagreeing
    kernel during a destroyed rep names the defect. **Bug fixed 08-01 before
    first run:** the indexer rerun passed `out=None`, but
    `indexer_forward_wrapper` has no `out` param → TypeError on every 2nd
    call → comparison would never have run. Rerun is now a plain 2nd call
    (wrapper allocates its output internally).
  - Both hooks validated by venv import dry-run before booting.
  - Driver: `lever_driver.py` — ladder of {control, empty_cache,
    clear_cute_caches, both, cublas_ws} × 4 trials × 3 reps. Fired = doc NLL
    > +1.0 over control median (destruction is UP; ambient steady-state
    flips are DOWN ~8%/rep — direction disambiguates).

Detection uses the ctrl payload (`ctrl/payload_b0_part1_uniform.json`, one
254.5k row, docs 0-6 uniform loss); destroyed signature = docs 4-6 up 1-5
nats, docs 0-3 pinned.

## Interpretation caveats

- flashmla may have a benign nondeterminism floor (split/atomic reductions):
  judge double-exec by CORRELATION with destroyed reps/chunks, not by any
  nonzero diff. Per-call zero/nonzero records land in dsa_double/*.jsonl.
- CPFS staleness: lever application is confirmed per-rank in levers/*.jsonl
  (call #, seq, applied) — check all 16 ranks applied before trusting a rep.

## Adjudication FINAL (04:39Z, 10 cycles complete) — aggregate

10/10 empty_cache-treated reps DESTROYED; heal rep clean every cycle.
3,887 adjudicated indexer calls in treated reps across 16 ranks:

- verdicts: agree 2,906 / all_differ 750 / run1_outlier 230 / run3_outlier 1.
- **203 whole-segment skips, row_tiles = [62,123] in ALL 203** — always
  EXACTLY the second THD segment (local rows 7957-15913), never the first,
  never partial-rows. Ranks 0-14 (r12 only partial events); r15: zero, ever.
  Concentrated at one call site (call%21==6: 131/203) + scatter.
- Heal/background reps: every disagreement (34/3,169) is run1_outlier with
  runs 2+3 bitwise-identical — run1 executes amid ambient concurrent
  launches, the back-to-back re-runs land in a quiet window and are
  correct. This ~1%/call background is the steady-state degradation.
- In treated reps all_differ dominates: while fresh mappings are being
  consumed even the re-runs are unstable.

**Complete causal chain, all measured:** conn unset (32 unserialized HW
queues) + first forward consuming freshly-mapped memory → CuTe indexer
persistent kernel skips the entire 2nd THD segment on the consumed
execution (leaves -inf prefill) → top-k over all-(-inf) rows selects
garbage KV → second-zigzag-chunk docs destroyed (5-11 nats) → later
forwards reuse warm mappings → only ~1%/call ambient skips remain.
Rep0-vs-rep1 difference = allocator cache emptiness. Nothing else.

Raw: armH7_data/ local mirror (adjudicate_rank*.jsonl, ladder + campaign
logs); devbox originals in rearm/armH7_0801_034625/.

## Adjudication first-look (04:25Z) — run1 is the corrupt one; skip = whole 2nd THD segment left at -inf

Triple-exec adjudicator (adjudicator.py, hot-installed via pycode lever)
on empty_cache-treated destroyed reps. First results (rank1, cycle 0):

- Huge events verdict **run1_outlier** with runs 2+3 bitwise IDENTICAL
  (n23=0): the CONSUMED output is the corrupt one; immediate re-execution
  is correct. n12 = 242,295,057 = the constant seen all session.
- Sample values: run1 = **-inf** where runs 2/3 = finite scores (e.g. local
  row 7957 col 0: -inf vs 41.744). row_tiles_bad = 62/124 = rows 7957-15913
  = EXACTLY the second zigzag chunk's THD segment; col_tiles 455/455.
  ⇒ **the kernel skipped the entire second query segment, leaving the -inf
  prefill in place**; top-k over all-(-inf) rows → garbage KV selection →
  destroyed second-chunk docs. Explains the positional law directly.
- Background small events: "all_differ" scattered partial tile skips in ANY
  of the 3 runs (run3 skips too) — the steady-state churn/flicker floor.

## Lever ladder result (04:04-04:24Z) — empty_cache IS the re-arm, 4/4

On the warm armH7 trainer, 4 trials x {control, empty_cache,
clear_cute_caches, both, cublas_ws} x 3 reps (treated rep = first after
lever): **empty_cache 4/4 FIRED** (docs 4-6 destroyed at 4.9-7.6, walls
70-87s, heals next rep), **both 4/4 FIRED**; control 0/12,
clear_cute_caches 0/4 (16s walls prove recompile happened), cublas_ws 0/4.
Kernel-instrument tie (rank1): treated reps 9/21 indexer calls
self-disagree (= boot-rep0 rate), heal reps 0/21.

⇒ **The cold-start ingredient is the empty CUDA caching-allocator pool.**
A forward that must map fresh physical memory mid-flight (cudaMalloc/VMM
ops interleaved with kernel launches, conn unset) makes the CuTe indexer
kernel emit corrupt scores on the big gathered-KV call at ~40% of calls.
Kernel recompilation, cuBLAS workspaces: exonerated. Repro cost: one
`{"actions":["empty_cache"]}` lever bump + one ~80s /forward. No restart.

## armH7 boot-window result (04:00Z) — THE KERNEL IS NAMED

Boot armH7_0801_034625 (golden config + prodenv trigger + harness6+7).
rep0 FIRED with double-exec active (4.54/6.92/7.60), healed rep1; low-state
visits rep2+rep5 (2/7 — small n). Double-exec (both kernels, every call,
all 16 ranks, bitwise):

| rep (state) | indexer bad/calls | note |
|---|---|---|
| rep0 DESTROYED | **131/336 (39%)** — 8-11/21 on EVERY rank 0-14, **0/21 on r15** | max diff 170-242M elems/call (whole-valid-region scale) |
| rep1 healed | **0/336** | bitwise clean |
| reps 2-7 steady | 6-15/336 (2-4%) | never r15; low-state reps NOT elevated vs median reps |

**FlashMLA: 0 disagreements / ~12k calls (all reps) — EXONERATED.**
Rank law matches the destroyed-chunk law exactly (hard-destroyed chunks
17-31 = second zigzag chunks of ranks 14→0; r15 owns chunk 16 = flicker
only). ⇒ **The defect is nondeterministic output of the cuDNN CuTe indexer
forward kernel** (`indexer_fwd_sm100`), self-disagreeing under back-to-back
identical inputs, at 39% of calls during rep0 vs 0% at rep1 vs 2-4%
steady-state. 21 indexer / 78 flashmla calls per rank per forward.
Analysis: `analyze_dd.py` + ad-hoc rep-boundary table (INVESTIGATION.md).
Raw: armH7_data/dsa_double (local mirror).

Note: harness6 as originally built would have compared nothing (out=None
TypeError on every rerun) — fixed this session before first use.

## Log

- Phase A baseline (03:31Z, warm fired ctrl boot tj-3y0gjkq job 74):
  nlls 3.79 3.62 4.45 2.85 | 3.35 3.80 3.93 → healed/median state.
- Phase A rebuild submitted 03:32Z (op ec6e7cb4); server unreachable during
  rebuild (expected — op blocks the loop); probe chained on-box.
- **RESULT: FIRED 1/1 (03:44Z).** Post-rebuild rep0 = 3.87 3.63 4.41 2.84 |
  **4.802 8.221 8.629** — byte-similar to the boot-window rep0 signature
  (ctrl: 4.777/8.358/8.602); heals by rep1 (3.44/4.01/3.83); rep3 doc6
  mild flicker 4.85. Rep0 wall 52s vs 9.5s warm.
  ⇒ **No restart required. Cold ingredient ⊂ {GPU memory free+realloc,
  HF weight reload, model/optimizer rebuild}. Compiled CuTe kernels, NCCL
  groups, python/CUDA-context state EXONERATED** (all survived the rebuild
  and it still fired). First-compile/first-launch timing is NOT the
  mechanism.
- Rebuild budget note: this was endpoint-rebuild #1 on this process (3rd
  rebuild historically deadlocks); lever hook is the scalable path.
- Timing note: the window survived ~2.5 min idle between rebuild-done and
  rep0 → the armed state is PERSISTENT until consumed by a forward, so the
  mechanism is something the first forward itself does, not background
  co-residency. Prime candidate: allocator cache is empty after
  rebuild/boot, so rep0 maps fresh segments for activations MID-forward
  (malloc/VMM ops interleaved with kernel launches on 32 unserialized
  queues); rep1+ reuses cached blocks. rep0 wall 52s vs 9.5s warm is
  consistent. `empty_cache` lever = the direct test.
