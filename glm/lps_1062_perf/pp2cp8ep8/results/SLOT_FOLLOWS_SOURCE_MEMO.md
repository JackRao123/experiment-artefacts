# SLOT-FOLLOWS SOURCE MEMO — per-slot metadata/buffer wrongness hunt (gauss, 2026-08-13 ~05:5x CDT)

cauchy's assignment, ~30-min window. The empirical constraints to cover:
slot-follows (middle partition P1 = a steady-phase slot; d5 exonerated when
moved out, d1/d2 corrupted when moved in), escalating toward the slot's end,
composition-dependent magnitude, 10–30× noise floor, AND same-boot
nondeterminism (padded PP2 range 0.030 over 4 samples; unpadded PP2 range
0.025). Source at mcore 57efae08b + trainer 73c24b00 lineage. No box actions.

## Verdict up front

- **Suspect (1) — 1F1B p2p buffer reuse/stale-tail: REFUTED at source** (§1).
  Recv buffers are fresh `torch.empty` per call, shape-exchanged and fully
  written under synchronous wait + deviceSync; a ragged tail cannot exist
  (buffer = exact payload shape). It also fails cauchy's covers-both rule:
  it could never fire on uniform padded shapes.
- **Suspect (2) — M=N seam fresh-but-WRONG metadata: CLEARED at source** (§2).
  Per-slot params are freshly constructed per partition and handed in-order;
  the schedule's FIFO input/output lists pair correctly;
  `set_current_microbatch` feeds only TE CUDA graphs (off tonight).
- **The live classes that DO cover the full signature are the
  kernel/scratch/atomic classes** (§3), ranked there with the audit points
  and the discriminator each needs. The strongest structural read: any
  per-layer ULP noise source produces position-escalating DSA/router
  selection flips (more candidate keys with position), and a
  slot-phase-dependent noise source (a scratch buffer whose content depends
  on slot history) produces slot-follows. The two observations jointly point
  at a per-slot scratch/workspace reuse channel in the DSA/cuDNN path, not
  the pipeline plumbing.

## 1. Suspect (1): p2p buffer reuse — refuted

The full recv path, every link cited:

- Shape exchange is synchronous and host-read: `_communicate_shapes` builds
  fresh shape tensors per call, `batch_isend_irecv` + `req.wait()` per
  request + `torch.cuda.synchronize()` (p2p_communication.py:186-263).
- Recv buffers are **freshly allocated per call**:
  `torch.empty(recv_shape, requires_grad=True, ...)` in
  `create_tensor_recv_prev/next` (p2p_communication.py:330-344). No
  cross-call buffer object exists to reuse.
- The buffer shape IS the sender's payload shape (variable-seq path:
  `:322-325` → per-call shape exchange), so the NCCL recv covers the whole
  buffer — **no ragged stale tail can exist**; and on the padded leg the
  shapes are uniform anyway. The batch race workaround
  (`batch_p2p_sync` deviceSync, `:261-263`, `:416-419`) is present.
- All entry points (`recv_forward` :424, `recv_backward` :455,
  `send_forward_recv_backward` :527, `send_backward_recv_forward` :560) run
  `_communicate` with default `wait_on_reqs=True` — no deferred reads of
  unwritten buffers.
- The schedule's saved-tensor lifetime is correct: recv'd inputs are held in
  `input_tensors` until their backward pops them (schedules.py:2354, 2409,
  2415, 2451); only *output* tensors are deallocated after send
  (`deallocate_output_tensor`, :2356/:2411), after the waited send.

## 2. Suspect (2): M=N seam per-slot metadata — cleared

- packed_seq_params freshly constructed per partition inside the packer loop
  (packer.py:134-197, `PackedSeqParams(...)` at :167).
- The schedule pulls `next(data_iterator)` in forward order
  (loss.py:190-191) and attaches that slot's own params
  (loss.py:205-234); the recv'd boundary tensor pairs in-order by 1F1B
  construction (schedules.py:2322-2436; FIFO pop(0) at :2415).
- `set_current_microbatch` (schedules.py:492-493) feeds only the TE CUDA-graph
  replay index (cuda_graphs.py:2520-2561, module.py:300) — CUDA graphs are
  off tonight (eager kernels in all traces).
- The runner's result mapping asserts per-slot counts
  (training_runner.py:420-427).

## 3. The live classes, ranked by signature coverage

**(3a) Per-slot scratch/workspace reuse in the DSA/cuDNN path — top suspect.**
Covers: nondeterminism (stale content varies run-to-run), slot-follows (the
scratch content depends on the previous slot's phase/shape),
position-escalation (noisy indexer scores flip more selections as the
candidate pool grows with position), both pad states (scratch reuse is
shape-driven, and even uniform shapes share the buffer across slots).
Audit points: the cuDNN top-k scratch chunking
(dsa_cudnn_kernels.py:489-523, `_TOPK_WRAPPER_MAX_SCRATCH_BYTES`), the
indexer score buffer (`torch.zeros` per chunk — safe — :561), the
FlashMLA sparse workspace (inside the fused kernel — not visible in-repo),
and the dispatcher's class-level shared D2H stream + event
(`MoEAlltoAllTokenDispatcher.cuda_dtoh_stream`, token_dispatcher.py:380,
:461-462, ordering at :931-955 — a stream is an ordering primitive, not a
buffer, but its event ordering across slots is worth the audit).
Needs: a buffer-content determinism dump (two same-config boots, checksum
the indexer score tensor / the recv'd boundary per slot) — localizes whether
corruption enters before or after the stage boundary.

**(3b) Atomic-order ULP nondeterminism amplified by the top-k
discontinuities.** Covers: nondeterminism, position-escalation,
composition-dependent magnitude. Does NOT by itself explain slot-follows
(would hit every slot), and would hit golden PP1 too — so it ranks below 3a
unless golden proves to have the same noise floor (then the PP2 specificity
is the amplifier's threshold, not the source). Audit points: the MoE
unpermute/combine reduction (token_dispatcher.py:862-885 → TE fused
unpermute kernel — atomicity not visible in-repo) and any `index_add_`/
atomic scatter in the DSA mask path (`scatter_topk_into_index_mask`,
dsa_masking.py:277 — a plain scatter, deterministic given distinct indices).
Needs: golden PP1 same-boot determinism measurement (a gap in tonight's
evidence — the 4-sample nondeterminism was measured on PP2).

**(3c) The p2p/NCCL transport class** — byte-exact by design (p2p is data
movement); the only nondeterminism it could add is via the recv buffer,
refuted in §1. Kept for completeness; ranked last.

## 4. Revisiting the allocator-reuse class (from my Gate-1 memo §5)

The caching-allocator recycled-memory class survives this hunt as the
*substrate* of 3a: `torch.empty` recv buffers are fully overwritten (§1), so
the allocator is not the corruption path there — but any kernel that reads a
freshly allocated scratch region before fully writing it reads the previous
owner's bytes (a prior slot's content — slot-follows) with run-to-run
variation (nondeterminism). The p2p path is cleared; the DSA/cuDNN scratch
path is where this class lives now.

## 4b. CAUCHY FRAMING SHIFT (2026-08-13 ~06:2x CDT) — folded in

- **The ship-unpadded escape hatch is gone:** the permuted verdict means the
  unpadded leg carries the slot defect too. The padded-vs-unpadded residual
  (+0.036, borderline) drops to a SECONDARY item; the slot defect is the
  primary hunt. (TAIL_PAD_DSA_SOURCE_MEMO.md §5 updated accordingly.)
- **Overlap question — YES, the cuDNN kernel-internal varlen thread overlaps
  the slot-defect hunt:** the tail-pad residual suspect (the cuDNN fused
  indexer's internal split) and this memo's top live suspect (3a, per-slot
  scratch/metadata wrongness) converge on the SAME component — the cuDNN
  fused indexer path consumes per-slot packed metadata
  (`_get_multi_packed_cp_thd_metadata` → `packed_cu_seqlens_*`,
  `packed_max_seqlen_*`, `packed_cp_size`, dsa_cudnn_kernels.py:157-176,
  passed at :262-275), and any kernel-side mishandling of that per-slot
  metadata fires slot-dependently with composition-dependent magnitude. The
  two hunts should share one kernel-level probe (a cuDNN-internal dump or a
  forced fallback to the non-fused path as the discriminator — see below).
- **Correction to the "two live suspects" framing:** suspects (1) p2p buffer
  reuse and (2) the M=N iterator seam are the ones this memo REFUTED/CLEARED
  at source (§1/§2) — the live suspects are 3a/3b in §3. If a quick
  discriminator is wanted for 3a-vs-kernel-internal: the non-fused fallback
  (the unfused/reference DSA path) bypasses the cuDNN fused indexer entirely
  — a same-config leg forced onto the reference path separates "cuDNN fused
  kernel" from "everything else" in one boot.

## 4c. The "larger-than-predecessor" audit (cauchy's high-water-mark question, 2026-08-13 ~06:4x CDT)

New empirical pattern: in BOTH document orders the corrupted partition is the
one LARGER than its predecessor (92960→98704, 105584→124784; smaller
successors clean); PP1/CP8 forward-only reproduces the slot structure (M=N/PP2
exonerated; the CP8 DSA forward path convicted; standalone DSA fwd
nondeterministic, spread 0.019). cauchy's question: is the top-k scratch (or
any DSA workspace) allocated once per process/first-call and NOT revalidated
per invocation?

**Direct answer: NO in the visible mcore path — every scratch/workspace is
per-call and shape-revalidated:**

- top-k selection scratch: chunk rows recomputed from the CURRENT
  `(n_rows, sk)` every call (`_indexer_top_k_wrapper_chunked`,
  dsa_cudnn_kernels.py:489-523); no persistent scratch tensor.
- indexer score buffer: `torch.zeros((b, sq, sk))` fresh per chunk per call
  (:561).
- top-k alignment padding: `F.pad(value=-1)` per call (:955-958);
  `_pad_topk_result` fills with -1/fp32-min per call (:710-731); head padding
  `new_zeros`/`new_full` per call (:965-972); `attn_sink` fresh
  `torch.full(-inf)` per call (:2028).
- RoPE cos/sin cache IS a high-water-mark cache but revalidates correctly:
  `seq_len > max_seq_len_cached` triggers a rebuild
  (yarn_rotary_pos_embedding.py:203-217) — a larger later microbatch cannot
  read past it.
- the only `lru_cache` in the path is the static per-device SM capability
  (:329-332); the only module globals are lazy function references
  (:314, :994).
- the per-call packed metadata (segment cu/max/offsets) is recomputed from
  the current microbatch's packed_seq_params every call
  (:766-801) — the call site is per-call-correct.

**The class is NOT exonerated — it relocates into the package internals:**
`_cudnn_dsa.indexer_forward_wrapper` / `indexer_top_k_wrapper` (the `cudnn`
package) and `flash_mla_sparse_fwd` (FlashMLA) hold their own execution
plans/workspaces, the classic first-call-sized cache site, and are not
visible from this repo. The conviction shape cauchy described — a plan or
workspace sized on the first/smaller call, then a larger microbatch's tail
rows reading garbage scores — is consistent with every constraint (larger-
than-predecessor, escalate-to-tail, garbage-content nondeterminism,
PP-independence) and CANNOT be confirmed or cleared from this repo's source.

**Conviction test + fix shape:** the non-fused reference-path leg (already
with doppler/poincare) is the discriminator — it bypasses both packages.
Corruption dies there ⇒ the workspace/plan cache inside cuDNN/FlashMLA is
convicted; the fix shape is per-shape plan revalidation or per-call workspace
allocation in the wrapper. Corruption survives ⇒ back to the visible glue
(cleared above) or the data path (poincare).

## 4d. poincare's workspace-growth refutation folded in (2026-08-13 ~07:0x CDT)

poincare's binned profiles: partition-level pattern CONFIRMED (corrupt =
larger-than-predecessor, both orders; smaller successors clean) but the
SIMPLE extent/high-water-mark model is REFUTED (onsets don't align with the
predecessor's extent by row, fraction, or datum; not a power-of-2 tile
boundary; P0 = localized spikes, P2 flat). Constraint set for any candidate:
(a) larger-than-predecessor gate, (b) composition-dependent mid-partition
onset, (c) tailward escalation, (d) nondeterminism.

**Against the visible path:** nothing in the mcore glue produces (a)-(d) —
the 4c audit stands (every scratch per-call, shape-revalidated). The one
visible composition-dependent-row structure (the indexer score chunking,
`_indexer_topk_from_score_chunks`, dsa_cudnn_kernels.py:615-689) is not on
tonight's path (multi-doc packed CP uses `_indexer_topk_multi_packed_cp_thd`,
:732-848, whose segment arithmetic I verified consistent: prefix segments,
per-doc lengths from the current cu_seqlens, max_segment_k covers all actual
segments :799-801, and the `local_seq_lens` clamp :828 never fires since
max_seqlen ≥ every doc length).

**The surviving candidate family (refined):** ~~a cuDNN/FlashMLA execution plan
or workspace cached under a coarse key~~ — **REFUTED at the design level by
the plan-cache hunt (results/CUDNN_PLAN_CACHE_HUNT.md):** every cache keys on
codegen params only, and the kernels are layout-dynamic by construction
(runtime cu_seqlens reads at utils/seqlen.py:47-86, runtime max_seqlen args,
dynamic layouts, per-shape recompile at top-k). ~~What survives of the family:
an in-kernel bug in the dynamic varlen/scheduling path itself~~ — **DEAD per
the pre-registered rule (§4e resolution): the wheel bump killed the variance
and the slot-mean together, so no in-kernel-varlen suspect is needed.** The
closed classes: p2p buffer reuse (§1), M=N seam (§2), cached-plan (§4d),
in-kernel-varlen (§4d/§4e). The defect was one raced kernel wheel.

**The discriminator (now landing per cauchy):** the full reference leg.
`_run_sparse_attention` falls back to `_unfused_absorbed_dsa_fn` (pure
PyTorch) when the fused path declines (dsa.py:147-171), and the indexer falls
back to `fused_qk_topk_naive` (pure torch.topk, dsa.py:563-609) — one config
knob (`dsa_kernel_backend`, set to "cudnn" at glm52_dsa.py:105-110; "none"/
unfused forces both fallbacks) bypasses BOTH packages with no code patch.
Corruption dies on the reference leg ⇒ the plan/workspace cache is
convicted (fix shape: per-shape plan revalidation or per-call workspace
allocation inside the wrapper — an upstream/cuDNN-bindings item). Corruption
survives ⇒ the visible glue is cleared (above), so the hunt turns to the
data/measurement path itself (poincare's domain) or the TE fused MoE kernels
(3b). NOTE: the indexer and attention share the one backend knob tonight —
separating them (indexer-only vs attention-only reference) would need a
two-line gate split if the full-reference verdict says "package internals".

## 4e. The race-class consistency question (cauchy, 2026-08-13 ~07:5x CDT) — ANSWERED

Question: is the slot-structured MEAN corruption consistent with a pure race
(biased, not just noisy), or must a race + the in-kernel-varlen suspect
coexist to explain both bias and variance?

**Answer: a pure race is sufficient — coexistence is NOT required — provided
the race's exposure is state-dependent across invocations.** The unifying
read, consistent with all four of poincare's constraints:

- The SM100 indexer kernel allocates the FULL 512-column TMEM per call and
  manages its lifetime with a manual mbarrier + dealloc
  (indexer_fwd_sm100.py:126-129, :476-478 "both epilogue WG0 and WG1 must
  arrive", :628-629, :716-723). This is exactly the #396 TMEM WAR habitat
  (serre's prior art: the box runs the retired 1.26.0+dsatopk1 frontend vs
  the Aug-3 race-fixed 1.27.0.dev pin — verified in the box dist-info).
- **(a) the larger-than-predecessor gate** comes from the allocator/TMEM
  pool state, not the kernel's logic: a larger invocation forces fresh
  cudaMalloc/TMEM territory whose content is stale or unwritten; a smaller
  successor reuses the predecessor's warm (fully-written) coverage → clean.
  P2-flat and P1-front both follow.
- **(b) the composition-dependent onset** sits at the tile-coverage
  divergence of the two layouts — a TMEM/tile structure boundary, NOT a row
  extent — which is exactly why poincare's onset does not align with the
  predecessor's extent (the simple extent model was rightly refuted; the
  tile-coverage model survives it).
- **(c) the tailward escalation**: deeper into the region the predecessor's
  coverage never reached / the race window widens with tile count.
- **(d) the nondeterminism**: the stale/raced content is timing- and
  history-dependent — run-to-run variance (0.019–0.03) is the race's
  signature, and the MEAN corruption is its structural exposure. A race is
  biased when the exposure is structural; noisy in whether each exposed tile
  is actually hit.

**What this kills:** the requirement for a second (in-kernel-varlen) suspect
to explain the bias. One state-dependent race produces bias + variance +
gate + onset + escalation.

**The decisive test is exactly the wheel bump:** if 1.27.0.dev (#396-fixed)
kills BOTH the variance and the slot-structured mean → the single race is
confirmed and the in-kernel-varlen suspect dies with it. If the variance
dies but a mean residual survives → the residual is NOT the race → reopen
the dynamic-varlen/scheduler audit (§4d) for the residual.

**RESOLUTION (2026-08-13, cauchy's reproducer verdict): CONFIRMED — single
race.** On frontend 1.27.0, the PP1/CP8 ×3 mean landed 12.30481 (golden
leg-C 12.30418, rel ~5e-5) with spread 0.00077 (25× collapse). Both the
variance AND the slot-structured mean died together → one raced wheel (the
retired 1.26.0+dsatopk1 stack's #396 TMEM WAR) accounts for the slot defect,
the same-boot nondeterminism, AND the bulk of the 0.19 gap (the slot-mean
component). The in-kernel-varlen suspect (§4d) dies with it per the
pre-registered rule. The only structural survivor is the chaos-amplifier
diffuse mean-neutral term (Gate-1 memo) — pending the padded-leg re-check
for the tail-pad component (TAIL_PAD_DSA_SOURCE_MEMO.md).

**Scope discipline — CORRECTED after cauchy's precision check (per-datum
decomposition of the 0.19, computed from parity/parity_pp2-unpadded.json vs
parity_pp1cp16-unpadded.json):** the 0.19 is NOT a diffuse structural shift —
it is ~entirely the slot-mean corruption. Token-weighted per-datum gaps
(pp2-unpadded − golden-unpadded, total +0.184 of the +0.190): datum 5
−1.084 (61% of the gap), datums 3+4+5 carry 92%; the clean docs (0, 6, 7, 8)
sit at the golden cluster (≤0.003). So the corrected split is:

- **The race owns the slot-structured SIGNED mean corruption ≈ the bulk of
  the 0.19** (the corrupted docs carry it).
- **The chaos amplifier owns the mean-neutral diffuse per-token |diff|**
  (~0.08/decile on mean-clean datums) — present everywhere, moves |diff| but
  not the signed mean, hence not the loss.
- A small signed residual on near-clean docs (datum 2 at −0.024) is the open
  sub-question — structural vs small race contribution, resolved by the
  post-wheel reading.

**The sharp wheel-bump pre-registration (cauchy's framing, confirmed):** if
the race owns the slot-mean, the post-wheel (1.27.0.dev) PP1/CP8 and PP2
means should land AT the golden cluster (~12.304, within the small residual
≤~0.02), NOT merely stabilize at ~12.50. Landing at 12.50-with-a-gap would
mean a residual structural term survives the race fix — reopen the hunt for
it. (My earlier wording — "the 0.19 is structural and stays" — was wrong on
the data and is corrected here.)

## 5. What settles it (for the morning defect hunt)

1. PP1/CP8 clean+deterministic (cauchy's in-flight boot) → localizes to PP2
   plumbing vs the shared DSA kernel path.
2. Boundary-tensor checksum dump per slot across two same-config runs
   (read-only, one boot) → before/after-boundary localization.
3. If 3a: the fix is scratch zeroing/ordering at the cited audit points —
   small, upstream-able. If 3b with golden noise: the slot-follows structure
   needs a rethink (the amplifier alone doesn't produce it).

## Confidence ranking

1. Suspect (1) refuted — high, full chain cited.
2. Suspect (2) cleared — high, cited.
3. 3a as top live suspect — moderate; it is the only class covering all six
   constraints, but the specific buffer is unproven (cuDNN internals not
   in-repo).
4. 3b — moderate as a contributor, low as the sole mechanism (slot-follows
   unexplained, golden-noise discriminator pending).
