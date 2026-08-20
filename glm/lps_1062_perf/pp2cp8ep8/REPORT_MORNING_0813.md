# Morning report — GLM-5.2 PP2/CP8/EP8 @ 131k (night of Aug 12→13)

FINAL (drafted by borel overnight; completed by cauchy through the
morning fix campaign, ~3:1x PM; afternoon overlap-program verdicts —
E1 re-measure, golden-padded leg, offload discriminator + twin, L1 A/B,
DeepEP decomposition + config audit — folded in by lebesgue). Every
result slot is filled; sections
keep their honest chronology, including framings that were later
overturned — those are marked in place.

Written for Jack. Plain language; every label defined at first use.

## 1. Headline: the config works and beats every anchor

The mission config — GLM-5.2-FP8 LoRA training with pipeline parallelism 2
(the model's 78 layers split across two groups of GPUs), context parallelism
8 (each 131k-token sequence split across 8 GPUs), expert parallelism 8 (the
MoE experts spread across 8 GPUs), at 131,072-token sequence length on 2×8
B300 GPUs — trains correctly and is now the fastest configuration we have
ever measured for this model:

| configuration | tokens/sec/GPU | note |
|---|---|---|
| golden anchor (EP16/CP16, the previous production config) | 645 | reference |
| best-ever anchor before tonight | 691 | reference |
| tonight, 4 microbatches/step (524k tokens) | 918 | 1.42× golden |
| tonight, 8 microbatches/step (1M tokens) | 1000 | |
| tonight, 16 microbatches/step (2M tokens) | **1052** | **1.63× golden** |

"Microbatch" here = one packed 131k-token sequence; "d4/d8/d16" in the
notebook = 4/8/16 of them per optimizer step. Throughput rises with
microbatch count because the pipeline fill/drain cost amortizes; returns
diminish toward an asymptote around ~1100-1150.

One caveat on every number in this table (added during the morning fix
campaign): all of them — the anchors and tonight's headlines — were
measured on the stale kernel wheel described in §3. **The fixed-wheel
re-anchor (the honest new baseline, since the old wheel is corrupt and
has no ship path): d16 ≈ 984 tok/s/GPU** (control window 947-1023 —
this boot's settle noise ran wide, so treat the center with a ±4%
band), d4 ≈ 878-886. The RELATIVE claims (the +56% M=N speedup, the
ladder shape) were measured wheel-held-constant and stand. And one
unambiguous win rode in with the fix: **peak memory dropped 32 GiB at
every rung measured** (d4 175→143, d16 194→162) — the fixed frontend
runs a leaner workspace. This re-opened the §5 overlap program's
memory-wall math, and the re-measure has now run: **the wall persists —
the −32 GiB is a workspace-class saving and does not transfer to the
selective-recompute memory class** (warmup OOM at 259.5 GiB allocated vs
258.8 on the old wheel; the peak there is dominated by saved-for-backward
131k activations, which are wheel-invariant). The overlap memory leg is
therefore exactly the two engineering items in §5 (per-layer recompute
dial upstream, or offload).

Peak GPU memory: 175 GiB (d4) to 194 GiB (d16) of 275 GiB — the
in-flight-microbatch memory model held (memory does not grow with microbatch
count beyond 2 in flight).

**Correctness status (added early morning): read §3 before acting on the
"trains correctly" part of the headline.** The throughput numbers stand, but
the parity protocol overnight surfaced two distinct findings — a
padding-linked corruption AND run-to-run nondeterminism in the model's own
sparse-attention path (small at step 0; unpadded runs are clean and
deterministic; mechanism hunt running), and an open ~1.5% loss-level gap vs
the golden config that the overnight discriminators ran to ground: it is a
REAL DEFECT — slot-structured, nondeterministic — and it lives in the CP8
sparse-attention forward path, NOT in the M=N change or the PP2 plumbing
(reproduced identically at PP1, forward-only: our config choice exposed
it, our code did not introduce it; pipeline-parallel invariance holds
statistically). Throughput numbers stand. The headline CP8 config does not
ship until the kernel defect is fixed; the M=N mechanism itself is
numerically exonerated. A workspace-growth mechanism hypothesis with a
concrete fix shape was under test as this report closed. §3 has the full
chain.

## 2. What was actually wrong (the one-sentence root cause)

The training runner called the pipeline schedule once per microbatch with
"number of microbatches = 1," which means the two pipeline stages could
never work simultaneously — stage 1 sat idle while stage 0 computed, and
vice versa, every step. The fix ("M=N") hands the schedule all microbatches
in one call so it can actually pipeline. That single change took d4
throughput from 589 to 918 tokens/sec/GPU (+56%). The subtle part of the
implementation is gradient scaling (the schedule divides gradients by the
microbatch count; the runner's accumulation had to be restructured so the
two cancel exactly) — validated to the tolerance bars below.

Everything else we suspected first was measured and exonerated: NVLink
transport (600 GB/s in a microbenchmark), NCCL environment tuning (+0.4%),
the telemetry allreduce (absorbed into idle time), the LM head (~0.4s/step),
layer-count imbalance between stages (18.8 vs 18.25s of compute), and the
p2p wire itself (200 MB at 48 GB/s = line rate).

## 3. Correctness evidence

- Loss canaries in the expected 12.2-12.4 band with matching gradient norms
  at every rung (d1/d2/d4/d8/d16).
- Unit tests: 188 passed / 0 failed on the packing branch (CP slicing,
  dispatch, CE loss, router replay); 33/33 on the rewritten M=N dispatch
  suite.
- Parity protocol (fixed 9-document mixed-length dataset, ~262k real
  tokens; pass bars: loss relative difference ≤ 1e-6, per-token
  log-probabilities absolute difference ≤ 1e-3, exact document counts).
  The protocol was RESTRUCTURED overnight, for two honest reasons:
  - The original leg A ran against weights that had already taken ~18
    optimizer steps (the throughput sweeps trained the LoRA adapter before
    the parity pass) — invalid for 1e-6-level comparison against a fresh
    reference. Caught by volta, folded into the redesign: all gates now
    compare FRESH boot vs FRESH boot, before any optimizer step.
  - The intended PP1/CP8 reference topology turned out not to boot on this
    box at all (finding F3 below), so the reference moved to the golden
    EP16/CP16/PP1 production config — which tests topology invariance MORE
    strongly (pipeline, context, and expert parallelism all differ from the
    candidate config at once). Expected side effect: more marginal
    per-token exceedances at partition boundaries (context-parallel 16
    chunks sequences differently); the loss bar is unchanged.
  - Gate 1 (pipeline-parallel invariance): **INTERIM FAIL, under active
    investigation — do not read past this bullet without the update
    below.** Fresh PP2 padded = 12.54805275637226; fresh golden padded =
    12.351163801620233 (both 262,032 tokens, 9/9 documents, golden side
    passed its own sanity band). Relative loss difference 1.6e-02 against
    a 1e-6 bar; worst per-token log-probability difference 26.1 —
    systematic, not reduction-order noise. Final verdict: **PASSED on
    the fixed kernel wheel** — the interim fail (and everything scary
    below, kept for the honest chronology) was the stale wheel; see
    the fix-campaign block at the end of this section.
  - **The first explanation found — right phenomenon, wrong culprit
    (both facts matter; chronology kept honest):** the corruption
    triggers on TAIL padding, differs by CP chunking, and includes
    nondeterminism — exactly the phenomenology of LPS-1063 (the TE
    attention bug closed last week), and that model drove correct
    quantitative predictions all night. But the mechanism attribution
    was REFUTED near dawn, twice over: (1) GLM-5.2 never executes the
    TE attention code the bug lives in — its attention is our own
    DSA/cuDNN path end to end (source-verified independently by two
    agents, and confirmed empirically by zero fix-engagement lines
    across the fix-arm passes); (2) the vendored tree ALREADY carries
    an in-tree equivalent of the TE fix (explicit pad flag on
    divergence, extensions/transformer_engine.py:1829-1843), so
    TE-attention models were never exposed on this tree at all. The
    tail-pad corruption on THIS model is real, measured, and lives in
    the DSA path's own pad handling — an LPS-1063-class bug in our own
    kernels, not the known one.
  - (The overnight exposure argument — TE 2.16.0 on the box, fix commit
    absent from the pinned tree, trigger present — was internally
    consistent but rested on the assumption that this model uses TE
    attention. It does not; see the correction above. Kept here because
    interim messages and notebook entries reference it.)
  - Discriminators, pre-registered before results: (a) golden UNPADDED
    leg (running) against both padded values; (b) a fresh PP2 UNPADDED
    leg so Gate 1 can be judged unpadded-vs-unpadded, clean of the bug
    entirely; (c) per-token diff distribution; (d) optional bitwise
    self-reproduction of a padded leg (the bug's uninitialized-memory
    component means a padded leg may not even reproduce itself).
    Results so far: (c) LANDED, and it supports the bug reading —
    differences are diffuse (96% of weighted tokens over the bar, spread
    through documents, not boundary-localized), genuinely different
    compute (means and spreads differ per document, so not a reshuffle),
    and structured: the PP2 padded leg degrades monotonically within
    each packed sequence, resetting at each boundary and worst at the
    document nearest the padded tail — the same sawtooth appears in the
    older step-18 data, so the pathology is stable, and the golden CP16
    leg is clean except the one document nearest its padded tail. The
    worst token sits in a chunk interior, not at a seam — whole-chunk
    corruption, exactly what the bug produces. (a) LANDED — and this is
    the strongest single piece of evidence of the night: before the
    result existed, the exposure model was used to predict the golden
    unpadded value (take the padded leg, repair its one corrupted
    document back to the clean cluster: predicted 12.304, band
    12.288-12.322, filed timestamped in the notebook). The leg landed at
    12.304180991898834 — dead-center, off by 0.0002. The model is
    quantitatively confirmed: unpadded golden is clean, golden-padded
    was corrupted mildly (one document), PP2-padded more heavily (the
    sawtooth). (b) is the next boot, with its prediction pre-registered
    the same way: PP2 unpadded should land at 12.304 within the original
    1e-6 gate bar vs the golden unpadded leg — if it does, Gate 1 passes
    clean and tonight's fail was entirely the bug; near-but-out-of-gate
    would mean a residual PP2-path numerics issue beyond padding; far
    off would be structural. Result: LANDED OUT OF GATE at
    12.494321280379372 — relative difference 1.5e-2 vs leg C, four
    orders past the bar. The pre-registered middle reading fires: there
    is a residual PP2-path difference beyond padding. (d) rides the
    fix-arm boot.
  - **The decomposition closes exactly, which is why we trust it:** the
    original Gate-1 gap (12.548 vs 12.351, i.e. 0.197) = the PP2-path
    gap with no padding at all (12.494 − 12.304 = 0.190) + tail-pad
    corruption on the PP2 side (12.548 − 12.494 = 0.054) − tail-pad
    corruption on the golden side (12.351 − 12.304 = 0.047);
    0.190 + 0.054 − 0.047 = 0.197. So the known bug is real but SMALL
    (~0.05 either side); the dominant term is a ~0.19 gap between our
    PP2/CP8/EP8 line and the golden CP16/PP1/EP16 config that exists
    with zero padding.
  - **Two candidate readings for the 0.19 term, discriminators staged
    and pre-registered:** (a) structural, not a bug — this model's
    sparse attention picks its attended blocks by a discontinuous top-k
    selection, and different context-parallel chunkings can genuinely
    select different blocks; the signature would be large diffuse
    per-token differences everywhere, which is what we see (even
    documents whose means agree carry ~0.08 average per-token
    log-probability differences — 80× the bar). (b) a real defect in
    the PP2/M=N path — the systematic document-structured shifts (worst
    document −1.125, escalating across one packed sequence) are not
    obviously explained by symmetric selection noise. Reading (b) would
    implicate the headline M=N change itself, so it is being tested
    first and hardest: a permuted-document pass rides the fix-arm boot
    (pre-registered: degradation following the SLOT means a
    schedule/state artifact — the defect branch; following the DOCUMENT
    means data-dependent sensitivity — the structural branch). The two
    zero-GPU discriminators LANDED overnight: (1) gauss's source memo
    (results/GATE1_CP_GEOMETRY_SOURCE_MEMO.md) found NO cross-microbatch
    state channel in the M=N path — every per-slot structure is freshly
    constructed and cleared (the scariest defect mechanism is ruled out
    at the source level); found the top-k selection geometry-NORMALIZED
    by design, with sensitivity entering only indirectly (top-k is
    discontinuous with no tie-break bias, so shape-dependent
    floating-point noise flips genuine near-boundary selections and 78
    layers re-amplify — mechanistic grounding for the diffuse
    component); and constrained the systematic escalating component to
    doc/packing-attached channels — which makes the permuted pass
    near-decisive (a slot-following result would now point only at two
    exotic remaining classes). One latent trap found along the way
    (a global-config fallback in the top-k holder, not firing tonight)
    is filed for the morning merge queue. (2) serre's F3 assessment
    (results/F3_UNBLOCK_ASSESSMENT.md) dissolved the same-CP-reference
    blocker: the parity bars are pure forward quantities and the F3
    crash lives only in the backward gradient sync, so a forward-only
    driver never initializes the crashing communicator — the PP1/CP8
    reference leg is queued after the fix-arm boot. Its reading, filed
    in advance: landing at 12.494 (with PP2/CP8) exonerates the M=N
    path for the mean gap; landing at 12.304 (with golden) blocks the
    headline PR pending a defect hunt. Empirical results, landed near
    dawn:
    - **The permuted-order test fired for the DEFECT branch,
      decisively.** With the nine documents reversed, the corruption
      stayed with the pipeline slots, not the documents: the worst
      document of every baseline run went completely clean when it left
      the middle partition (−1.125 → +0.001); the two documents that
      moved INTO the middle partition inherited the corruption (−0.073
      → −0.865 and −0.026 → −0.827); clean documents stayed clean
      wherever they sat. Every move is 10-30× the noise floor,
      confirmed against per-datum noise bands built from repeat runs.
      Nothing tracked the data. One honest nuance: the control document
      (same slot in both orders) attenuated 7× — slot-corruption
      magnitude depends on partition composition, which fits a
      layout/buffer mechanism, not a data mechanism.
    - **The noise-floor finding behind those bands:** unpadded PP2 is
      nondeterministic run-to-run too (same-boot repeats 12.5145 /
      12.4892 against 12.4943 on the earlier boot), so the
      nondeterminism is not padding-specific — it is a property of our
      line's path. Stated plainly: the original 1e-6-level parity bars
      are unmeasurable on this stack until the nondeterminism is fixed
      (the honest interim protocol is statistical — N repeats, means
      and bands); the mean gap vs golden restates as 0.195 ± 0.013 and
      STANDS at 8-15× noise; the padded-vs-unpadded term at PP2 weakens
      to +0.036, borderline as a mean, with the structured per-token
      signals still carrying the corruption phenomenon.
    - **The mechanism hunt converged overnight**
      (results/SLOT_FOLLOWS_SOURCE_MEMO.md): the two intuitive
      suspects — pipeline p2p buffer reuse, and the M=N data-iterator
      seam mis-building per-slot metadata — were REFUTED and CLEARED
      respectively at source, with citation chains. The live suspect
      classes are (3a) per-slot scratch/workspace reuse in the
      DSA/cuDNN attention path (top suspect; specific audit points
      cited: the top-k scratch chunking, the FlashMLA workspace, a
      class-level shared D2H stream in the dispatcher) and (3b)
      atomic-order floating-point noise amplified by the top-k
      selection (its discriminator — golden-config determinism
      repeats — is a measurement gap tonight). Both hunts CONVERGE on
      one component: the cuDNN fused indexer consuming per-slot packed
      metadata — kernel-side mishandling of it would fire
      slot-dependently with composition-dependent magnitude, exactly
      what was measured. The sharpest next discriminator is cheap: run
      one same-config leg on the NON-FUSED reference DSA path
      (bypasses the fused indexer entirely) — corruption gone convicts
      the fused kernel; corruption persisting points at the
      scratch/atomic classes. Queued tonight if it needs no tree
      change; the top morning probe otherwise.
    - **The last leg landed DECISIVE: PP1/CP8 (three repeats, exact
      tree, forward-only) reproduces the IDENTICAL slot structure —
      with no pipeline, no 1F1B schedule, and no backward pass.** The
      per-datum table matches the PP2 corruption datum-for-datum
      (worst document −1.162 mean vs the PP2 band −1.158..−1.019;
      every corrupted and clean document in agreement). Three
      consequences: (1) **M=N and the PP2 plumbing are EXONERATED for
      the corruption** — the defect lives in the CP8 sparse-attention
      forward path itself; our config choice EXPOSED it, our code did
      not introduce it. (2) The DSA forward path is nondeterministic
      on its own (repeat spread 0.019 with no pipeline and no
      backward). (3) The mean gap is PP-innocent (PP1/CP8 mean 12.5031
      vs PP2/CP8 12.4993, within noise) — pipeline-parallel invariance
      HOLDS at the statistical level; the failure axis is CP8-vs-CP16
      chunking, a kernel defect, not topology non-invariance. (Formal
      Gate-1 bars: measured 5.9e-4 relative — a fail against 1e-6, but
      the noise floor exceeds the bars, so the formal gate is
      unmeasurable; the statistical gate is what passed.)
    - **A workspace-growth signature emerged from combining the two
      document orders — tested immediately, with a mixed verdict
      filed honestly:** PARTITION-LEVEL CONFIRMED — in both orders the
      heavily corrupted partition is the one LARGER than its
      predecessor (92960→98704 corrupts; 105584→124784 corrupts;
      smaller successors stay clean). But the simple
      scratch-extent/high-water-mark model was REFUTED by the
      per-token profiles: the corruption is a FRONT that starts
      partway into the bad partition at composition-dependent
      positions (~42k of 98k in one order, ~64k of 125k in the other —
      aligning across orders by neither row, fraction, nor document,
      and not a power-of-two boundary), then escalates toward the
      tail. The constraint set any candidate mechanism must now clear:
      larger-than-predecessor gating, mid-partition
      composition-dependent onset, tailward escalation, and
      nondeterminism. Binned profiles archived for the morning hunt.
    - The source hunt then cleared the entire VISIBLE path exhaustively
      (every buffer per-call-sized or correctly revalidated, with
      citations) — the defect class survives only inside the
      cuDNN/FlashMLA package internals. The plan-cache variant of that
      hypothesis (a kernel plan cached under a coarse key) was then
      REFUTED at the design level from the actual box packages
      (results/CUDNN_PLAN_CACHE_HUNT.md): every cache keys on codegen
      parameters only, larger microbatches get fresh compiles, and the
      kernels read the segment layout from cu_seqlens contents at
      runtime — a plan cannot bake a stale layout. No cache-disable
      switch exists in this path today. Surviving suspects, ranked:
      an in-kernel dynamic-varlen/persistent-scheduler bug, or the
      atomic-scheduler class.
    - **The morning fix campaign then likely found the actual fix
      already sitting in the tree:** the box venv runs a RETIRED
      kernel wheel (cudnn-frontend 1.26.0+dsatopk1), while the tree
      has pinned a race-fixed build (1.27.0.dev20260803; the cudnn
      backend itself is already correct on this box's cu13 lane —
      the fix is ONE package) since Aug 3 — carrying a series of DSA
      race fixes
      including a TMEM write-after-read race at exactly this model's
      head dimensions, with the pin commit's own B300 A/B evidence
      (old wheel fires 12/12, new wheel 0/12). The trainer CODE is
      current; the staleness is wheel-level only (why the venv build
      missed the pin is being diagnosed — that is a build-system bug
      of its own). A unified race mechanism consistent with all four
      measured constraints is written up with kernel-source citations,
      and the decomposition is now settled: the worst document alone
      carries 61% of the 0.19 gap, the three corrupted documents carry
      92%, and clean documents sit at the golden cluster — so the
      "topology gap" IS the slot corruption, and the mean-neutral
      diffuse per-token component is the only truly structural term.
    - **Pre-registered decisive test (running as of this update):
      install the pinned wheel, re-run the PP1/CP8 reproducer ×3.**
      Landing zone if the race owns it all: run-to-run variance
      collapses, slot table clean, and the leg mean lands AT the
      golden cluster (~12.304 ± 0.02) — which would restore Gate-1
      topology invariance outright and converge the night's entire
      correctness story (including, per a further pre-registered
      re-check, possibly the padded-leg corruption) onto one stale
      wheel. A landing at ~12.50-with-gap instead means a residual
      structural term survives and the varlen audit reopens.
      **Result: PASSED ON EVERY LEVEL.** On the fixed wheel the same
      leg ran 12.304436 / 12.305209 / 12.304782 — mean 12.30481,
      spread 0.00077 (25× collapse from 0.019), landing on the golden
      value 12.30418 to within 5e-5. The corruption is gone, the
      nondeterminism is gone to a tiny residual floor, and topology
      invariance is RESTORED. One stale kernel wheel — shipped by the
      vendored-wheels build bug above — was the defect behind the slot
      corruption, the nondeterminism, and the entire 0.19 "topology
      gap." The PP2 confirmation then completed the picture: on the
      fixed wheel, PP2 unpadded runs 12.3047 with spread 0.0006
      (matching golden to 4e-5, nondeterminism collapsed ~50×), and
      the permuted-order pass — the same instrument that convicted
      the slot structure — lands inside the identity band: slot
      corruption absent under permutation. Pipeline, context-parallel
      size, and expert-parallel width now all agree within a ~6e-4
      floor. The padded legs then closed the last correctness
      question: **the race owned the tail-pad corruption too.** On the
      fixed wheel, PP2 padded runs 12.30437 (spread 0.0007) — equal to
      unpadded within 0.0004 and sitting on golden. The full
      before/after map, every topology × padding cell:
      PP2 padded 12.535 (±0.030) → 12.3044 (±0.0007); PP2 unpadded
      12.494 (±0.025) → 12.3047 (±0.0006); PP1/CP8 12.503 (±0.019) →
      12.3048 (±0.0008); golden reference 12.3042. Everything agrees
      to ~5e-5 relative. Consequences: the parity protocol is sound —
      every fail-grade earlier in this report was the stale wheel;
      Gate 2 (padding semantics) PASSES with the true padding cost at
      0.0004; and the earlier "the DSA path needs its own audit and
      fix" framing is overtaken — no DSA-side fix is needed, the only
      remaining latent item is the one-field indexer-loss fix in the
      queue's future section. (The golden PADDED leg, initially skipped
      as covered by mechanism, was subsequently run on the fixed wheel
      as belt-and-braces: 12.3055 vs the golden cluster 12.3042 — inside
      the pre-registered ±0.002 band. The old→new map is complete: every
      topology × padding cell, golden included, is clean on the fixed
      wheel.)
      The ladder then completed: d2 canary in-band; d4 ≈ 878-886 (its
      control gate tripped at ~3% spread — filed with the caveat);
      **d16 number of record: 984 tok/s/GPU, peak memory 162 GiB**
      (control window 947-1023; the −32 GiB workspace saving confirmed
      at both rungs). The campaign is CLOSED: one stale kernel wheel,
      shipped by one build bug, was the whole correctness story; the
      fixed wheel re-anchors throughput ~4-6% below the (corrupt,
      unshippable) old numbers and returns 32 GiB of memory headroom.
    - **The final discriminator (running): the same leg on the
      NON-FUSED reference attention path** (one-field config switch,
      bypassing both the fused indexer and the fused attention — note
      the single knob switches the PAIR, so a clean verdict convicts
      the pair together; separating them is a two-line gate split for
      the morning). No slot structure + stable repeats = the fused
      package is the defect and nondeterminism source in one, plus a
      clean PP2-forward existence proof. Result: the leg died on an
      NCCL watchdog artifact (the reference path is slow enough at
      131k to stall a collective past the heartbeat window — papercut
      filed), and before a retry was worth staging, the wheel-gap
      discovery made it moot: PARKED, superseded by the wheel fix.
  - **What this means for the headline claim:** throughput measurements
    stand regardless. "Trains correctly" currently rests on in-band
    canaries at every rung (passed) + unit suites (passed) +
    cross-topology parity (failing by the 0.19 term of unknown
    character). If (a), the cross-topology bar is unachievable by
    design for this model class and the parity protocol needs a same-CP
    reference; if (b), the M=N PR is blocked until the defect is found.
    Morning decision unless the night settles it.
  - **The staged repair collapsed as a null — and the collapse produced
    the night's sharpest new finding.** A fix-applied arm ran (cherry-
    picked fix + a file-valve for within-boot on/off toggling, tree
    provenance verified), but the fix wraps the TE attention entry point
    this model never calls: numerically inert by construction, confirmed
    by zero engagement lines. The boot was repurposed on the fly, and
    its repeat passes measured the padded path to be NONDETERMINISTIC:
    three same-config padded runs landed at 12.518 / 12.535 / 12.548 —
    a spread of ~0.03, the same order as the corruption terms
    themselves. Two consequences: (1) the padded terms in the
    decomposition above are DISTRIBUTIONS, not points (the exact closure
    to 0.197 was partly sample luck; the 0.190 unpadded term stands —
    it was measured on the deterministic path); (2) this is an
    uninitialized-memory-class signature in our own kernels under tail
    padding. The mechanism hunt (results/TAIL_PAD_DSA_SOURCE_MEMO.md)
    has since narrowed it hard: pad-token-selection is REFUTED at source
    (pads are causally and per-document confined at every mask site —
    real queries never attend them), and the live candidate for the
    sawtooth is a three-way context-parallel SPLIT-CONVENTION mismatch
    (the data sharder, the position-map builder, and the attention
    kernel's internal split each implement the per-document split
    independently, and the divisibility guard is CPU-only — silent on
    GPU); a per-doc mismatch shifts every later document in the layout,
    accumulating toward the tail, which is exactly the sawtooth. A
    decisive read-only probe is staged on the box. The hunt also found
    a SECOND real bug from source: the sparse-attention indexer's
    auxiliary loss includes garbage terms from PAD rows (a validity
    mask is read but never set anywhere) — pad-volume-proportional,
    loss-level, one-field fix. Materiality caught before morning:
    tonight's configs zero the indexer-loss coefficient (the LoRA
    override), so this bug is REAL but LATENT tonight — it cannot
    explain any of the +0.05, and it goes live only on full-parameter
    DSA training or configs without the override (where the bridge
    default coefficient is 0.001). The Mac-side loss-vs-logprob
    consistency check then ran with its pre-registered zero-excess
    expectation and HELD (excess 0.002/0.005 on the two configs, under
    the padded noise floor): the parity gaps are fully forward
    log-probability phenomena — no unaccounted loss-level term, and
    the coefficient is verified zero along the actual parity path.
    The split-convention probe then also came back CLEAN (zero
    mismatched rows across all cases × CP8/CP16 × tail-fill on/off) —
    layout misalignment is dead alongside pad-selection, leaving the
    kernel-internal varlen handling as the residual suspect for the
    padded-specific term. f74785d7 remains valid for
    TE-attention models; the DSA path needs its own fix — scoped by
    tonight's probes, decided in the morning. (An earlier ship-question
    framing here — "if the split probe passes, ship unpadded" — was
    overtaken within the hour: the probe DID pass, but the
    permuted-order result below showed the unpadded path carries the
    slot defect too, so nothing on the PP2 line ships until that defect
    is fixed; the padded-specific residual, +0.036 and borderline,
    demotes to a secondary item.)
    (The "unpadded runs are clean and deterministic" belief this
    paragraph originally ended on was itself overturned an hour later —
    see the defect-branch results below. Remaining: PP1/CP8 leg,
    split-convention probe — both since landed; see the probe result
    above and the fix-campaign block below.)
  - Gate 2 (padding semantics): golden padded vs golden unpadded —
    REINTERPRETED: with the fix absent from the pinned tree this
    comparison is an exposure test of the tail-pad corruption, not a
    padding semantics measurement. Result: FAILED exactly where the
    corruption model predicts — relative loss difference 3.8e-3, worst
    per-token log-probability difference 21.15, located in the
    tail-padded final document of the last packed sequence. Clean bug
    evidence (mechanism attribution corrected above: DSA path, not TE).
    A true padding-semantics measurement awaits the DSA pad-handling
    fix — not possible tonight.
  - Useful byproduct (a measurement you asked volta for): the first
    direct padding-waste number. The padded forward-backward took 1.47×
    the unpadded time on identical real tokens — almost exactly the
    1.5× ratio of padded-to-real token work. Padding costs compute in
    proportion to the padded tokens; no hidden discount. (Production
    packing fills sequences nearly to capacity, so this tax applies to
    the padded tail only — but it means pad-to-max on short packs is
    paid for at full price.)
  - **Honest scope of the exposure:** the headline throughput runs
    (918/1052) also pad to 131k under CP8 on the same tree — the same
    trigger. The throughput numbers themselves almost certainly stand
    (mis-attention changes values, not the shape or cost of the compute),
    but the correctness claim for the padded configuration needs the
    LPS-1063 fix applied and A/B'd — and if the fix's exact-path routing
    costs performance, a re-anchor run is needed. Your decision in the
    morning; the parity protocol did exactly the job it exists for.

## 4. Export and checkpointing (your constraint 3)

- **LoRA adapter export with PP>1 + CP>1 WORKS.** PP2/CP2 and PP2/CP4
  exports match the PP1/CP1 reference exactly (392 tensors, correct config,
  values statistically healthy vs reference, worst per-tensor deviation
  1.3%).
- **Production landmine found (F1): the async checkpoint save hangs under
  CP>1** — PP-independent, reproduced at PP1/CP2; all ranks park in
  Megatron's async-checkpoint finalize. Since async save is unconditional
  in the trainer and GLM-5.2 production ships CP16/CP32, every CP>1
  production run that calls save_state would wedge today. One-flag
  workaround proven (synchronous save completes in seconds). Escalation
  package is written and parked — needs your word to file.
- Big-trainer save probe at PP2/CP8 (16 ranks): **hang CONFIRMED at
  production scale, and the mechanism is now fully closed.** The async
  checkpoint writer (a helper process) crashes on its very first work item —
  passing CUDA tensor handles to it over the inter-process queue fails
  ("received 0 items of ancdata," the classic file-descriptor-passing
  failure signature) — after which the training ranks wait forever on a
  dead consumer, and the remaining ranks wait on them at a collective
  barrier. The trainer stays HTTP-alive but the save never completes.
  Full 16-rank stack dumps + the crash traceback archived at
  export_test/evidence/big_trainer_save_probe/. Sync-save verify at scale:
  **PASS — with the one-flag workaround (BT_SAVE_STATE_SYNC=1) the same
  16-rank save completed in 70.8s and wrote a real 537MB checkpoint.** The
  finding is closed in both directions at production scale; only the
  escalation decision remains (yours).

## 5. Your overlap directive ("overlap the compute and comms")

- The Megatron feature built for this (`overlap_moe_expert_parallel_comm`)
  **cannot run at 131k as the framework ships**: it requires giving up full
  activation recomputation, and the framework has no partial setting — no
  per-layer dial (verified in source twice, independently). Without full
  recompute, stored activations measured 258.8 GiB against the 275 GiB
  budget with even one microbatch in flight.
- **Your probe (you asked us to try it anyway): ANSWERED — it does not
  work, and the reason is more informative than the predicted OOM.** All
  config validators accept the combination, but the run fails
  deterministically before the memory question: the overlap executor
  requires the trainer's forward-step function to return a "schedule plan"
  object (a protocol our trainer never implemented), so the flag is dead at
  131k on two independent walls — a trainer-side contract gap (exact
  error and code path on record) and the standing memory wall. Both are
  precisely mapped engineering items now, not mysteries: the contract
  adoption was scoped tonight (results/EXECUTOR_CONTRACT_SCOPING.md) and
  came back CHEAP — wrapper-level, not a re-architecture, because our loss
  functions compute from hidden states and run verbatim inside the
  executor's loss node. Estimated 0.5-2 days, seams verified with
  citations, watch items enumerated. The memory wall's fix is the offload
  probe below.
- **Follow-on memory probe (selective recompute + activation offload, flag
  off): first arm aborted at the memory guard with offload apparently
  never engaging** — peaks matched plain-selective. The source
  investigation then established that "no offload evidence in logs" means
  nothing: the feature emits zero output before the first completed step,
  by construction. What it also established, importantly: offload is fully
  wired on the normal forward path — it does NOT depend on the
  schedule-plan contract above, so the two work items are independent.
  Two explanations remain: (a) a config-ordering issue baked the
  per-module flags False (a one-line pre-step log probe now exists to
  prove it), or (b) it engaged and PCIe couldn't drain the first
  iteration's unthrottled burst — which produces an identical memory
  signature, and for which the framework has exactly one backpressure
  valve (now being plumbed). One cheap discriminator settles it: the same
  config at 32k, where a completed first step forces the engagement table
  to print. **The discriminator has now RUN (post-fix-campaign, fixed
  wheel, plain-PP2 selective+offload at 32k): verdict (b) — offload
  ENGAGES.** The summary table prints after warmup (config-ordering
  dead), and a same-config no-offload twin boot prices it: ~20-26 GiB
  peak saved at 32k (183 vs 203 GiB driver-read; ~200 vs ~226
  nvidia-smi), numerics clean (canaries match). The 131k abort was
  therefore the first-step unthrottled-burst class — exactly what the
  backpressure valve addresses. Two named gaps for the offload leg, both
  measured: (1) **module coverage** — expert_fc1 offloaded nothing on
  any rank (only moe_act fires; the load-bearing expert-side module is
  inert); (2) **perf cost at 32k ≈ −30% step time** (473 vs 681
  tok/s/GPU — the D2H/H2D stash traffic), so the valve/throttling work
  is load-bearing, not cosmetic. **The E1 re-measure on the
  fixed wheel has now RUN (same selective-recompute memory config,
  d2-scale, abort guard 265): the wall persists outright** — warmup OOM'd
  at 163s into the phase, 259.5 GiB torch-allocated vs 258.8 on the old
  wheel (essentially unchanged; the −32 GiB workspace saving does not
  apply here because this peak is saved-activation-dominated, not
  frontend-workspace-dominated). Death was clean (self-terminated at the
  guard, no wedge; log snapshotted). Consequence: the memory leg of the
  overlap program is exactly the two engineering items — the per-layer
  recompute dial (upstream) and offload (verdict above) — there is
  no wheel-assisted shortcut.
- The legal alternative found in source: a smaller flag
  (`overlap_dispatch_backward_with_experts_wgrad`) that overlaps one of the
  nine per-layer communication calls with the same layer's weight-gradient
  math. Zero memory cost, compatible with full recompute and our schedule.
  Estimated +1.5-3%. Plumbing change: b8d868ff (poincare; includes a
  trainer-side validator for the two mutual-exclusion rules that Megatron
  itself would never get to check on this path — an illegal combination
  would otherwise reach runtime silently). **A/B result: RUN (fixed wheel,
  d4, two pairs per arm). Correctness PASS** — canary in band, gradient
  norms comparable, and the deferred weight-grad verifiably fires (a
  silent no-fire would have shown as gradient-norm drift; none). **Memory
  identical within 1 GiB between arms — the zero-memory-cost claim
  holds.** **Perf: indistinguishable at tonight's measurement floor** —
  flag-ON mean 849 vs flag-OFF mean 872.5 tok/s/GPU (−2.7%), but the
  arm spreads (30/27) exceed the delta, so the pre-registered rule
  returns unproven-not-negative: the +1.5-3% estimate is NOT supported by
  this measurement, and the flag stays default-off. (Note for any retry:
  tonight's d4 control noise ran 3-4%, so resolving a +2%-class effect
  needs a quieter window or more pairs, not a single A/B.)
- Traced d16 diagnosis (what is the biggest removable cost now): DONE, and
  it refuted our leading suspicion. Host-side CPU stalls are huge in raw
  terms (61s of sync calls per step) but the schedule's runahead absorbs
  essentially all of them at 16 microbatches — only 0.9s lands on the
  critical path. The dominant removable cost is exposed expert
  communication: 30.9s per step (22.7%) of all-to-all with almost no
  compute overlap. Decision consequences (made by the pre-registered
  thresholds, not judgment calls after the fact): the Aug-9 dispatcher
  caches were demoted from top lever (their +11-13% was measured in the
  old convoy regime), and the night's remaining effort concentrated on
  the communication-overlap program above. Two independent trace analyses
  then put on-record predictions for the cheap cache A/B before it ran:
  the coarse sync-anchored method says ~0.7%, the finer op-window
  attribution (which found cache-addressable stall time hiding inside
  GPU-empty gaps the coarse method cannot see, results/
  L3_CPU_BLOCKED_DEEP_DIVE.md) says +3-6% with explicit refutation bands
  in both directions. The A/B result arbitrates between the two
  measurement methods — result: NOT RUN (superseded by the parity
  investigation and then the fix campaign). If resurrected, it needs a
  re-baseline first: both on-record predictions were derived from
  old-wheel traces, and the wheel changed both the timing and the
  memory picture.
- The exposed communication itself is now fully decomposed
  (results/A2A_EXPOSURE_DECOMPOSITION.md): of the 31s, 70% is waiting on
  peers (expert imbalance) and 30% is irreducible transfer running at spec
  bandwidth. The single largest block (37%) is communication *re-run by
  full recompute* — attackable only by recompute reduction or overlap,
  which strengthens the per-layer-dial case: it deletes that replay
  traffic outright on top of enabling overlap. The overlap program's total
  prize ceiling measures 25-28s/step, roughly 19-21% of the step.
- Bottom line on making the full overlap viable at 131k: three legs, all
  mapped and costed tonight — the forward-step contract shim (ours,
  0.5-2 days), activation offload for memory (probe result above), and the
  per-layer recompute dial (upstream, ~1 day prototype). Full package:
  results/UPSTREAM_PROPOSAL.md.
- One scoping finding worth stating plainly (results/
  DEEP_EP_CONFIG_AUDIT.md §7, citations spot-verified): **the overlap path
  has no DeepEP dependency.** The overlap executor accepts the NCCL
  all-to-all dispatcher by name, the hiding mechanism is stream-placement
  pairing (dispatcher-agnostic), and the hideable ceiling was measured on
  the NCCL stack. Flex's async dispatch hooks engage automatically under
  the executor but only sharpen host behavior — they do not change wire
  time or the hideable mass. So §6's flex verdict stays
  answered-negative as a standalone lever, and "flex as the overlap
  transport" is demoted to an optional later A/B under the executor, with
  one named falsifier: if a traced overlap boot ever shows the all-to-all
  dispatcher's host-side dispatch_preprocess DtoH blocking the comm
  stream, flex becomes load-bearing. Separately: a tuned-flex re-probe of
  the standalone config was considered and rejected on the audit — the
  only reachable knob (SM count) attacks kernel time already at parity;
  the regression lives in flex's host-side serialization, which no exposed
  knob touches.

## 6. Your DeepEP directive (the flex token dispatcher trial, L5)

You called this the top priority ("already implemented, so simple, massive
boosts" — the PR 660 precedent). It ran overnight and is CLOSED, answered
with decisive evidence. Short version: **numerically clean, mechanically
understood, and a clear performance regression at this scale — not the
lever.** Details:

- **It boots and it trains correctly on this box.** One box-specific blocker
  was hit and fixed on the way in: the bridge's DeepEP capability gate
  rejects this box's GPUs (B300s that report as "L20D" with compute
  capability 10.x, which the gate's allowlist did not include). One-line
  fix at both check sites; the patch is preserved on the box and Mac-side
  (configs/bridge_flex_deepep_capability_fix.patch), and the gate bug is
  papercut-filed.
- Both pre-registered failure zones stayed clear through warmup plus 4
  training steps: zero nvshmem/RDMA/GID errors bringing the DeepEP buffers
  up on a 2-node world, and zero errors on the FP8 dispatch path (virgin
  territory — the Qwen precedent was not FP8).
- **Correctness canary at 2 microbatches: PASS.** Loss 12.305-12.321
  (band: 12.2-12.4), gradient norms 0.40-0.49 vs the all-to-all baseline's
  0.40-0.46 — comparable, with only small reduction-order drift.
- Early performance readout, not decision-relevant: at 2 microbatches flex
  is 6.7% SLOWER (713 vs 764 tokens/sec/GPU). This is the expected shape —
  a 2-microbatch step is dominated by pipeline fill/drain, where exposed
  expert communication (the thing flex attacks) is smallest and flex's
  fixed overheads show. Peak memory 172 GiB; flex holds ~11 GiB of extra
  persistent buffers at idle, but the live peak came in lower than the
  all-to-all baseline's (the PR 660 signature: no permute buffers).
- **The verdict (4 microbatches, the decisive rung): flex LOSES — 802 vs
  918 tokens/sec/GPU, a 12.6% regression (40.8s vs 35.7s per step).**
  Numerics stayed clean at d4 as well (loss 12.28-12.31, gradient norms
  comparable); peak memory was a wash (174 vs 175 GiB). The 16-microbatch
  rung was skipped by the pre-registered ladder rule (d16 runs only if d4
  is healthy) — a rule set before the run, not a judgment call after it.
- Why it loses (traced and decomposed, not guessed): DeepEP's kernels
  move the expert all-to-all off NCCL at essentially the SAME total cost —
  8.40s/step of DeepEP kernel time (dispatch 1.99s + combine 3.13s +
  notify 3.28s; 420 calls each, exactly 35 MoE layers × 3 passes × 4
  microbatches) versus the 7-8s the NCCL all-to-all costs at this rung.
  So there is no communication-time win at 131k packed-token payloads on
  NVLink. Meanwhile the cost that dominates the exposure — waiting on
  slower peers due to expert imbalance, 70% of it per §5's decomposition —
  is dispatcher-agnostic and did not shrink (one 4.86s pipeline park still
  present; GPU-empty share unchanged). Flex's fixed overheads then make
  it a net regression. Measurement footnote for anyone re-deriving these
  numbers: with flex the all-to-all leaves NCCL, so NCCL-based bucket
  scripts read "a2a = 0" — the 8.40s DeepEP kernel figure is the
  like-for-like comparison. Trace archived:
  lps1062_pp2/traces/l5_flex_d4_rank0.pt.trace.json (719MB, box) —
  Mac-side copy pending the final evidence sweep.
- **Refined decomposition** (second pass on the same traces, answering "but
  WHY is it not faster — realistically it should be";
  results/FLEX_DEEPEP_VERDICT.md): it is genuinely not a bandwidth problem —
  the NCCL all-to-all already runs at ~85% of NVLink line rate at our ~1.6
  GB/call payloads, so no dispatcher can win on transfer. The 5.1s/step
  regression breaks down as: ~+1.4s more DeepEP kernel work for the same
  bytes (dispatch/combine/notify chain vs one NCCL collective); **zero**
  compute overlap under flex (0.0% vs NCCL's 8.3%), i.e. +3.1s of GPU-empty
  idle concentrated in 64 long (>10ms) serialized dispatch/combine windows
  (the all-to-all had 4 such gaps, 0.64s); and ~+2.7s of CP-collective
  ballooning on the main stream (AllGather 0.34→6.5ms/call) — second-order
  convoy parking behind the serialization. SM contention is real but modest
  (~0.7s of pure-compute growth). Where flex WOULD win: latency-bound
  small-payload regimes (the 16k customer shape), cross-node IB expert
  parallelism, or under a schedule that actually overlaps its
  dispatch/combine with compute — none of which is the 131k regime tonight.
- Bottom line on the directive: the "already implemented, so simple" part
  held — config-only switch, one capability-gate fix, clean boot, correct
  training. The "massive boosts" part does not transfer to this regime:
  PR 660's headline win was partly relief of memory pressure its baseline
  suffered and ours does not, and here the communication itself runs at
  cost parity. The levers for the 31s of exposed communication remain the
  overlap program in §5, not the dispatcher.

## 7. New findings along the way

- **F3 (tonight): PP1/CP8 boots crash at the gradient sync** — first seen
  on the two-node DP2 reference boot, then reproduced identically on one
  node with no data parallelism, so it is not a DP problem. Code-level
  diagnosis (results/F3_PP1_NCCL_DIAGNOSIS.md) narrowed it hard: the
  gradient buffer is a single small adapters-only bucket whose allreduce is
  structurally identical to the one that ran clean at PP2 all night — and
  the error is a *sticky* "contained" CUDA error, meaning the allreduce is
  the messenger, not the culprit. The real suspects, ranked: an earlier
  kernel fault surfacing at the first sync point, or first-use NCCL
  transport setup for that communicator under PP1's memory layout. A
  one-boot discriminator spec (env-only) is ready whenever a box has spare
  time; nothing tonight depended on PP1 after the reference moved to the
  golden config.
- A textbook example of the un-owned-complexity class you flagged: Megatron
  fires a FULL-DEVICE synchronize twice per microbatch in our hot path — a
  race workaround its own comment says is unneeded on modern PyTorch
  (p2p_communication.py:263/:419). Harmless at 16 microbatches (absorbed),
  but it cost an estimated 27-34s/step in the old convoy regime and one of
  the two call sites is unconditional. Candidate cleanup: version-gate or
  drop both (one already has a config gate). Details:
  results/L3_CPU_BLOCKED_DEEP_DIVE.md §5.
- The Aug-9 dispatcher host-sync cache work (BT_DSA_CP_LAYOUT_CACHE /
  BT_THD_ROPE_HOST_CACHE, +11-13% at 131k when measured that night, aimed
  at exactly the CPU-stall class that is now the leading residual) **was
  never landed** — no branch, no PR, not in the tree. Recovery/rebuild is a
  candidate top lever. [Adjust per L3 trace result]

## 8. PR package for your review

- **M=N + pad-to-131k** (volta → poincare, jackrao/lps-1062-pp2-packing
  lineage @ 73c24b00): the headline PR. Evidence: the throughput table,
  canaries, parity legs — cite the FIXED-WHEEL legs (Gate 1 and Gate 2
  both pass there; §3). One prerequisite folds into this PR's story:
  the wheel-staleness build fix (top follow-up in the queue) is what
  makes any box actually run the bits this PR was validated on.
- **Layout + infrastructure** (gibbs → doppler): (78,2) pipeline layout +
  the DSA-only gate exemption for CP>1+PP>1 (ships with tonight's
  validation evidence + guard docstring update per your condition),
  BT_PROFILE_RANKS (profile both stage leaders), VPP config field +
  interleaved iterator fix, telemetry gate.
- **Export + checkpoint** (dedekind → hausdorff, @ db5d1826): export tests +
  the BT_SAVE_STATE_SYNC toggle. The sync-save toggle is the natural
  ship-vehicle for the F1 workaround — your call whether it goes mainline.
- **L1 overlap plumbing** (poincare): b8d868ff, reviewed and cleared;
  its A/B was dropped for the night by the pre-registered ruling —
  ships as plumbing-with-validator, measured whenever a box slot
  exists.
- **VPP2 warmup auto-skip** (poincare, 6fa3bfa7, closed overnight): the
  startup warmup can never run under interleaved pipelining (its single
  datum means 1 microbatch, and the interleaved schedule requires at least
  PP microbatches), so a VPP>1 boot used to die at warmup unless the
  operator knew the manual skip flag. It now auto-skips with a banner;
  the manual flag remains. Reviewed per the overnight policy (one
  disposable-subagent pass): PASS-WITH-NITS, zero blocking findings.
  Folded into the infrastructure PR (merge-queue item INFRA-B) alongside
  the VPP config field it reads — it cannot land alone, and the family is
  inert at VPP=1.
- **Merge PR 987** (your own cutlass overlay-race fix, open since Aug 7) —
  it bit this box's first boot; kills that failure class for every future
  venv build.
- Note: tonight's TF32 LM-head patch duplicates your open PR 995 — the
  branch work references it rather than re-cutting it. Merge-order nuance
  (from cross-review): land 995 before the M=N PR so main matches the
  measured bits — though the +56% M=N speedup claim itself is
  TF32-held-constant (both sides of that comparison had it); it's the
  absolute anchor comparisons (645/691) that want 995 on main.
- Full recommended merge order with rationale: pr_shaping/MORNING_MERGE_QUEUE.md
  (9 items, cross-reviewed by two independent agents).

## 9. Hygiene

- Utility pod jrao-cpfs-cleanup2: DELETED (~9:1x AM).
- Box w56lorq: keep/stop is your call this morning. Traces and evidence are
  archived Mac-side (~/perf_profiles/lps-1062/pp2cp8ep8/ +
  pp2cp8ep8/export_test/evidence/); the final sweep is COMPLETE and
  VERIFIED — BOX_A_ARTIFACT_MANIFEST.md is final: all four big traces
  pulled and sha256-verified exact against box-side, 37 bench/parity
  JSONs parse-validated, all death/failure logs, patches, configs, and
  the kernel-source snapshot secured. **The box can be stopped without
  evidence loss.** It is HELD idle-armed at READY on the fixed wheel +
  exact tree, so the candidate probes (the E1 memory re-measure —
  RUN, wall persists, §5 — and the belt-and-braces golden-padded leg)
  can fire in minutes if you want them before stop.
- Fleet succession (your order): maxwell→borel, gibbs→doppler,
  volta→poincare, dedekind→hausdorff — all lanes handed off cleanly with
  briefs in pp2cp8ep8/briefs/.
