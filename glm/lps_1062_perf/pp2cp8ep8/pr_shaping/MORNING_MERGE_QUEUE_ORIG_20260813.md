# MORNING MERGE QUEUE — 2026-08-13

weierstrass, for Jack's morning review. Draft only — nothing created,
pushed, or merged. Plain language; terms defined at first use. Companion
pieces: PR_DRAFT_INFRA.md, PR_DRAFT_EXPORT.md, SELF_REVIEW.md, and
lebesgue's PR_DRAFT_MN_PACKING.md (the headline M=N PR).

## The two open PRs that interact with tonight's work

**PR 987 — `fix(server): eliminate the cutlass-dsl cu13 overlay race at
resolution`** (Jack's own, open since Aug 7, branch
`jackrao/cutlass-dsl-cu13-overlay-race`).
It bit tonight's first boot on w56lorq: the cutlass-dsl 4.5.2
`libs_base`/`libs_cu13` wheels ship ~100 overlapping Python files, and
whichever overlay wins the race decides whether the sm100 kernels import.
Tonight's boot root-caused it (~12:2x CDT, NOTEBOOK) and repaired the venv
with a sequenced force-reinstall — the equivalent end state, but every
fresh venv build rolls the same dice until this merges. Zero interaction
with any LPS-1062 package.

**PR 995 — `perf(server): run the GLM fp32 LM-head GEMM on tensor cores
via scoped TF32 (LPS-1062)`** (open, GitHub reports MERGEABLE, branch
`jackrao/lps-1062-tf32-lm-head`).
Tonight's on-box TF32 patch (`configs/tf32_head_port_21d0c578.patch`,
env-gated `BT_TF32_LM_HEAD=1`, +73/−10 in `chunked_lm_head.py`) is a port
of the same change — **it duplicates PR 995; reference it, do not
re-cut.** This matters for ordering, not just hygiene: the night's
headline numbers (918 tok/s/GPU at 4 microbatches/step, 1052 at 16) were
measured on `73c24b00` **with the TF32 patch applied**, and the 645/691
anchors also included a TF32 head. The M=N PR's evidence therefore
describes bits that include PR 995's content.

## Recommended merge order

1. **PR 987 (cutlass race)** — first. Unrelated to everything below,
   actively biting every new box/venv build, already diagnosed end-to-end.
2. **PR 995 (TF32 LM head)** — before the headline PR, so that main
   matches the bits tonight's evidence was measured on.
3. **INFRA-A: (78,2) layout + DSA-only CP>1+PP>1 gate exemption**
   (21d0c578) — the ship condition is met (tonight's validation evidence +
   the guard docstring update; PR_DRAFT_INFRA.md §INFRA-A). This is the
   gate that any PP2 GLM-5.2 golden config needs.
4. **M=N headline PR** (lebesgue's package: pad-to-131k 36c3c8f4, fixtures
   b34b7aed, M=N d1c939c3, seam docs 8b0ef108; possibly poincare's
   b8d868ff — note that hunk has **not booted on hardware yet**, its
   validation status is lebesgue's to present). Lands after 995 and
   INFRA-A so its evidence chain is intact on main.
5. **INFRA-B: VPP config field + interleaved iterator fix + VPP warmup
   auto-skip** (61c41d9e + 8c13ed31 + 6fa3bfa7) — adjacent to M=N. The
   first two share one marked conflict seam in `training_runner.py` with
   the M=N PR (the 8b0ef108 comment); whichever lands second reconciles
   it. 6fa3bfa7 (poincare) auto-skips the single-datum startup warmup
   under VPP>1 with a banner — the interleaved schedule requires
   microbatches ≥ PP, so a VPP>1 boot otherwise dies at warmup unless the
   operator knows `BT_SKIP_WARMUP=1`. It reads the
   `virtual_pipeline_parallel_size` config field, so it hard-depends on
   61c41d9e and belongs in this PR, not alone. Small (+16 `backend.py` +
   69-line test pinning skip/run/env-lever both directions), inert at
   VPP=1. Review status per policy: one disposable-subagent review,
   PASS-WITH-NITS, zero blocking (the commit-message wording nit is with
   poincare; the PR-description nit is resolved by the two verbatim
   sentences below).
   All three commits are needed for the queued E2a/E2b overlap
   experiments. Honest status (lebesgue, late-night): **no VPP2 boot
   completed a driver step tonight** (0-for-5: E2a schedule-interface,
   L0 warmup-M=1 — the failure 6fa3bfa7 fixes — L0 executor contract,
   L0b memory abort, and a new named VPP2-path bug: the M=N+VPP loss
   path invokes the chunked CE loss on a chunk lacking `output_layer`,
   `chunked_lm_head.py:196`, stage-1 ranks; unfixed, poincare-domain).
   INFRA-B merges as unit-tested substrate with a named first hardware
   bug, not as a validated feature.
   - For the PR description (poincare, verbatim): "The warmup auto-skip
     keys on the REQUESTED config value
     (virtual_pipeline_parallel_size > 1). An operator setting VPP>1 on a
     (num_layers, PP, VPP) tuple absent from
     _GLM52_DSA_PIPELINE_LAYOUTS silently gets the non-interleaved
     schedule (pre-existing upstream gap) and still skips the warmup —
     conservative direction: a legal warmup is skipped, never an illegal
     one run." And: "Tests cover virtual_pipeline_parallel_size=1 (warmup
     runs) and =2 (warmup skips); the field is non-Optional int default
     1, so None is unrepresentable."
   - FUTURE item (not a queue slot, deliberately split from this hunk):
     fail-fast at apply time when a requested VPP>1 matches no layout
     tuple — the proper fix for the upstream silent-fallback gap. See
     "Future follow-ups" below: same silent-fallback family as
     SELF_REVIEW.md bringup-finding 3 and gauss's DSA top-k finding; one
     follow-up PR could close all three.
6. **EXPORT: `BT_SAVE_STATE_SYNC` toggle** (cherry-pick of db5d1826's
   `megatron_config.py` hunk as a tiny PR) — **RE-PRIORITIZED
   SHIP-CRITICAL (turing, 2026-08-14, P4/S3 finding): land with the
   headline stack, not "anytime".** The S3 save leg proved the hunk is
   NOT on the ship tree: on 73c24b00-class trees `BT_SAVE_STATE_SYNC`
   is read NOWHERE (grep = zero hits) — a silent no-op — while
   `megatron_config.py:325` hardcodes `async_save=True`, so the F1
   CP>1 async-save wedge is LIVE in any production `/save_state` on
   this tree. It fired at soak-end (iteration 61: distcp write stalled
   at 1.3 GB, trainer wedged at the closing barrier,
   `checkpointing.py:1581`; py-spy captures banked as a fresh F1
   reproduction). Every campaign boot set the env var per the standing
   rule and got false comfort; benches never call `/save_state`, so S3
   was the first real exercise. This hunk is the ship vehicle for the
   F1 workaround: the sync path completed Aug-13 in 70.8s / 537MB on
   the full 16-rank trainer (EXPORT_TEST.md §F1), and the S3 re-run
   (hunk applied on-box, sha-verified) re-validates it end-to-end on
   soak-class state. Also the fourth member of the silent-fallback
   family below (an env var accepted but never read). The F1
   escalation package — now strengthened by tonight's reproduction —
   remains parked awaiting Jack's word; outward-facing, nothing filed
   autonomously.
7. **Dispatcher host-sync cache pair — B/F (A/B GATE NOW SATISFIED;
   branches real and pushed).**
   **✅ PARITY FLAG RESOLVED — HOLD LIFTED (turing, 2026-08-14 15:00
   CDT, on kolmogorov's ruling).** The P4 S2 parity leg initially
   failed its bars (per-token 3.495 vs 1e-3; loss rel 9.11e-5 vs 1e-6)
   and STOPPED. A pre-registered four-cell noise matrix
   (S2_NOISE_MATRIX_PREREG.md) then proved the CONVENTIONAL path's own
   run-to-run nondeterminism floor covers the divergence on all three
   statistics (within-boot B/F-OFF repeats: per-token max-abs 3.7–5.5,
   95.7% pervasive, loss-rel spreads spanning 9.11e-5) — **B/F
   EXONERATED; the caches remain bitwise-exact by construction.**
   Mandatory ship-claim wording: "off-vs-on indistinguishable from the
   path's intrinsic nondeterminism (per-token and loss level); canary
   agreement ≤7e-4" — NOT "parity proven at 1e-6/1e-3": those bars
   exceed this stack's demonstrated floor at the mission topology, and
   all future parity gates on this stack must be noise-relative.
   Also ruled out during localization: env drift (boots env-symmetric,
   shas byte-exact) and the LPS-1063 tail-pad/uninit-LSE class (fix
   present + guarded at mcore 57efae08b `transformer_engine.py:1843`).
   Side finding: the same per-token noise floor exists WITHOUT the
   executor — UPSTREAM_ESCALATION_DSA_EXECUTOR.md's "conventional
   clean, executor exposes" contrast is falsified at per-token level
   and corrected (addendum §7, hertz) — read §7 before forwarding.
   Full chain: NOTEBOOK re-adjudication entries + DOPPLER_STATUS.md.
   **P4 soak result (S1, 60 steps @131k d16, B/F armed): PASS — zero
   cache fallbacks/misses/stale across all 60 steps; throughput
   deep-settle 1080–1091 (number-of-record steady-state refined to the
   1089–1103 band; the A/B's 1103 was the fresh-pair read of the same
   quantity); loss/gn trajectories training-healthy. Two flags on
   record, BOTH RESOLVED (kolmogorov's final adjudication, NOTEBOOK
   ~22:3x entry): (1) memory +2.8 GiB/60 steps, 0.8 over the
   pre-registered 2 GiB bar — RESOLVED BENIGN, not a leak, not a B/F
   concern, no merge flag: the metric split shows torch-ALLOCATED flat
   (164.6 at soak end ≈ the 163 reference class) with all growth in
   reserved-not-allocated = the allocator pool holding
   freed-but-unreturned memory; the shape is plateau-leaning (an early
   +2.1 GiB pool-warmup jump, then near-plateau with the late slope
   decelerating ~40% below the span average — a constant-rate leak
   does not decelerate); both caches are structurally bounded (the
   FIX B carrier dies per-microbatch, FIX F is FIFO-64) and telemetry
   ran clean ×60. Residual: a documented note (+0.69 GiB late-bucket
   tail; a longer run settles it). Cache entry counts are not exposed
   by telemetry — instrumentation gap, papercut pc_03d6c4817ae5.
   (2) main54 one-step gn spike 0.97, full recovery at main55, clean
   logs — an ACKNOWLEDGED bar trip (the smoothness stop fired per the
   letter) whose event is benign-class synthetic-data variance;
   telemetry cleanliness cited as the override basis; future-soak
   tooling item = a per-step watcher instead of batched reporting.**
   UPDATED 2026-08-14 (kolmogorov/bohr,
   W1c): the earlier framing ("never committed anywhere") was corrected
   during the rebuild — the pair exists on the fork as **Megatron-LM PR
   #26** (`jackrao/lps-1062-ship-bf` = trainers-main + 2 commits, DRAFT,
   MERGEABLE; byte-verified == the Aug-9 validated patches 0002+0003,
   with FIX A v2 parked default-OFF riding along). The runnable pointer
   chain is built and pushed: **bridge `jackrao/lps-1062-bf-rebuild`
   (0e356eb2)** bumps 3rdparty/Megatron-LM → 500ce306a; **trainers
   `jackrao/lps-1062-bf-rebuild` (e864115a)** bumps
   server/vendor/megatron-bridge → 0e356eb2 (off the campaign tip;
   fresh recursive clone verified end-to-end). Both gates env-gated
   default-OFF (gate-off = byte-identical path). **The A/B gate is
   satisfied — W1c (wprm693, canonical bits, fixed wheel): d16
   1088/1118 (mean 1103) vs off-arm 1038/1063 (1050.5) = +5.0%,
   distinguishable beyond spread, in the pre-registered +4–7% band; d4
   warm-class +4.9%; both gates verified ACTIVE ×16 ranks; memory flat;
   canaries clean** (full protocol/verdict: W1C_SPEC.md + NOTEBOOK
   2026-08-14 ~06:2x entry). Honest headline decomposition (bayes): the
   fresh-venv record env itself contributed +6.8% over the old 984
   record; B/F adds +5.0% on top — carried as separate terms. The
   commit-message TO-FILL slot fills with the W1c numbers. ~~Remaining
   closure item~~ RESOLVED (2026-08-14, jacobi's W1C_TRACE_READS.md):
   the mechanism read CONFIRMED leak-causality — on-arm vs off-arm
   same-night pair: `aten::nonzero` 55,328 → 1,328/step (EXACT vs the
   ~1,330 pre-registered survivor count; replay side zero = carrier
   cache hit), RoPE host 35.3 → 1.07 s, pure idle 12.00 → 4.07 s,
   −539k launches/step, micro-tax 9.51 → 0.89 s, SendRecv flat,
   traced-step span −8.3%/−8.5% (r0/r8); off-arm reproduced the fe127
   census exactly (validity). Ordering after INFRA-A/M=N stands (land
   the validated packages on their measured pin, then move the pin).
   Landing properly is also the dirty-tree cleanup path
   (DISPATCHER_CACHE_ARCHAEOLOGY.md §7).
8. **INFRA-C: diagnostic toggles** (ad39a97d `BT_PROFILE_RANKS`,
   760021be `BT_PEAK_MEM_REPORT`) — anytime; default-unchanged env gates
   with unit tests.
9. **PR 1000 (ship NCCL env defaults, draft)** — whenever ready. Measured
   +0.4% on this topology (the all-to-all is intra-node here); it is the
   apples-to-apples anchor environment, not a tonight blocker.
   Post-M=N confirmation (W1c Arm-1, 2026-08-14): ship-ON 920 vs ship-OFF
   926 at d4, |Δ| 6 ≤ spread 10 ⇒ **INERT on PP2/CP8/EP8 @131k** (the three
   knobs are NET/IB-scoped; only PP p2p rides the wire here, and it is
   wire-speed/park-dominated). Keep in the ship package — load-bearing
   +51% for golden EP16/CP16 on RoCE, harmless-neutral on PP2.

## Do NOT merge (explicit)

- **The dedekind branch's G1/G2 gate relaxations** (8a4ae08b's
  `megatron_config.py` hunks): test-branch-only by deliberate design. The
  mainline version of the gate question is INFRA-A's DSA-only exemption.
  Merging both is a textual conflict and a semantic clash — the export PR
  draft (§"Gate relationship") spells it out for reviewers.
- **The on-box TF32 patch file** — it is PR 995's content; merge 995.
- **The export test suite itself** stays on `jackrao/lps-1062-pp2-export`
  as the standing validation harness (its Qwen3 legs need the G2
  relaxation to boot); mainlining it is a deliberate follow-up decision,
  not part of this queue.

## Not merge items but morning decisions

- **F1 escalation** (CP>1 async-DCP wedge): paste-able summary ready in
  EXPORT_TEST.md §F1. Awaiting Jack's go before filing anywhere.
- ~~Dispatcher cache A/B~~ (RESOLVED 2026-08-14, W1c): the A/B ran and
  landed in-band — see updated queue item 7; the commit-message TO-FILL
  slot fills with the W1c numbers (d16 +5.0% distinguishable).

## Future follow-ups (deliberately NOT queue slots)

**Standalone work items (each its own PR):**

- **HIGH PRIORITY — trainer-venv build resolves stale gitignored
  `vendor/wheels/` over the uv.lock pin** (build-system bug, found
  during the fix campaign; NOTEBOOK ~12:2x, papercut pc_c89b5acdeed2):
  the box's trainer venv shipped the retired, race-carrying
  cudnn-frontend `1.26.0+dsatopk1` while the lock pinned
  `1.27.0.dev20260803` — the stale vendored shim won resolution because
  `make fetch-wheels` never refreshed; the sampler venv on the SAME box
  resolved 1.27.0 correctly. Consequence: the night's entire
  kernel-corruption hunt traced here — the 1.26 stack lacks the LPS-1003
  DSA race-fix series (incl. #396, the TMEM WAR race at head_dim 576/512
  = exactly GLM-5.2); the pin commit's own B300 A/B fired 12/12 on 1.26
  vs 0/12 on the new wheel, and the bumped venv passed both DSA
  regression files (5 passed / 22s). The pre-registered reproducer gate
  (PP1/CP8-fwd ×3: variance collapse + clean slot table + mean ~12.304
  ± 0.02) is the final confirmation, in flight at fold time. Fix shape:
  staleness check / fetch-wheels invalidation tied to the lock hash, so
  a vendored wheel can never silently outrank the lock again. Also note
  for whoever picks it up: tonight's perf anchors (645/691) and the
  918/1052 headlines were all measured on the STALE wheel — post-bump
  perf may move; the validation ladder re-measures.
- **DSA indexer loss reads pad rows as real queries** (gauss source memo
  `results/TAIL_PAD_DSA_SOURCE_MEMO.md` §3.4): `real_token_mask_q` is
  read at `dsa_masking.py:386-395` but set NOWHERE in mcore or the
  trainer (grep-verified), so `query_valid_rows=None` and pad query rows
  contribute garbage KL terms to the indexer auxiliary loss — a
  pad-volume-proportional corruption, worst on tail-filled partitions.
  Fix is one field: set `real_token_mask_q` from the packer's existing
  pad mask. **Materiality nuance (verified against
  `glm52_dsa.py:105-110`):** tonight's GLM-5.2 LoRA path sets
  `dsa_indexer_loss_coeff = 0.0` — the KL term is multiplied by zero, so
  tonight's runs (and the parity legs) are NOT corrupted by it. The bug
  is live on any padded THD run with the indexer loss active (i.e.
  full-parameter DSA training, or any config that doesn't zero the
  coefficient). Real bug, never-set field, worth its own small PR near
  the top of the follow-ups — but it is latent for tonight's configs,
  not a tonight correctness fire. The zero-coefficient point was flagged
  to cauchy/gauss and ACCEPTED as written (latent-not-live stands);
  probe (c) in the memo now carries the pre-registered expectation of
  zero loss-level excess, pending gauss verification.

**Silent-fallback family** — one small follow-up PR could close all
three: a requested/expected condition quietly degrades to a default
instead of failing loudly (house rule 3).

1. **VPP layout fail-fast** (from item 5): assert at apply time when a
   requested `virtual_pipeline_parallel_size > 1` matches no
   `(num_layers, PP, VPP)` tuple in `_GLM52_DSA_PIPELINE_LAYOUTS` —
   today that silently yields the non-interleaved schedule (pre-existing
   upstream gap; poincare's PR-description sentence in item 5 documents
   the conservative direction of the warmup skip in the meantime).
2. **vpp>1 + PP=1 silent no-op** (SELF_REVIEW.md bringup-finding 3): the
   layout lookup misses and the provider field stays unset — same
   fail-fast covers it.
3. **DSA top-k holder global-config fallback** (gauss source memo,
   `results/GATE1_CP_GEOMETRY_SOURCE_MEMO.md`): when both
   `packed_seq_params` and `attention_mask` are None, the DSA top-k
   carrier falls back to the process-global config object
   (`dsa.py:1599`) — a latent cross-microbatch/cross-step top-k leak for
   any future non-THD DSA caller. Not firing tonight: the THD path
   always passes `packed_seq_params`. Recommended fix: upstream assert
   (vendored Megatron-LM fork — could ride the same fork PR mechanics
   as item 7's cache pair).
- **Branch hygiene at cut time:** the bringup branch is 2 behind
  origin/main (no interacting files); the morning PRs should be cut as
  fresh branches off current main with the listed commits cherry-picked,
  per the split in PR_DRAFT_INFRA.md. Note the branch tip has moved past
  the reviewed state: the activation-offload plumbing commits (f2407a10,
  6d8b22da, 4e7d5d3e — the P2 big-win program) sit above everything in
  this queue and are experimental, not morning material. Cut at explicit
  commits, never at the branch tip.
