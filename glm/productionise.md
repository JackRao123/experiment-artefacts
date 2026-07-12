# GLM-5.2 131k LoRA SFT (CP32) — productionization checklist

## Open PRs + image (2026-07-10)


| What                                                                                                        | Link                                                                         | State                                                                          |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Trainer: CP zigzag port + router fix + registry row + wheels/lock                                           | [trainers#592](https://github.com/basetenlabs/trainers/pull/592)             | OPEN, rebased onto main 2026-07-10, head `7c8c093d` (12 commits)               |
| Bridge: hub-id raw-config fix + vectorized FP8 dequant + LM repin                                           | [Megatron-Bridge#17](https://github.com/basetenlabs/Megatron-Bridge/pull/17) | OPEN (head `fa04a142`; supersedes Bridge#12)                                   |
| Megatron-LM: packed-CP indexer `q_causal_offsets` fix                                                       | [Megatron-LM#14](https://github.com/basetenlabs/Megatron-LM/pull/14)         | OPEN (the CP-correctness fix; report upstream to NVIDIA too)                   |
| **Trainer image (builds green from the branch)**                                                            | `baseten/trainers-server:jackrao-glm-131k-cp-7c8c093`                        | published via `build-trainer-server-image.yml` workflow_dispatch on the branch |
| billip: GLM-5.2 `deploy_checkpoints` fix (image override v0.24.0 + `--max-model-len 262144` + glm47 parser) | [baseten#23109](https://github.com/basetenlabs/baseten/pull/23109)           | OPEN 2026-07-10 — see `sampler_deploy_repro.md`                                |


Merge order: LM#14 → Bridge#17 (repin its LM gitlink if LM commits get
rebased on merge) → trainers#592 bridge-pin bump. #592 is self-contained today
via SHA pins (all fetchable) — so do NOT delete the `jackrao/*` work branches
on the LM/Bridge forks until the post-merge repins land, or the pinned SHAs
can become unfetchable and submodule init / image builds break. Review flag: main rolled its bridge pin BACK to
`d93bb5bc` (#590/#597) for stability; this branch deliberately moves forward.
Registry S131K row intentionally NOT image-pinned yet (Jack's call, 2026-07-10).

Target config: **TP1 / PP1 / EP32 / CP32 (DP=1)** on 4×8 B200 — the only
profiled config that fits 131 072 tokens. Re-measured on the upstream stack
2026-07-09: **~99 GiB peak-alloc, ~20 s per 131k fwd-bwd+optim (≈199 TPS/GPU,
~2.7× the old fork stack)**; old-fork numbers (98.2/138.2 GiB, ~56 s) in
`profiling.md`.

**STACK PIVOT 2026-07-09 → upstream GLM-5.2.** NVIDIA merged GLM-5.2 DSA + CP
into main; Paras rebased the fork onto it as `trainers-main-20260907` on both
[Megatron-LM](https://github.com/basetenlabs/Megatron-LM/tree/trainers-main-20260907)
(`038760cd8`) and [Megatron-Bridge](https://github.com/basetenlabs/Megatron-Bridge/tree/trainers-main-20260907)
(`66d5cefc`, pins that LM). This obsoletes the old fork PRs:

- **Megatron-LM#9** (dev sync) — **CLOSED** (superseded by the upstream rebase).
- **Megatron-LM#7** (GLM contiguous-CP) — **CLOSED** (upstream fused DSA+CP #5099/#5246
replaces it; its mHC fix was DSv4-hybrid-only; its 131k memory hardening is mostly
upstream — MoE-dispatcher-clear = `922ebcbdb` — leaving only chunked-indexer-scorer +
GLU-offset-skip as a parked memory follow-up **if CP32 131k OOMs**).
- **Megatron-Bridge#12** — SUPERSEDED by Bridge#17, which carries its two surviving
commits (vectorized FP8 dequant; hub-id raw-config resolution) rebased onto
`trainers-main-20260907`. Close #12 once #17 lands.

"Megatron-LM = zero fork commits" is DEAD: LM#14 (q_causal_offsets) is a hard
correctness requirement for CP>1 until NVIDIA takes it upstream.

Devbox `tj-wp8y89q` KILLED 2026-07-10 (validation complete; everything is in
the PRs/image). Survives on shared NFS (Weka/birch cluster, any future box
there): clones `/root/.cache/user_artifacts/trainers_glm{,_cp}`, harness +
parity scripts + logs under `/root/.cache/user_artifacts/glm_prof/`
(`topk_parity2.py`, `topk_parity_multi.py`, `topk_kernel_direct.py` — the
indexer-parity harnesses that caught the upstream bug; candidates for CI).
The old nvrx `.base_version` probe-hack note is obsolete: the committed lock
resolves PyPI `nvidia-resiliency-ext==0.6.0` and torch stays 2.11+cu128 (the
feared cascade did not materialize; CI-image-proven).

## P0 — correctness blockers (nothing trains until these close)

- [ ] **NaN gradients on the first optim step — ROOT CAUSE FOUND 2026-07-09
  (`tj-wp8y89q`): a data-independent ~2.35×/layer BACKWARD gradient explosion
  in the trunk of the real 800B model.** **Paras owns the fix.** Localized with
  a per-parameter `main_grad` non-finite scan hooked at PRE-REDUCE (inside
  `finalize_model_grads`, before the DP all-reduce), POST-BWD, and PRE-OPTIM in
  `megatron_controller.py` (devbox-only monkeypatch, reverted). Findings:
  - The residual-stream gradient (measured via the LoRA-`B` adapter grads on
  attention + shared-expert projections) grows geometrically from the loss
  toward the embedding: on **real** ChatQA2 data (finite, clean, loss
  descends 8.83→5.58) the profile runs ~~1 at gL77 → **4.3e28 at gL0**, mean
  **2.35×/layer** over 78 layers. On the **synthetic ramp** the same profile
  is ~2 orders steeper and crosses fp32 max (~~3.4e38) at **gL≈7-8**, going
  `inf` there and flooding gL0-7 with NaN (`inf−inf`/`inf×0`). Forward is
  correct throughout (loss finite) — the bug is purely in the backward.
  - **Not the optimizer/LoRA/loss/data.** LoRA `B`=0 at init ⇒ `grad_A` must be
  0 on step 0; it is NaN only because the *upstream* grad is already NaN, i.e.
  NaN is generated in the frozen backbone backward, not the adapters. Onset is
  PRE-REDUCE (before DP reduce), so not a reduce artifact. Adam's per-param
  scale-invariance is why real training still shows descending loss despite a
  1e28 gradient spread; it only turns fatal on overflow→NaN.
  - **Data axis is a red herring** (same-boot A/B): real data is clean only
  because short answers give ~~89 supervised tokens; the backward runs on the
  **un-normalized token SUM** (per-token 1/loss_tokens is deferred to~~
  `execute_optim_step`~~), so ~8190 synthetic supervised tokens scale the whole
  profile up and tip it over. Even seq-512 synthetic NaNs. Deferred
  normalization is an *amplifier* (~~token-count× headroom), not the root.
  - **Depth/model-specific, NOT recompute.** The 8-layer debug checkpoint
  (single-node PP1, same full-recompute config, same synthetic ramp) shows
  FLAT grads (~1×/layer, no explosion) → recompute and the per-layer op are
  fine on a shallow/bf16 model; the explosion needs the deep real (FP8) GLM
  weights. PP-independent in mechanism (repro'd PP16; debug PP1 clean only
  because 8 layers can't compound to overflow).
  - **Excluded suspects:** bridge bump `0b663638`/LM#6 (indexer RoPE — forward,
  frozen+detached, no backward); frozen-linear dgrad fix `721e79830` (both
  branches verified numerically correct by hand); #566 recompute/comm-overlap
  (recompute refuted by debug model; overlap flags no-op at DP-optim off);
  chunked-LM-head NaN sentinel (detached on CE path).
  - **Fix target:** the per-layer backward scale in the vendored Megatron-LM
  DSA/MLA layer (`.../experimental_attention_variant/`, absorbed-MLA +
  shared-expert residual path). Next isolation step: sublayer backward-norm
  instrumentation (attn-block vs post-attn-norm vs MLP-block `dh` ratio) to
  name the amplifying sub-op, then diff against the upstream GLM-5.2 merged
  2026-07-09. Retest: one optim-enabled point per config with finite grad_norm
    - descending loss.
  - **✅ FIXED by the upstream-rebased stack — validated 2026-07-09 on**
  `tj-wp8y89q`**.** The bug lived in the OLD fork's pre-rearch DSA/MLA backend
  (`glm_dsa_fused.py`/`csa.py`); the upstream fused DSA backend replaces it.
  Validated the golden PP16 (TP1) stack rebased to bridge
  `codex/glm-cp-bridge-upstream-main` (`66d5cefc`) + Megatron-LM
  `codex/glm-dsa-cp-upstream-main` (`038760cd8`, pinned by that bridge): the
  exact synthetic-ramp probe that NaN'd on the old stack now runs finite
  grad_norm (rc=0), and lr=1e-4 gives **monotonic loss descent** (11.13 →
  10.96 → 10.95 → 10.84 over 4 steps, no poisoning). Integration notes:
  trainers controller needs **zero code edits** (imports + boot + LoRA all
  work as-is; confirmed by import smoke + full 800B boot + 4 clean optim
  steps). Remaining for Paras's trainers-main update: (a) bump the two vendored
  submodules; (b) uv.lock dep updates — new TE `b9d690e0`, FlashMLA `nv_dev`,
  and a real `nvidia-resiliency-ext>=0.6.0` (the devbox has dev33, whose
  version string fails a `>=0.6.0` gate; I patched `nvrx.py` to compare
  `base_version` **as a probe hack only** — the proper fix is the lock pin, NOT
  a PyPI `>=0.6.0` which cascades torch 2.11+cu128 → 2.13/cu13 and breaks every
  compiled kernel); (c) optional small patches — drop `linear_kv_up_proj` from
  LoRA targets when TP>1 (the new absorbed-MLA raises `NotImplementedError`
  there; golden is TP1 so unaffected), and stop setting the now-dead
  `apply_dsa_kernel_fusion`/`recompute_split_attn_mlp` provider fields
  (silently ignored upstream; the latter's loss means higher activation memory
  — watch the 131k OOM margin). Devbox `trainers_glm` left on the new stack;
  `glm_prof/REBASE_RESTORE.sh` reverts submodules to the old pins. Still TODO:
  re-run at CP32 131k on the CP stack, and the NANSCAN per-layer profile on the
  new stack to confirm the ~2.35×/layer amplification is flat (symptom is
  conclusively gone; this would be the belt-and-suspenders root-cause proof).
- [x] **CP zigzag port — CP32 131k fwd-bwd + optim_step VALIDATED 2026-07-09
  (evening, `tj-wp8y89q`).** Single-doc 131k probe: rc=0, 4 optim steps,
  monotonic descent 0.0381→0.0292→0.0278→0.0188 at lr=1e-4, finite grad norms
  (the NaN→0.0 sanitize never fired), **peak-alloc 99.11 GiB** (matches the old
  stack's 98.2), ~~**20 s/131k fwd-bwd+optim vs ~56 s on the fork stack (~~2.7×,
  ≈199 TPS/GPU)**. The 131k loss ≈0.03 is itself an indexer-correctness signal:
  the synthetic ramp has period 30000, so the supervised half is only
  predictable by attending ≥30k tokens back. Three bugs found and fixed to get
  here (first two shipped in the trainer edits, third patched in the vendored
  LM):
  1. **Warmup backward crash (trainer fix,** `megatron_controller.py`**)** — the
    LoRA freeze flipped `moe_router_enable_expert_bias` on the config AFTER
     model build, but TopKRouter captures it into `self.enable_expert_bias` at
     `__init__` (router.py:176), so `_apply_expert_bias` still ran in every
     grad-enabled recompute-forward and hit an upstream broadcast bug with any
     non-None THD padding_mask (`routing_map [n,E] & ~mask [n]` — broken for
     every n≠E, NOT just the degenerate warmup S_local=2). Fix: the freeze pass
     now also sets `module.enable_expert_bias = False` on router instances
     (right for LoRA anyway: no bias drift, no token-count accumulation).
  2. **UPSTREAM: single-doc CP indexer top-k silently wrong + 131k IMA**
    (`dsa_cudnn_kernels.py::_indexer_topk_from_score_chunks`) — the cuDNN
     `indexer_forward_wrapper` masks TOP-LEFT causal by default (row i keeps
     keys j≤i) but the packed-CP segment chunks carry ABSOLUTE causal
     positions; upstream never passes the kernel's `q_causal_offsets` arg, so
     every zigzag chunk except rank 0's front chunk scored a chunk-local
     window (~2 k keys) instead of its true prefix (up to 131 k). Result:
     wrong top-k everywhere (measured 5–25 % overlap vs exact reference; the
     8k probe "trained" on it without crashing) plus the 131 k
     `cudaErrorIllegalAddress` in `indexer_top_k` (async messenger). Fix: pass
     `q_causal_offsets=[bottom_right_key_start+row_start]` per chunk.
  3. **UPSTREAM: multi-doc packed-CP THD indexer, same defect**
    (`_indexer_topk_multi_packed_cp_thd`) — same missing `q_causal_offsets`,
     per THD segment (= `segment_k_lengths - segment_q_lengths`). Reproduced
     standalone: overlap 0.02 + the exact IMA at docs=[65536,65536] rank 31
     (this is what killed the first 2-doc trainer probe). Fixed identically.
  **Index parity is now EXACT (overlap 1.0000, zero causal/doc OOB) vs a
  pure-torch fp32 reference** at 8k/32k/131k single-doc (ranks 0/15/31) and
  multi-doc [8192,4096]/[65536,65536]/[131008,64] — harnesses
  `glm_prof/topk_parity2.py`, `topk_parity_multi.py`, `topk_kernel_direct.py`
  (top-k kernel itself verified exact; the score kernel's mask was the bug).
  Upstream's only real-kernel test for this alignment
  (`test_cudnn_indexer_topk_single_packed_cp_real_kernel_uses_bottom_right_alignment`)
  is DISABLED as flaky (cutlass `ThrMma` build issue) — all passing tests mock
  the kernel, which is how this shipped. **Report/PR both q_causal_offsets
  fixes upstream; until merged, Megatron-LM is no longer zero-fork-commits
  (1 patch,** `$SCRATCH/mlm_qcausal_fix.diff`**, also needed in the trainer
  image).** Full-FT escaped upstream notice likely because indexer-loss-ON
  takes a different scoring path; our frozen-indexer LoRA hits the fast path.
  Remaining sub-items:
  - [x] 2-doc end-to-end probe on the fixed stack — VALIDATED same evening:
    2×65536 docs in one CP row, rc=0, loss 0.0718→0.0403 (ramp-copy loss ≈0.04
    requires each doc's supervised half to attend ≥30k back through the
    multi-doc indexer — end-to-end correctness signal), peak 97.95 GiB; plus
    single-doc 131k regression re-probe clean after the multi-doc patch
    (0.0204→0.0164, 98.01 GiB, zero grad-norm sanitize events all boot).
  - [ ] loss + grad-norm parity vs CP1/golden at 64k on real data; monotonic
    descent at 131k on ChatQA2 (synthetic done).
  Port edits (devbox `trainers_glm_cp`; snapshot `$SCRATCH/cp_port_devbox.diff`):
  - `packing.py::pack_thd_cp_microbatch` — builds the GLOBAL row, pads EACH doc
  to a multiple of `2*cp_size`, returns global tensors + global cu_seqlens/
  cu_seqlens_padded (no slicing); dropped `cp_rank`, `local_pad_multiple`.
  - `megatron_controller.py::_pack_thd_cp_microbatches` — zigzag-shards the global
  row via `tex.thd_get_partitioned_indices` (mirrors bridge
  `_partition_packed_batch_for_cp`); `PackedSeqParams` now upstream-shaped
  (global cu_seqlens, dropped `cp_partition_mode`/`pad_between_seqs`,
  `cp_group`/`local_cp_size`=None).
  - `megatron_controller.py::_freeze_router_expert_bias_for_lora` — also flips
  `enable_expert_bias` on router INSTANCES (fix 1 above).
  - provider CP setup — `+cp_comm_type="allgather"` (upstream DSA+CP requirement),
  dropped `cp_partition_mode`, `sequence_packing_scheduler`→`variable_seq_lengths`.
- [x] **Weight-sync/export under CP>1.** Validated 2026-07-08 on `q8onmd3`:
  debug-model CP2-vs-CP1 step-0 exports are **bit-identical** (122/122
  tensors, lora_B all zero, adapter_config identical — init weights are
  CP-invariant so this is an exact-equality gate on the CP gather path), and
  a **real CP32 export from the 800B trainer succeeded** (1230 tensors, all
  78 layers, r16, `base_model_name_or_path` stamped with the HF id for loops
  pairing; target set = attention + shared experts, routed experts excluded —
  by design per `_lora_targets.py`). Export path is CP-safe by construction:
  all ranks gather (collectives matched), global rank 0 publishes. Harness:
  `glm_prof/scripts/export_smoke.sh` + `compare_exports.py` on the devbox.
  Remaining sub-items:
  - [ ] `save_state`/resume under CP untested.
  - [x] sampler-side load of a CP-exported adapter — VALIDATED 2026-07-10 via
    the standalone-deploy repro: `VBnwM20` (pirate SFT exported by the CP
    trainer image) loads + serves through vLLM 0.24 LoRA on 8xB200, pirate
    behavior confirmed. Standalone `deploy_checkpoints` of GLM was broken
    (vLLM 0.22 image, no sparse-MLA backend on B200 + uncapped 1M seq len) —
    fix in [baseten#23109](https://github.com/basetenlabs/baseten/pull/23109);
    details in `sampler_deploy_repro.md`. (Loops weight-sync pairing smoke in
    a live session still untested.)



## P1 — needed to call it a golden config

- [x] **HF-id loading fix.** Fixed in Bridge#12 `c578c95f` (+ unit tests):
  glm5_bridge resolves the raw `config.json` via `hf_hub_download` when
  `base_model` is a hub id (offline-safe — transformers has already cached
  it), loud warning when unresolvable. Verified on `q8onmd3` 2026-07-08:
  offline hub-cache resolution returns the true dims (qk_nope=192,
  qk_rope=64; transformers reports head_dim=192), and a full 4-node CP32
  boot with `base_model="zai-org/GLM-5.2-FP8"` loaded the 800B checkpoint
  to healthy in 631 s. trainers#592 repinned to the fixed bridge
  (`6548f05a`; the old pin `216bb411` was a local-only SHA — unfetchable).
- [x] **Registry + config plumbing.** Done in trainers#592 `d664c934`
  (2026-07-08): `cp` field (`ge=1`) existed from the CP stack; added the
  `Model.GLM_5_2_FP8 / B200:8 / SeqLen.S131K` row (TP1/PP1/EP32/ETP1/CP32,
  4 nodes), `load_registry` validation that `max_seq_length` splits into cp
  equal 64-token-aligned rank slices (lockstep with the server's
  `THD_CP_LOCAL_PAD_MULTIPLE`), payload/round-trip/markdown tests. CAVEAT:
  the row inherits `DEFAULT_TRAINER_IMAGE`, which predates the CP stack —
  blocked on the trainer-image item below before the row is publishable.
- [ ] **Land the branches** — all authored + pushed 2026-07-10 (see PR table at
  top): LM#14 (q_causal_offsets, the CP-correctness gate) + LM#15 (nvrx gate,
  stacked) → Bridge#17 (hub-id fix + vectorized dequant + LM repin; supersedes
  #12) → trainers#592 (rebased onto main, head `7c8c093d`). Remaining: reviews
  - merges (jerry review of the #592 CE/recompute merge resolutions still
  applies; also flag the deliberate bridge-pin forward vs main's #590 rollback),
  post-merge gitlink repins if commits are rebased on merge, and reporting the
  q_causal_offsets bug upstream to NVIDIA/Megatron-LM.
- [x] **Trainer image build — DONE 2026-07-10:
  `baseten/trainers-server:jackrao-glm-131k-cp-7c8c093`** (built green from the
  #592 branch via `build-trainer-server-image.yml` workflow_dispatch; also
  pre-rebase `…-53572b2`). What it took beyond the pin chain: vendor the
  `fast_hadamard_transform` cp312/cu128 wheel (its git sdist can't metadata-build
  in the CI's isolated env — legacy setup.py needs setuptools/torch/nvcc) with an
  `override-dependencies` pin to neutralize megatron-core's git source, and
  regenerate the STALE committed `server/uv.lock` (85 → 289 packages; image
  builds had been silently re-resolving every time). The lock resolves
  **PyPI** `nvidia-resiliency-ext==0.6.0` — the feared torch-2.13 cascade did
  NOT materialize (torch stays 2.11.0+cu128, all vendored wheels path-resolve;
  CI-proven), so no nvrx git pin and no `.base_version` hack needed in-image.
  Relock gotcha for next time: repo-root `uv.toml` sets `exclude-newer = "14 days"` (relative → full re-resolution on every lock), and `uv lock` needs the
  vendored bridge submodule present (editable dep) — regenerate on a box with
  the submodule checked out. NOT done: pinning the S131K registry row to this
  image (deliberately deferred — the row still inherits `DEFAULT_TRAINER_IMAGE`,
  so a loops run through the row won't get the CP stack until pinned or a
  per-user image override is set).
- [ ] **CI/benchmark wiring.** Benchmark harness auto-generates from
  `TRAINER_CONFIGS` → add a `baseline.json` entry; turn the debug-model
  CP1-vs-CP2 smoke (`scripts/smoke_cp.sh`) and the export bit-identity smoke
  (`scripts/export_smoke.sh` + `compare_exports.py`) into repeatable 2-GPU
  integration tests (`test_sft.py`'s `Parallelism` already has `cp`); update
  `MODEL_SUPPORT.md` (GLM-5.2-FP8 row: 131k CP32 + caveats).
- [ ] **One real 131k SFT run end-to-end via loops**: dataset prep at 131k,
  checkpointing, throughput sanity, sampler pairing.



## P2 — production quality (not blockers)

- [ ] **Effective batch size.** DP=1 ⇒ one sequence per fb microbatch — NOT
  per optimizer step. Batch up via grad accumulation across sequential fb
  calls before `optim_step` (accumulation contract exists — verify under CP)
  rather than DP: the profiling showed the memory constraint is tokens
  concurrently in flight per forward-backward through EP32 (CP4/DP8 was WORSE
  than golden), and accumulation is sequential ⇒ memory-flat (~56 s per 131k
  sequence ⇒ ~N·56 s per optim step). Optionally sweep CP16/DP2, CP8/DP4 for
  the wall-clock/batch tradeoff.
- [ ] **CP scope limits (CE-only v1).** No per-datum logprobs under CP (blocks
  RL/loops-style consumers later, fine for SFT), no DPO, MTP off. Document;
  widen when needed.
- [ ] **Perf polish.** 72.9 TPS/GPU @131k is workable. Levers: recompute
  tuning (full vs selective), check whether jerry's NCCL 1-channel pin (a
  DP×MoE alltoall deadlock workaround) is needed at DP=1 or just throttles,
  chase the ~1e-3 nondeterminism in the CP forward (identify the kernel;
  decide if acceptable).
- [ ] **Headroom under real load.** 138/179 GiB was one synthetic datum, no
  export in flight. Re-measure with variable-length real batches +
  weight-sync running before declaring the margin.
- [ ] Replace the grad_norm NaN→0.0 response sanitize with a real fix once the
  P0 regression lands (it currently masks NaN norms in reporting).



## Suggested order

P0-1 is unblocked-by-us (Paras). P0-3 closed 2026-07-08 (export validated at
debug and 800B scale, minus save_state/resume + sampler pairing sub-items).
P0-2 is the remaining runnable gate — spawn a box on the same cluster and
everything (clones, weights, harness) is already staged. P1 remainder:
land the branches + the trainer image build (now the gate on the S131K
registry row), then CI wiring and the e2e loops run.