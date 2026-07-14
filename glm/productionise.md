# GLM-5.2 131k LoRA SFT (CP32) — productionization checklist

## Open PRs + image (2026-07-13; refreshed after trainers#592 rebase)


| What                                                                                                        | Link                                                                         | State                                                                                         |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Trainer: THD CP path, logprob reconstruction, router fix, and stack lock                                   | [trainers#592](https://github.com/basetenlabs/trainers/pull/592)             | OPEN; head `98e11451` (4 commits); PR tests green, but GitHub reports `BEHIND` + review required |
| Bridge: hub-id raw-config fix + vectorized FP8 dequant + LM repin                                           | [Megatron-Bridge#19](https://github.com/basetenlabs/Megatron-Bridge/pull/19) | OPEN, head `32d432cc`, targets `trainers-main`; replaces closed #17                          |
| Megatron-LM: packed-CP indexer `q_causal_offsets` fix                                                       | [Megatron-LM#16](https://github.com/basetenlabs/Megatron-LM/pull/16)         | OPEN, head `a1fab1bb`, targets `trainers-main`; replaces closed #14                          |
| **Trainer image (current #592 head)**                                                                       | `baseten/trainers-server:jackrao-glm-131k-cp-98e1145`                        | Built green from `98e11451`; digest `sha256:aa36f62ee6251d6597fa9d23f123b83a58bb4509baf97247ff217dd95165e2e9` |
| billip: GLM-5.2 `deploy_checkpoints` fix (image override v0.24.0 + `--max-model-len 262144` + glm47 parser) | [baseten#23109](https://github.com/basetenlabs/baseten/pull/23109)           | OPEN 2026-07-10 — see `sampler_deploy_repro.md`                                               |

The #592 rebase passed the repository pre-push `make check` suite (Ruff
lint/format plus type checks). The full 4×8 B200 runtime smoke passed on
`98e11451` on 2026-07-13: two 131k THD CP32 fwd-bwd + optim steps returned
finite per-datum logprobs that reproduced scalar CE, the short text example
overfit from 10.44 to 0.10 in eight steps, and both checkpoint endpoints
completed.

The replacement PRs can be reviewed in parallel but must merge in dependency
order: LM#16 → Bridge#19 (repin its LM gitlink to the landed LM
`trainers-main` SHA if LM is squash- or rebase-merged) → a Trainers `main`
bridge-pin bump. #592 currently pins Bridge#19 `32d432cc`, which in turn pins
LM#16 `a1fab1bb`; do not delete the `jackrao/*` work branches until the
post-merge repins land, or the pinned SHAs can become unfetchable for
submodule init and image builds.

The S131K registry row and its then-current image pin are already on `main`
via trainers#616 and trainers#624. They are no longer part of the #592 diff;
after the stack chain lands, rebuild the image from the rebased #592 head and
repin the row to that artifact.

Target config: **TP1 / PP1 / EP32 / CP32 (DP=1)** on 4×8 B200 — the only
profiled config that fits 131 072 tokens. Re-measured on the upstream stack
2026-07-09: **~99 GiB peak-alloc, ~20 s per 131k fwd-bwd+optim (≈199 TPS/GPU,
~2.7× the old fork stack)**; old-fork numbers (98.2/138.2 GiB, ~56 s) in
`profiling.md`.

**STACK PIVOT 2026-07-09 → upstream GLM-5.2.** NVIDIA merged GLM-5.2 DSA + CP
into main; the curated bases now live under `trainers-main` on both
[Megatron-LM](https://github.com/basetenlabs/Megatron-LM/tree/trainers-main)
(`038760cd8`) and [Megatron-Bridge](https://github.com/basetenlabs/Megatron-Bridge/tree/trainers-main)
(`8e2d2db5`, pins that LM). The dated `trainers-main-20260907` aliases were
deleted, which automatically closed their dependent PRs. This obsoletes the
old fork PRs:

- **Megatron-LM#9** (dev sync) — **CLOSED** (superseded by the upstream rebase).
- **Megatron-LM#7** (GLM contiguous-CP) — **CLOSED** (upstream fused DSA+CP #5099/#5246
replaces it; its mHC fix was DSv4-hybrid-only; its 131k memory hardening is mostly
upstream — MoE-dispatcher-clear = `922ebcbdb` — leaving only chunked-indexer-scorer +
GLU-offset-skip as a parked memory follow-up **if CP32 131k OOMs**).
- **Megatron-LM#14** — CLOSED when its dated base was deleted; replaced by
  [LM#16](https://github.com/basetenlabs/Megatron-LM/pull/16) on
  `trainers-main`.
- **Megatron-Bridge#17** — CLOSED for the same reason; replaced by
  [Bridge#19](https://github.com/basetenlabs/Megatron-Bridge/pull/19) on
  `trainers-main`.
- **Megatron-Bridge#12** — superseded by Bridge#19, which carries its two
  surviving commits (vectorized FP8 dequant; hub-id raw-config resolution).

"Megatron-LM = zero fork commits" is DEAD: LM#16 (q_causal_offsets) is a hard
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
  a **real CP32 export from the 800B trainer succeeded**. The exact
  `98e11451` smoke wrote a 15.4 GB PEFT adapter with 116,448 HF tensors across
  all 78 layers, r16, and `base_model_name_or_path` stamped with the HF id.
  Of those, 115,200 are routed-expert tensors: the generic MLA
  `linear_fc1`/`linear_fc2` targets include routed as well as shared experts.
  The earlier note that routed experts were excluded was incorrect. Export is
  CP-safe by construction:
  all ranks gather (collectives matched), global rank 0 publishes. Harness:
  `glm_prof/scripts/export_smoke.sh` + `compare_exports.py` on the devbox.
  Remaining sub-items:
  - [x] `save_state`/resume under CP — validated on `98e11451` 2026-07-13.
    `/save_state` wrote an 81.8 GB, 32-rank DCP checkpoint at step 10 and
    `/load_state_with_optimizer` restored it successfully at step 10.
  - [x] sampler-side load of a CP-exported adapter — VALIDATED 2026-07-10 via
    the standalone-deploy repro: `VBnwM20` (pirate SFT exported by the CP
    trainer image) loads + serves through vLLM 0.24 LoRA on 8xB200, pirate
    behavior confirmed. Standalone `deploy_checkpoints` of GLM was broken
    (vLLM 0.22 image, no sparse-MLA backend on B200 + uncapped 1M seq len) —
    fix in [baseten#23109](https://github.com/basetenlabs/baseten/pull/23109);
    details in `sampler_deploy_repro.md`. The exact `98e11451` 15.4 GB adapter
    has not yet been loaded by the sampler; Loops weight-sync pairing in a live
    session is also still untested.



## P1 — needed to call it a golden config

- [x] **HF-id loading fix.** Carried by Bridge#19 `32d432cc` (+ unit tests):
  glm5_bridge resolves the raw `config.json` via `hf_hub_download` when
  `base_model` is a hub id (offline-safe — transformers has already cached
  it), loud warning when unresolvable. Verified on `q8onmd3` 2026-07-08:
  offline hub-cache resolution returns the true dims (qk_nope=192,
  qk_rope=64; transformers reports head_dim=192), and a full 4-node CP32
  boot with `base_model="zai-org/GLM-5.2-FP8"` loaded the 800B checkpoint
  to healthy in 631 s. The rebased trainers#592 pins Bridge#19 directly;
  the old `6548f05a` / `216bb411` pins are superseded.
- [x] **Registry + config plumbing.** Landed on `main` in trainers#616 and
  trainers#624: `cp` field (`ge=1`) existed from the CP stack; added the
  `Model.GLM_5_2_FP8 / B200:8 / SeqLen.S131K` row (TP1/PP1/EP32/ETP1/CP32,
  4 nodes), `load_registry` validation that `max_seq_length` splits into cp
  equal 64-token-aligned rank slices (lockstep with the server's
  `THD_CP_LOCAL_PAD_MULTIPLE`), payload/round-trip/markdown tests. The row is
  image-pinned today; it needs a follow-up repin after rebuilding the image
  from rebased trainers#592.
- [ ] **Land the branches** — replacement PRs opened 2026-07-13 against the
  live `trainers-main` bases (see PR table): LM#16 (`q_causal_offsets`, the
  CP-correctness gate) → Bridge#19 (hub-id fix + vectorized dequant + LM
  repin; supersedes #12) → a Trainers `main` bridge-pin bump. Review can happen
  in parallel; merges cannot. Remaining: post-merge gitlink repins if commits
  are rebased on merge, the Trainers pin bump, and reporting the
  `q_causal_offsets` bug upstream to NVIDIA/Megatron-LM.
- [x] **Rebuild trainer image from rebased #592.** The current-head image is
  `baseten/trainers-server:jackrao-glm-131k-cp-98e1145`, built from
  `98e11451`; digest
  `sha256:aa36f62ee6251d6597fa9d23f123b83a58bb4509baf97247ff217dd95165e2e9`.
  Its packaging work remains required: vendor the
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
  the submodule checked out. The S131K row is currently image-pinned by
  trainers#624; replace that pin with the rebuilt #592 image rather than
  relying on `DEFAULT_TRAINER_IMAGE`.
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
- [ ] **CP scope limits (CE-only v1).** Per-datum logprobs are implemented and
  full-scale validated at CP32/131k on trainers#592. DPO/RL,
  image inputs, CP+DP, and MTP remain unsupported; document those remaining
  limits.
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