# GLM-5.2 CP-RL profiling — logprob stitching + RL losses under THD CP32

**TL;DR (overnight run 2026-07-13): per-token logprobs and the five
token-separable RL losses (dppo / importance_sampling / ppo / cispo / dro) now
RUN and VALIDATE under THD context parallelism (branch
`jackrao/glm-131k-cp-rl`, [trainers#642](https://github.com/basetenlabs/trainers/pull/642)).
Debug-model CP1-vs-CP2 parity PASSES on all 6 loss fns (loss rel ≤ 2.8e-3;
stitched logprobs at fp-noise level, ~130× below the off-by-one floor;
grad_norm cp-ratio uniform across losses to 0.2%). Full-scale CP32 131k on the
real 800B model: CE/ppo/cispo all step at ~19–28 s / peak 102.8 GiB (unchanged
vs the SFT-only envelope), and the returned 131 072-token logprob row
reproduces the trainer's scalar loss to 1.03e-9 relative. DPO stays rejected
under CP (sequence-level pairwise nonlinearity).**

Validation of the CP-RL branch on devbox `3mlmgkq` (**4×8 B200**, birch/Weka,
same inherited stack that validated CP32 SFT — `trainers_glm_cp` clone synced
to `f14f4c23`). Mechanism notes live in `cp_explainer.md` §8; ship checklist in
`productionise_rl.md`. This file is the evidence.

What the branch changes (all in `server/…/dp_worker/api/`):

- **Stitch**: the loss fn all-gathers each CP rank's `(1, S_local)` zigzag
  logprob slice over the CP group and inverts the per-document zigzag
  permutation (`packing.thd_cp_partitioned_indices` — pure-torch mirror of
  `tex.thd_get_partitioned_indices` — + `unshard_thd_cp_rows`);
  `thd_logprobs_to_loss_fn_outputs` carves per-datum wire rows via the global
  cu_seqlens. Wire alignment identical to bshd (`wire[k] = logπ(T[k])`,
  masked → 0.0).
- **RL**: `pack_thd_cp_microbatch` packs `logprobs` / `advantages` /
  `temperatures` with the bshd pad conventions (0.0/0.0/1.0) and they zigzag-
  shard exactly like the pre-shifted labels — a position's inputs travel with
  its token, so the CE grad-scaling argument (schedule ×cp, DDP ÷(dp·cp),
  optim ÷n_tokens) transfers verbatim to any token-separable loss.

## Measurement channels

- **Loss parity** — same 3-datum batch (lengths 173/96/41, deterministic LCG
  tokens, weights mask the prompt half, advantages with interior zero holes,
  per-datum temperatures incl. T=0.7) submitted to a CP1(DP2) and a CP2(DP1)
  boot of the 2-GPU debug GLM; relative loss diff per loss_fn. fp-noise bar is
  the documented ~1e-3 (gather/FlashMLA reduction order).
- **Logprob alignment** — mean/max |Δ| between the CP2 stitched wire rows and
  the CP1 rows at the same positions, **plus an off-by-one control**: the same
  mean |Δ| with the rows shifted ±1. Any stitch misplacement anywhere reads at
  the shifted level (~0.6–0.8 nats on this model), not the noise level.
- **grad_norm cp-ratio** — `/optim_step` (lr=1e-10) grad_norm after each loss's
  forward_backward, CP2/CP1 ratio per loss fn. The gate is *uniformity across
  losses* (RL must transform under CP exactly like the already-validated CE),
  not ratio==1 — see the 4× accounting below.
- **Wire-consistency identity (131k)** — the returned logprob row must
  reproduce the scalar loss: `mean(-row[supervised]) == loss_reported`. A
  stitching error cannot survive this.

Driven by `experiment_artefacts/glm/scripts/cp_rl_parity.py` (runs all 6
loss fns + writes JSON; `--compare` applies the gates) via `smoke_cp_rl.sh`
(single-node 2-GPU boot per variant), and
`cp32_131k_rl_smoke.py` against the 4-node CP32 trainer.

## Experiment R0 — zigzag permutation exactness (GPU fuzz)

The stitch inverts the permutation with a pure-torch index builder
(CPU-testable) rather than calling `tex` — so prove it IS the kernel's
permutation: 200 random layouts (cp ∈ {1,2,4,8,16,32} × 1–6 docs × random
padded lengths), all ranks, `thd_cp_partitioned_indices` vs
`tex.thd_get_partitioned_indices` on device.

**200/200 exact-equal, zero mismatches.** (Gotcha for reruns: `tex` takes the
GLOBAL total_tokens, not S_local, and needs `import transformer_engine.pytorch`
first.)

## Experiment R1 — debug GLM CP1 vs CP2, all 6 loss fns

`smoke_cp_rl.sh {cp1,cp2}` on the leader (glm-52-debug-clean, 2 GPUs, EP2;
CP1 ⇒ DP2, CP2 ⇒ DP1), identical bytes to both. `/forward_backward` +
`/optim_step` per loss fn.

| loss_fn | loss rel Δ | logprobs mean\|Δ\| | max\|Δ\| | off-by-one floor | grad ratio CP2/CP1 | verdict |
|---|---|---|---|---|---|---|
| cross_entropy       | 3.11e-05 | 4.84e-03 | 4.27e-02 | 6.16e-01 | 4.0004 | OK |
| dppo                | 1.94e-05 | 5.95e-03 | 5.76e-02 | 7.68e-01 | 4.0021 | OK |
| importance_sampling | 2.78e-03 | 5.95e-03 | 5.76e-02 | 7.68e-01 | 3.9947 | OK |
| ppo                 | 9.98e-07 | 5.95e-03 | 5.76e-02 | 7.68e-01 | 3.9938 | OK |
| cispo               | 1.52e-03 | 5.95e-03 | 5.76e-02 | 7.68e-01 | 4.0009 | OK |
| dro                 | 1.13e-03 | 5.95e-03 | 5.76e-02 | 7.68e-01 | 4.0020 | OK |

**PARITY: PASS.** Readings:

- Aligned logprob diffs sit ~130× below the off-by-one floor — position-exact
  stitching; residual is kernel fp noise (CP1 runs bshd dense attention, CP2
  runs THD + allgather DSA — different kernels, expected ~1e-3-class noise).
  The RL rows share one max/mean because they share the forward; CE differs
  only via its weights-vs-advantages mask.
- **The 4.000 grad ratio is NOT a CP bug and is NOT new**: at this A/B shape
  CP1/DP2 splits the 3-row batch into 2 microbatches (schedule averages over
  microbatches → ×½) and Megatron DDP averages over DP=2 (×½); CP2/DP1 packs
  ONE THD microbatch and the schedule's ×cp cancels DDP's ÷(dp·cp). 2×2 = 4,
  measured 4.0004. Both factors are pre-existing bshd/THD normalization
  semantics; at the production CP32/**DP1** config (one THD microbatch) neither
  exists. The PR-relevant gate is that the ratio is **identical across all six
  losses to 0.2%** — RL gradients transform under CP exactly as CE does.
  (Side observation worth its own look someday: bshd grad magnitude ∝
  1/num_microbatches from `_split_batch` while loss_tokens normalization
  doesn't see the split — pre-existing, `productionise_rl.md` P2.)
- grad norms finite everywhere (the debug-model NaN era is over on this stack).

## Experiment R2 — CP32 131k full scale (real GLM-5.2)

Same 4-node boot recipe as the SFT validation
(`glm52-b200-pp1-ep32-cp32-131k.json`, `trainers_glm_cp` src). One synthetic
131 072-token datum (ramp vocab 30k, first half masked: weights=0 AND
advantages=0), `forward_backward` + `optim_step` per loss fn.

| loss_fn | loss | grad_norm | fb (s) | peak_alloc max (GiB) | row checks |
|---|---|---|---|---|---|
| cross_entropy | 0.041245 | 0.386 | 28.0 | 101.1 | len=131072 ✓, masked 0.0 ✓, last 0.0 ✓ |
| ppo | −0.470839 | 0.034 | 19.2 | 102.8 | same ✓ |
| cispo | 0.024290 | 0.254 | 19.1 | 102.8 | same ✓ |

- **Envelope unchanged vs SFT-only CP32**: ~19–28 s/step, peak 102.8 GiB
  (SFT probes: ~20–40 s, 99–103 GiB). The stitch all-gather (32 × 4k fp32) and
  the three extra sharded fp32 rows are noise at this scale.
- CE loss 0.0412 reproduces the known-good CP32 value — and remains the
  long-range indexer signal (the ramp's 30k period is only predictable by
  attending ≥30k back).
- **Wire-consistency identity: `row_mean_nll = 0.040853` vs
  `loss_reported = 0.040853`, rel diff 1.03e-9** over 65 535 supervised
  positions (all finite; masked half and final no-target position exactly 0.0).
- **Finding (check-design, not a bug): 16.8% of supervised logprobs are
  EXACTLY 0.0** — the 800B base in-context-learns the ramp to p≈1 and fp32
  rounds logπ to 0 (nonzero values run down to −19.5). The smoke's original
  ">95% nonzero" heuristic false-failed on this; replaced with the
  row-reproduces-loss identity. Don't treat exact-0.0 as "masked" on confident
  models.

## Boot debugging trail (infrastructure, for the record)

Three failed 4-node boots before R2, all pre-code (during the first NCCL
collective), all one root cause: **node 2 of the fresh devbox shipped without
the IB userspace stack** (`ibv_devinfo` AND `libibverbs.so` absent; nodes
0/1/3 had full rdma-core 50.0). Its 8 ranks silently fell back to NCCL TCP
while the other 24 negotiated IB:

1. attempt 1+2 — bootstrap handshake corruption on every node at once
   (`socketFinalizeAccept … wrong type 3 != 4`, NCCL 2.28.9), leader error
   pointing at peer node 2.
2. attempt 3 (socket threads serialized) — got past bootstrap, then SIGSEGV
   in `ncclProxyService` (the thread that owns IB transport setup).

Decisive because misleading: a 2-node/2-rank all-reduce PASSES (pure TCP), so
the minimal smoke does not catch it. Diagnose with `ibv_devinfo | grep -c
hca_id` on EVERY node; fix `apt-get install libibverbs1 ibverbs-providers
ibverbs-utils infiniband-diags libibmad5 libibumad3 librdmacm1t64
rdmacm-utils` on the bare node. Also note `run_trainer_node.sh` computes
`NCCL_IB_HCA` from `ibv_devinfo` per node — a node missing the CLI gets no pin
at all, compounding the mismatch. Attempt 4 (post-install, stock env): clean
boot, healthy in ~8 min.

## Reproduction

- Branch `jackrao/glm-131k-cp-rl` (base `jackrao/glm-131k-cp`), PR
  [trainers#642](https://github.com/basetenlabs/trainers/pull/642). Devbox
  clone `/root/.cache/user_artifacts/trainers_glm_cp` @ `f14f4c23`.
- Harness (mirrored in `experiment_artefacts/glm/scripts/` and
  `glm_prof/scripts/` on the shared NFS): `cp_rl_parity.py` (run + `--compare`
  gates), `smoke_cp_rl.sh` (debug-model variant boots), `cp32_131k_rl_smoke.py`
  (full-scale), plus the R0 fuzz inline in the PR conversation.
- Results JSON: `glm_prof/results/rl_parity/{cp1,cp2}.json` + driver logs;
  131k log `glm_prof/results/rl_parity/cp32_131k.log`.
- CPU-side: `server/tests/unit/dp_worker/api/test_cp_thd_slicing.py` (19
  tests — packing invariants, zigzag reference, shard→unshard round-trip,
  wire carving, phantom shard, end-to-end pack→shard→stitch→wire).

## Experiment R3 — production PP16/64k vs CP32/131k forward-loss parity

Run 2026-07-15 on devbox `tj-wlerpv3` (nodes 0–3, 4×8 B200). The same
deterministic three-datum payload and explicit loss configs were submitted
through `/forward` once per loss, first to the native PP16/EP2/CP1/DP2 layout
and then to the PP1/EP32/CP32/DP1 layout. No backward or optimizer step ran.

**PARITY: PASS — all five scalar losses satisfy
`math.isclose(pp16, cp32, rel_tol=1e-2, abs_tol=1e-5)`.**

| loss_fn | PP16/64k loss | CP32/131k loss | absolute Δ | relative Δ | verdict |
|---|---:|---:|---:|---:|---|
| dppo | 0.3050678649 | 0.3054275322 | 3.5966727e-4 | 1.1775863e-3 | PASS |
| importance_sampling | -2.2303958442 | -2.2301155583 | 2.8028588e-4 | 1.2566643e-4 | PASS |
| ppo | -0.2872851425 | -0.2871782462 | 1.0689631e-4 | 3.7209134e-4 | PASS |
| cispo | 2.1867433048 | 2.1854776811 | 1.2656237e-3 | 5.7877105e-4 | PASS |
| dro | 3.6830295471 | 3.6827317105 | 2.9783653e-4 | 8.0867267e-5 | PASS |

Evidence and provenance:

- Base checkout: `6d47915dec6b898f09979faa577def4feb69c54b`.
  Both runs used the same local patch, recorded in each artifact as binary-diff
  SHA-256 `95e9983b81a4fc54b11b130b54d509766f6006f7bca34ad039a922522533a575`.
- The patch is required: the raw branch head deterministically crashes the
  PP16 startup warmup on the phantom-only DP shard. `PackedMicrobatches.real_rows`
  includes the shape-alignment phantom, while `_run_forward_backward` receives
  lengths only for real local datums; the last stage therefore tried to convert
  one logprob row with zero `datum_lengths`. The fix trims wire outputs using
  `len(datum_lengths)`. Its regression failed before the fix, passed after it,
  and the surrounding controller/packing suites passed 66/66.
- The historical launcher also needed the branch's now-required
  `BT_TRAINER_SERVER_CONFIG_PATH`; this was experiment harness compatibility,
  separate from the phantom-row server bug.
- Config SHA-256: PP16
  `9dd19d9cb220b1427d2208a1619bc95bbcb40a687bcc960888786a9fb90f8187`;
  CP32 `a9b6ae361c7acfe7b9d935916564d6ba0b2158146be2b036eed885e9aacdc91a`.
- Canonical payload SHA-256:
  `c57d10a565f34eae3eb9fd76f25d116b4bd779be1650ed935fee9a251d7520f2`.
  Datum lengths were 65,521 / 32,779 / 8,197, temperatures 0.7 / 1.0 / 1.3,
  with explicit position-aligned terminal targets. Every loss returned exactly
  those three row shapes in both layouts; the large arrays were discarded.
- Compact artifacts:
  `experiment_artefacts/glm/data/pp16_cp32_parity/pp16_64k.json` and
  `experiment_artefacts/glm/data/pp16_cp32_parity/cp32_131k.json`.

This pass validates scalar forward-loss equivalence across these two complete
production layouts, not isolated context-parallel equivalence: PP, EP, DP, CP,
maximum sequence length, and kernels all change together. Conversely, a failed
gate would not identify CP as the cause; it would need localization across those
dimensions.
