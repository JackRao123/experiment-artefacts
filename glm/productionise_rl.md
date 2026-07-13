# GLM-5.2 131k RL under CP32 — productionization checklist

RL-scope only. The base CP32 SFT productionization (stack pins, image, NaN-grad
history, registry row) lives in `productionise.md`; mechanism crib sheet in
`cp_explainer.md` §8; validation evidence in `profiling_rl.md`.

## Open PRs (2026-07-13)

| What | Link | State |
| --- | --- | --- |
| Trainer: CP logprob stitching + RL losses (dppo/is/ppo/cispo/dro) under THD CP | [trainers#642](https://github.com/basetenlabs/trainers/pull/642) | OPEN, head `f14f4c23`, base branch `jackrao/glm-131k-cp` (#592) |

Merge order: #642 lands into the #592 branch (it is a pure server-code delta on
top of it — no new submodule pins, no lock changes), so it rides #592's own
merge to main. If #592 merges first, retarget #642 to main and rebase (should
be conflict-free; it only touches `packing.py`, `megatron_controller.py`, and
the CP slicing test).

Target config unchanged: **TP1 / PP1 / EP32 / CP32 (DP=1)** on 4×8 B200.
RL adds no measurable memory or step time at 131k (peak 102.8 GiB, 19–28 s/step
— see `profiling_rl.md` R2).

What #642 ships (and deliberately does not):

- Per-token logprobs to the client under CP for CE **and** the RL losses —
  zigzag slices all-gathered + un-permuted in the loss fn, per-datum wire rows
  carved via global cu_seqlens; wire alignment identical to bshd.
- RL losses `dppo` / `importance_sampling` / `ppo` / `cispo` / `dro` under
  CP>1: token-separable, inputs pre-aligned with target_tokens, packed +
  zigzag-sharded like the pre-shifted labels; no new gradient scaling anywhere.
- **NOT: `dpo` under CP** — pairwise sequence-level nonlinearity needs a
  CP-aware per-sequence logprob all-reduce the chunked head doesn't do.
  Still hard-rejected at request time with an explanatory error.
- Wire nit (documented in-code + PR): an explicit `target_tokens` row whose
  FINAL position has a real target gets a real logprob there under CP, 0.0 at
  cp=1 (bshd drops position S−1 unconditionally). Same class as the existing
  scoring nit; harmless for GRPO-style consumers (last sampled token's target
  is beyond the rollout).

## P0 — correctness (all closed 2026-07-13)

- [x] **Zigzag stitch permutation exactness** — pure-torch
  `thd_cp_partitioned_indices` == `tex.thd_get_partitioned_indices`, 200/200
  GPU fuzz (cp 1→32, multi-doc). The unshard is provably the shard's inverse.
- [x] **CP1-vs-CP2 parity, all 6 loss fns** (debug GLM, identical batch):
  loss rel ≤ 2.8e-3; stitched logprobs mean|Δ| ≈ 5e-3 nats vs 0.62–0.77
  off-by-one floor; grad_norm cp-ratio uniform 4.000 ± 0.2% across losses
  (the 4× is pre-existing bshd microbatch×DP normalization semantics of the
  A/B shape, absent at CP32/DP1 — accounting in `profiling_rl.md` R1).
- [x] **CP32 131k full-scale smoke** (real 800B, CE + ppo + cispo, fb + optim):
  finite losses/grad-norms, envelope unchanged, and the 131 072-token wire row
  reproduces the scalar loss to **1.03e-9 rel** — the stitch cannot be wrong
  and pass this.
- [x] CPU unit coverage: `test_cp_thd_slicing.py` 19 tests (RL-field pad
  invariants, zigzag reference, round-trip, wire carving, phantom shards).

## P1 — needed before RL-on-CP32 is usable in anger

- [ ] **Review + land #642** (then it rides the #592 merge). Reviewer prep:
  `cp_explainer.md` §8 has the defense; the two questions to expect are the
  grad-ratio 4× (answer: pre-existing bshd normalization, uniform across
  losses, absent at DP1) and DPO (answer: sequence-level, follow-up).
- [ ] **Trainer image rebuild.** The published CP image
  (`…-d8d32edd`/`…-f2d10c3`) predates this branch — RL requests against it
  still hard-reject. After #642 lands on the branch, re-dispatch
  `build-trainer-server-image.yml` and update whatever row pin / per-user
  override is serving the 131k config (see `productionise.md` for the
  override-beats-row-pin trap).
- [ ] **End-to-end RL loop at 131k via loops**: sampler rollouts → per-token
  logprobs from the sampler → forward_backward(loss_fn=ppo/cispo) →
  weight-sync → sampler reload. The trainer side is validated; the loop-level
  integration (incl. the sampler prompt-logprobs OOM patch — see
  `sampler-prompt-logprobs-oom` memory / `_vllm_prompt_logprobs_patch.py`) is
  not. Watch response payload size: a 131k logprob row is ~2.5 MB JSON per
  datum over the wire.
- [ ] **Real-data RL descent probe** (the RL analogue of the ChatQA2 SFT
  parity item): a short GRPO/ppo run with real rollouts at CP32, reward
  moving the right way, vs a CP1 golden-config reference at 64k.
- [ ] **CI wiring**: `cp_rl_parity.py` + `smoke_cp_rl.sh` are one `srun` away
  from being a 2-GPU integration test (debug model, ~10 min/variant boot);
  gate = the compare-mode exit code. Add RL loss_fns to the existing CP
  integration matrix; note in `MODEL_SUPPORT.md` that GLM-5.2 131k CP32 now
  covers RL losses (DPO excluded).

## P2 — follow-ups (not blockers)

- [ ] **DPO under CP.** Design sketch: the chunked head already returns
  grad-carrying per-token logprobs; DPO needs the per-sequence masked SUM,
  which under CP is one extra all-reduce of `(logprobs · w).sum(dim=1)` over
  the CP group per pair — the nonlinearity applies after the reduce, so it is
  CP-sound; the work is making the pairwise row bookkeeping (chosen/rejected
  co-location) survive the single-row THD pack (pairs currently ride
  `row_group_size=2` on the bshd splitter, which THD bypasses).
- [ ] **bshd num_microbatches grad-scale quirk** (pre-existing, surfaced by the
  R1 accounting): on the bshd path, gradient magnitude scales with
  1/num_microbatches from `_split_batch` while the deferred `1/loss_tokens`
  normalization doesn't know about the split — i.e. the effective LR of a
  forward_backward depends on how the splitter chunked it. Invisible at
  microbatch_size=1 driver settings and at THD (always 1 microbatch); worth a
  deliberate decision + test, not an accident.
- [ ] **Exact-0.0 logprob semantics.** fp32-confident tokens produce logπ=0.0
  which is indistinguishable from the masked-position sentinel in the wire
  format (both 0.0 under datum_lengths alignment). Consumers that need the
  distinction should use the request's own mask (they sent it); if that ever
  bites, the fix is NaN→null pass-through (no datum_lengths) or a separate
  mask field — wire-format change, coordinate with SDK.
- [ ] **Stitch cost at scale**: currently one 32-way all-gather of S_global
  fp32 (~16 MB total at 131k) + a python loop over 32 index builds per
  microbatch — unmeasurable today; if per-datum logprobs are ever hot-path at
  high call rates, batch the index builds (they're deterministic per layout).

## Devbox state (2026-07-13 morning)

`3mlmgkq` (4×8 B200, birch) was left RUNNING with the CP32 trainer up (healthy,
port 8000 on the leader) — teardown deliberately skipped. Node 2 got the IB
userspace stack apt-installed (node-local — a REPLACEMENT pod loses it; recheck
`ibv_devinfo` before the next multinode run, full trail in `profiling_rl.md`).
Clone `trainers_glm_cp` @ `f14f4c23` (branch `jackrao/glm-131k-cp-rl`); parity
results under `glm_prof/results/rl_parity/`. Stop with
`truss train stop --remote baseten --job-id 3mlmgkq` when done.
