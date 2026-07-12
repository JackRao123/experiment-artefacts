# Hendrycks GRPO KL benchmark — BF16-native vs NVFP4-dequant trainer

**Status: RUN 2026-07-07 (box w6lm1yw, 5×8 B200, tree `88010add` + local
`_lora_targets` fix).** Pre-registered at 20 steps; **truncated to 12 steps
per arm mid-experiment** (time call, made before either arm's data was
inspected past arm A step 9; both arms truncated identically, dataset slices
per step unchanged). Headline: **the dequant-trainer arm has ~3.5× lower
systematic mismatch on every step, but the k3 gate fails it 3/12 on
single-token tail events** — the gate metric punishes the numerically better
arm. Details below; per-token capture rerun in `capture_armB/`.

Goal: prime-rl-style per-step mismatch-KL gate table (their new-model merge
gate: every step's mismatch KL < 0.015 over 20 steps on their math env,
batch 64 — [docs/development.md#L140](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/development.md#L140)),
measured for two trainer weight sources against the same NVFP4 sampler.
**Primary readout is the Arm A vs Arm B delta**, not the absolute gate: the
0.015 anchor is loosely comparable at best (different model family, task
variant, group size, aggregation).

**Regression found & fixed en route:** on current main, `_lora_targets.py`
dispatches on `isinstance(provider, MambaModelProvider)` but the vendored
megatron-bridge refactor made `NemotronHBridge` register the *base*
`HybridModelProvider` (Mamba is now a compat subclass) ⇒ Nemotron falls to
unanchored `_DEFAULT_TARGETS` ⇒ the MTP head gets LoRA-wrapped ⇒ first
`save_weights` dies with `Expected mapping for adapter base 'mtp.layers.…'`.
Hit identically on both checkpoints (dequant exonerated). Local fix on the
box tree: dispatch on `HybridModelProvider`. Needs upstreaming + a
regression test.

## Arms

| arm | sampler (behavior policy) | trainer (target policy) |
|---|---|---|
| A | NVFP4 vLLM (`…Ultra-550B-A55B-NVFP4`) | native BF16 ckpt (`…Ultra-550B-A55B-BF16`) |
| B | NVFP4 vLLM (same) | NVFP4→bf16 dequant ckpt (`/root/.cache/user_artifacts/nemotron3-ultra-550b-nvfp4-dequant-bf16`) |

Hypothesis: Arm B (trainer numerically closer to the sampler's NVFP4 weights)
shows lower mismatch KL than Arm A; the earlier single-trace probe (above)
bounds the NVFP4-vs-bf16 quant term at k3 ≈ 5e-4–4e-3, so the expected arm
delta is order ~1e-3 — detectable only because each step averages ~512
completions.

## Config (identical across arms)

- Dataset: Hendrycks MATH (`PrimeIntellect/Hendrycks-Math`), fixed slice +
  order, same sampling seed both arms; graded with `math_verify`.
- GRPO: 20 steps × batch 64 problems × group 8 = 512 completions/step;
  `max_tokens=2048`, T=1.0, top_p=1.0, thinking off; LoRA r=32, lr 4e-5,
  `importance_sampling` loss; local weight sync each step.
- Trainer: 4×8 B200, TP8/PP4/EP8, `max_seq_len=4096`, seq-trim main.
- Sampler: 1×8 B200 TP8, golden B200 config with `max_loras=1`,
  `MAX_NUM_SEQS=512`.
- KL probe: each step, after sampling and before optim_step, all 512
  sequences teacher-forced through trainer `POST /forward` (cross_entropy);
  per-token metrics over sampled completion tokens, same conventions as the
  256k probe above (k3 estimates KL(behavior‖target); wire alignment
  `wire[plen−1 : plen−1+clen]`; clip band [0.8, 1.2]).
- Trainer target policy at step N = base + LoRA-after-(N−1)-optim-steps ==
  exactly the weights the sampler serves at step N (post-sync), so the probe
  measures engine/numerics mismatch, not policy lag.

## Per-step gate tables (the prime-rl-shaped readout)

### Arm A — trainer = native BF16

| step | k3 | mean \|Δlp\| | max \|Δ\| | ESS/N | clip% | reward | degen% | datums | comp tok (mean/p95) | <0.015 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00422 | 0.0241 | 4.04 | 0.9921 | 3.5% | 0.695 | 92% | 40 | 638/2048 | PASS |
| 1 | 0.00587 | 0.0259 | 5.25 | 0.9893 | 3.6% | 0.699 | 80% | 104 | 663/2048 | PASS |
| 2 | 0.00698 | 0.0240 | 6.78 | 0.9909 | 3.4% | 0.717 | 77% | 120 | 756/2048 | PASS |
| 3 | 0.00524 | 0.0233 | 5.73 | 0.9924 | 3.3% | 0.676 | 84% | 80 | 594/2048 | PASS |
| 4 | 0.00393 | 0.0216 | 4.23 | 0.9920 | 3.1% | 0.650 | 86% | 72 | 614/2048 | PASS |
| 5 | 0.00402 | 0.0227 | 3.80 | 0.9918 | 3.2% | 0.586 | 89% | 56 | 632/2048 | PASS |
| 6 | 0.00572 | 0.0229 | 6.22 | 0.9915 | 3.2% | 0.721 | 88% | 64 | 630/2048 | PASS |
| 7 | 0.00442 | 0.0226 | 3.52 | 0.9913 | 3.2% | 0.729 | 91% | 48 | 655/2048 | PASS |
| 8 | 0.00412 | 0.0241 | 2.76 | 0.9916 | 3.4% | 0.738 | 88% | 64 | 695/2048 | PASS |
| 9 | 0.00463 | 0.0234 | 3.67 | 0.9905 | 3.3% | 0.746 | 83% | 88 | 553/2048 | PASS |
| 10 | 0.00420 | 0.0218 | 3.88 | 0.9928 | 3.1% | 0.717 | 98% | 8 | 483/1714 | PASS |
| 11 | 0.00475 | 0.0253 | 3.35 | 0.9913 | 3.6% | 0.631 | 89% | 56 | 626/2048 | PASS |

### Arm B — trainer = NVFP4-dequant-bf16

| step | k3 | mean \|Δlp\| | max \|Δ\| | ESS/N | clip% | reward | degen% | datums | comp tok (mean/p95) | <0.015 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00146 | 0.0148 | 2.47 | 0.9971 | 1.6% | 0.699 | 84% | 80 | 641/2048 | PASS |
| 1 | 0.00221 | 0.0164 | 4.93 | 0.9962 | 1.8% | 0.691 | 78% | 112 | 669/2048 | PASS |
| 2 | 0.07104 | 0.0149 | 10.18 | 0.9966 | 1.6% | 0.709 | 81% | 96 | 737/2048 | **FAIL** |
| 3 | 0.00144 | 0.0145 | 2.00 | 0.9969 | 1.6% | 0.680 | 84% | 80 | 612/2048 | PASS |
| 4 | 1.22556 | 0.0134 | 12.88 | 0.9974 | 1.5% | 0.662 | 88% | 64 | 624/2048 | **FAIL** |
| 5 | 0.00137 | 0.0138 | 2.26 | 0.9973 | 1.5% | 0.602 | 92% | 40 | 625/2048 | PASS |
| 6 | 0.00133 | 0.0140 | 1.68 | 0.9973 | 1.5% | 0.721 | 83% | 88 | 631/2048 | PASS |
| 7 | 0.00158 | 0.0141 | 2.13 | 0.9970 | 1.6% | 0.725 | 88% | 64 | 646/2048 | PASS |
| 8 | 0.00151 | 0.0151 | 2.27 | 0.9969 | 1.6% | 0.736 | 83% | 88 | 698/2048 | PASS |
| 9 | 0.00140 | 0.0142 | 1.41 | 0.9971 | 1.6% | 0.754 | 86% | 72 | 552/2048 | PASS |
| 10 | 0.65110 | 0.0124 | 12.00 | 0.9976 | 1.4% | 0.717 | 97% | 16 | 489/2048 | **FAIL** |
| 11 | 0.00283 | 0.0155 | 6.12 | 0.9969 | 1.7% | 0.629 | 92% | 40 | 647/2048 | PASS |

## Summary (headline numbers)

| arm | mean k3 (clean steps) | max k3 | steps ≥ 0.015 | mean \|Δlp\| (all steps) | gate verdict |
|---|---|---|---|---|---|
| A (native BF16 trainer) | 0.00484 (12/12 clean) | 0.0070 | 0/12 | 0.0235 | PASS |
| B (NVFP4-dequant trainer) | 0.00173 (9 clean steps) | 1.2256 | 3/12 | 0.0144 | FAIL (tail-driven) |
| **B − A delta (bulk)** | **−0.0031 (B ≈ 3.5× lower)** | — | — | **−0.0091 (B ≈ 1.6× lower)** | — |

**The two-regime result.** Arm B (dequant trainer) has *lower systematic
mismatch on every one of the 12 steps* — mean |Δlp| 0.0144 vs 0.0235, ESS/N
0.997 vs 0.991, clip 1.5–1.8% vs 3.1–3.6%, clean-step k3 0.0013–0.0028 vs
0.0039–0.0070. But it fails the literal gate 3/12 because three steps each
caught **one single catastrophic-disagreement token** (max |Δ| 10.2 / 12.9 /
12.0 nats; sign = trainer scores the sampled token as ~e^10–e^13 *less*
likely). The arithmetic closes exactly: e^10.18/377,549 ≈ 0.070 (step 2),
e^12.88/319,404 ≈ 1.23 (step 4), e^12.00/250,116 ≈ 0.65 (step 10) — remove
one token per step and every step is ≈ 0.0014. k3's e^{−r} term makes a
"<0.015 every step" gate a test of Poisson tail arrivals, not of systematic
engine mismatch. Arm A's tail topped out at 6.8 nats (contribution ~3e−3,
sub-gate), so its 12/12 PASS partly reflects tail luck, not only better
numerics — its *bulk* mismatch is 1.6–3.5× worse.

Paired per-step delta (same problems + sampling seed both arms; identical
rollouts at step 0, near-identical after):

| statistic | value |
|---|---|
| steps where B mean \|Δlp\| < A | 12/12 |
| steps where B k3 < A k3 | 9/12 (the 3 exceptions are the tail steps) |
| step-0 (identical rollouts) k3 | B 0.00146 vs A 0.00422 |

## Run health (context for the KL numbers, not results)

| arm | wall-clock/step (mean) | sample tok/s | KL probe s/step | save_weights s (steady) | pass@1 before → after 12 steps (n=64) |
|---|---|---|---|---|---|
| A | 214 s | ~4,900 | 83–110 | ~2.0 | 0.656 → 0.672 |
| B | 218 s | ~5,000 | 87–104 | ~2.0 | 0.656 → 0.641 |

Reward trajectories match closely step-for-step (both peak ~0.75 at step 9,
dip ~0.60 at step 5); pass@1 movement after 12 steps is noise-level in both
directions, consistent with the June finding that 12 steps barely moves
held-out accuracy. Degeneracy 77–98%/step (a 550B model on Hendrycks with
group 8): 8–120 datums survive per step, enough for real optim steps but the
training signal rides on a small slice of problems.

## Caveats (pre-registered, plus what actually bit)

- k3 is heavy-tail dominated (this doc measured ~100× per-trace spread) —
  **confirmed with force**: single tokens flipped three of Arm B's steps from
  0.0014 to 0.07–1.23. The paired bulk statistics (mean |Δlp|, ESS, clip) are
  the defensible comparison; step-level k3 is a tail-arrival counter.
- 0.015 anchor caveats: prime-rl's gate is Qwen-family, their math env,
  batch 64 × group ~16, batch-mean aggregation, 20 steps (we ran 12). Treat
  gate pass/fail as directional.
- Reward/degeneracy differences between arms would confound the KL comparison
  (different surviving-datum mix); the KL probe deliberately scores **all**
  512 sampled completions, not just non-degenerate datums, to avoid that.
- LoRA is trained on top of *different* base weights per arm, so trajectories
  diverge after step 0 even with identical seeds — step 0 is the only pure
  same-policy comparison (and it favors B, 0.0015 vs 0.0042); later steps
  compare each arm's own self-consistent loop (the production-relevant
  question).
- For the RL update itself the tail tokens are benign-direction: r ≈ −13 ⇒
  importance weight e^r ≈ 3e−6 ⇒ the token's gradient contribution is
  crushed (and clipped). The dangerous direction (trainer ≫ sampler) did not
  appear at magnitude.
- Truncation 20→12 steps was a mid-run time decision; both arms identical,
  per-step data slices unaffected. Comp-token p95 pinned at the 2048 cap
  every step ⇒ a meaningful fraction of rollouts truncate; mismatch numbers
  are for this ~640-mean-token regime.

## Per-token capture rerun (Arm B, 12 steps, fresh LoRA, same seeds) — DONE

Full per-token capture over **3,874,942 scored tokens** (12 steps × ~320k).
Gate: 11/12 PASS (one FAIL, step 5, k3 = 0.0163 — again exactly one token:
e^8.49/326,946 ≈ 0.0149 + 0.0014 bulk). Bulk metrics reproduce the main
arm B run (clean-step k3 0.0013–0.0019, mean |Δlp| 0.013–0.016).

**Tail census (signed r = trainer − sampler logprob):**

| threshold | tokens | rate |
|---|---|---|
| \|r\| > 1 | 328 | 8.5e−5 |
| \|r\| > 2 | 14 | 3.6e−6 |
| \|r\| > 3 | 5 | 1.3e−6 |
| \|r\| > 5 | 3 | **~1 per 1.3M tokens** |
| \|r\| > 10 | 0 | — |
| **r > +2 (trainer MORE likely)** | **0** | — |

Three key findings:

1. **The tail is entirely one-sided.** Not a single token in 3.87M had the
   trainer assigning > e² more probability than the sampler. Catastrophic
   disagreement only occurs in the gradient-suppressing direction
   (importance weight → 0); the RL-dangerous direction (exploding weight)
   never appeared. Consistent with fp4-dequant *flattening* rare-token mass
   rather than concentrating it.
2. **Outlier context is mundane mid-arithmetic tokens**, not degenerate
   text — the three |r| > 5 records: token `3` after a complete-the-square
   step (behavior −1.70, trainer −8.88), token `2` mid divisibility check
   (behavior −0.20 (!), trainer −5.75), token `}` closing a LaTeX fraction
   (behavior −4.13, trainer −12.63). The sampler found them ordinary; the
   trainer scored them 250–5,000× less likely. All at positions 300+ of
   ~500-token completions.
3. **Arrival is Poisson-like and magnitude varies per run**: first arm B run
   caught 3 tokens at 10–13 nats (3 gate FAILs); this rerun (same config,
   same seeds, slightly diverged trajectories) caught 3 at 5.5–8.5 nats
   (1 gate FAIL). Same underlying rate, different luck ⇒ per-step k3 gate
   outcomes are draws from the tail lottery, reproducing the run-1-vs-run-2
   ~100× k3 spread this doc measured at 16k.

Not yet done: same capture on Arm A (native BF16 trainer) to compare tail
*rates* — arm A's 12-step max was 6.8 nats so its rate is plausibly similar,
with only magnitude luck differing, but that's unverified.

## Artifacts

- Metrics: `/root/.cache/user_artifacts/rl_klprobe/logs/metrics_arm{A,B}.jsonl`
  (arm B's crashed pre-fix stub archived as `metrics_armB.crashed-run.jsonl`)
- Driver logs: `.../rl_klprobe/logs/grpo_arm{A,B}.log`; AFTER evals via
  `.../rl_klprobe/eval_after.py` (12-step adapters, same eval rows as BEFORE)
- Per-token capture: `.../rl_klprobe/capture_armB/` +
  `logs/{grpo,metrics}_armBcap.*` (rerun, fresh LoRA, same seeds)
- Drivers: `train_grpo_hendrycks_klprobe.py` (+ `_capture` variant on the
  box); local copy in this directory
- Local code fix on box tree (needs upstream PR):
  `server/src/trainers_server/dp_worker/api/_lora_targets.py` — dispatch on
  `HybridModelProvider` (git diff on `trainers_main`)
