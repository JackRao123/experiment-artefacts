# LPS-1003 fix validation — 100 steps of Mudith's recipe, GLM-5.2-FP8, B300

**VERDICT: 0 loss spikes in 100 steps** (baseline: 26 in 147, ~17.7%/step).
Fisher exact vs the FP8 baseline: **p = 6.5e-7**. Poisson probability of
observing zero at the baseline rate over 100 steps: **2.1e-8**.

Run date 2026-08-02, devbox `q8x5ky3` (2 x 8 B300), 05:01–12:28 UTC.

## What was run

Faithful replay of `repro/loops_sft.py`'s recipe against the fixed trainer —
NOT an execution of the Loops SFT stack (see *Scope* below).

| | |
|---|---|
| model | `zai-org/GLM-5.2-FP8`, LoRA rank 32, cp16, max_seq_len 262144 (`trainer-config.flash.json`) |
| recipe | lr 5e-4 cosine over 1528 steps, betas 0.9/0.95, global batch 32 |
| data | batches 0–99, byte-exact prod batches: same `conversations.train.jsonl`, `random.Random(0)` shuffle, thinking-off template, same >131072/trivial filters, consecutive groups of 32. First 1024 rows verified byte-identical to the prod bundle used for the 12-step Part B run. |
| driver | `train_replay.py --steps 100 --tag fix100` (POSTs `/forward_backward` + `/optim_step`) |
| fix | `+dsatopk4` dedicated launch stream — installed `_interface.py` sha256 `bba5b2e73e02fc497fbf3034a39caba129746fc0b3a4ca4df30746ca70660aad`, byte-identical to `rearm/fix_candidate/interface_dedicated_stream.py` (the artifact that passed Part B) |
| env parity | verified in the LIVE process: `CUDA_DEVICE_MAX_CONNECTIONS` **unset** (the bug is NOT masked), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `BT_SKIP_FULL_WARMUP=1` |

## Results

```
steps=100  first=0.7641  last=0.4086  min=0.3604  max=0.7641  median=0.4287
SPIKES calibrated (>1.6x local median): 0   (0.0% of steps)
SPIKES strict     (>2.0x local median): 0   (0.0% of steps)
grad_norm: min=7.28e-04  max=4.20e-03  median=1.28e-03  max/median=3.28x
wall: 5.45h total, mean 196s/step, median 164s, max 561s
0 errors, 0 TTL-evicted retries in the log
```

The curve never exceeded its own step-0 value: `max == first == 0.7641`. For
contrast the baseline's first 147 steps ran min 0.3981 / **max 2.6474** with
spikes already at steps 26, 27, 29 (2.05–2.46x local median).

**Detector calibration.** `analyze_replay_spikes.py` uses a local-median
window (WIN=5) and is calibrated so it *reproduces the documented baseline*:
at RATIO=1.6 it finds 25 spikes in the baseline's first 147 steps (17.0%)
against the 26 (17.7%) recorded in `repro/README.md`, and 17.8% across all 381
steps. So the clean result is not an artifact of a lenient threshold. The
stricter 2.0x threshold is reported alongside and also finds zero.

**grad_norm caveat, stated honestly.** The documented spike signature is a
2–4x NLL excursion *with* grad_norm 3–45x alongside. Our largest grad_norm
excursion was 3.28x its local median — at the very bottom edge of that band —
but it was **not** accompanied by any NLL excursion, so it does not meet the
spike definition. In absolute terms grad norms stayed in 7.3e-4 … 4.2e-3,
matching the validated 12-step Part B range (7e-4 … 1.9e-3) modulo the longer
window.

**Historical control.** `loss_histories.json` also holds a 30-step run on the
`dsatopk1` wheel (the earlier topk-OOB patch, PR #814): it still spiked at
13.3%/step. That patch is the *base wheel* under our venv; the launch-stream
change is layered on top. So the thing being validated here is specifically
the launch-stream fix, not the older patch.

## Scope — what this does and does not prove

Proves: with the launch-stream fix and prod-faithful env (conn unset), the
exact prod batches through the real trainer produce no sawtooth over 100
steps, where ~18 spikes were expected.

Does not cover: the Loops SFT stack itself (orchestration, checkpointing,
eval, sampler, W&B logging) — the driver talks to the trainer API directly.
Also only the first 6.5% of the 1528-step cosine schedule. And because it
replays fixed pre-built batches, a client-side or data-ordering cause would
not be re-tested here — though that was already ruled out separately (all
1024 docs clean, worst 1.153 nats vs prod's 5–11; packing provably
deterministic).

Absolute NLL is not comparable step-for-step with the W&B history (baseline
step 0 = 1.4617 vs our 0.7641) — different LoRA init draw and loss
normalisation. Our numbers *are* directly comparable to the b5 12-step Part B
run on the same vehicle (0.7656 → 0.4637); we reproduced 0.7641 → 0.4637-ish
on the same trajectory. The spike comparison is structural (relative to local
median), which is unaffected.

## Evidence

- `fix_validation_100step/replay100.jsonl.gz` — 100 `train_step` rows: `step`,
  `mean_nll`, `lr`, `wall_s`, `optim_metrics.grad_norm`, per-datum `nll` under
  `datums`
- `fix_validation_100step/replay100.log` — one line per step
- `analyze_replay_spikes.py` — detector + `--baseline` self-check

## Reproduce

```bash
python3 analyze_replay_spikes.py --baseline          # detector calibration
python3 analyze_replay_spikes.py --replay <jsonl>    # verdict
```

## Operational notes (cost time this session)

- SLURM task 0 landed on ssh node **-1**, not the devbox leader (-0).
  `wait_trainer_health.sh` on -0 times out forever while the trainer is
  healthy; the uvicorn 8001 listener is on -1.
- `ss` is **not installed** on these nodes — it silently reports nothing.
  Use `grep :1F41 /proc/net/tcp` to find the 8001 listener.
- The `+dsatopk4` fix exists ONLY as an in-place edit in the CPFS venv; the
  repo checkout's `server/patches/` carries just the older topk-OOB patch. Any
  venv rebuild silently reverts to the buggy code. Land PR #875 before this
  box is recycled.
- The whole session was blocked for hours by the org-wide CPFS 1M inode cap —
  see `CPFS_QUOTA_ESCALATION.md`.
