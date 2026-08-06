# PR #910 GLM 256k seven-datum soak

Run date: 2026-08-06

## Verdict

PR #910 is stable on the original LPS-1003 seven-datum, 254.5k-token
positive-control payload across 100 identical forwards. No divergence from the
recoverable fixed-wheel baseline was detected.

- 100/100 reps completed.
- 0 nonfinite logprobs across 25,451,000 recorded target positions.
- 0 destroyed reps (tail datum NLL > 4.4).
- 0 low-state reps (tail datum NLL < 3.2).
- All 254,510 target-position logprobs were saved for every rep, including the
  252,812 positions that were unsupervised in the original training payload.

## Setup

- Devbox: `tj-328pr23`, 2 x 8 B300 (`NVIDIA L20D`).
- PR head: `7a34934be366812746c2d9018d4fe3607e4c3f8b`.
- Venv: dedicated shared
  `/root/.cache/user_artifacts/trainers_pr910_soak/server/.venv`.
- Runtime: torch `2.11.0+cu130`, nvidia-cudnn-frontend
  `1.27.0.dev20260803+git7478516` from the PR's `0cu130` wheel,
  nvidia-cutlass-dsl `4.5.2`.
- Topology: TP1 / PP1 / EP16 / CP16, max sequence length 262,144.
- Environment: `CUDA_DEVICE_MAX_CONNECTIONS` unset,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `BT_SKIP_FULL_WARMUP=1`.
- Payload: batch-0 docs 0..6, lengths
  `[41381, 30061, 20896, 12169, 60181, 34647, 55182]`, total 254,517
  tokens, uniform supervision over all positions.
- Payload SHA-256:
  `a1d5f7433c7624c9a699ccbd09299593a0992061be624c9deda98a3c78bade42`.

## Fixed-baseline comparison

The saved armH9 artifact is named as a 150-rep capture, but the gzip is
truncated after 31 complete reps (`0..30`) and lacks its end marker. The
all-position comparison therefore uses those 31 recoverable reps. This is a
baseline-evidence limitation, not a problem with the new capture; the new
100-rep gzip passes `gzip -t`.

Per-datum current mean minus fixed-baseline mean, in nats:

| Datum | Delta |
|---:|---:|
| 0 | -0.00891 |
| 1 | -0.00329 |
| 2 | -0.00304 |
| 3 | +0.00134 |
| 4 | +0.00681 |
| 5 | +0.00425 |
| 6 | -0.00611 |

Every datum moved by less than 0.009 nats. Position-level temporal variation
also matches:

| Statistic | Fixed baseline | PR #910 |
|---|---:|---:|
| Position std p50 | 0.1846 | 0.1932 |
| Position std p95 | 3.1480 | 3.1215 |
| Position std p99 | 4.8224 | 4.7424 |

The absolute shift between per-position means has pooled-standard-deviation
effect size p50 0.138, p95 0.399, p99 0.529, and max 1.025. Only 0.285% of
current position means fall outside the finite 31-rep baseline min/max range.
This is consistent with the baseline's normal per-token churn, not a divergent
state.

For the originally supervised 1,698 tail positions, mean-logprob absolute
delta is p50 0.00021, p95 0.0922, p99 0.2044, max 0.6310 nats. The originally
unsupervised prompt positions are also retained and analyzed in
`comparison_vs_armH9.json`.

## Original-unfixed comparison

The complete original steady-state series has 146 observations. It recorded 8
low-state observations; this run recorded 0/100. Current datum means are close
to the original median state, with current-minus-original-median deltas of
`[-0.0022, +0.0044, -0.0060, +0.0027, +0.0406, +0.0818, +0.0254]` nats.

## Artifacts

- `full_logprobs_100rep.jsonl.gz`: every target-position logprob for all 100
  reps. This 221 MB raw capture is retained locally and at
  `/root/.cache/user_artifacts/lps1003/pr910_soak_20260806/` on the devbox,
  but is not committed because it exceeds GitHub's 100 MB blob limit. SHA-256
  `c2bcf027b69e444c3b7e1a79396d1103298c06462df8b24f90a945fba6ba241e`.
- `comparison_vs_armH9.json`: all-position comparison against the 31
  recoverable fixed-wheel reps. SHA-256
  `a52032dde469a396007192b3bd8c9e68c3df132bccccb97f706994685c6205a5`.
- `comparison_vs_original_unfixed_nlls.json`: NLL comparison against the full
  146-observation original soak. SHA-256
  `d79668740b8e3fd814a6d94b452c682c80307f7f3b97c3bd123bf941b23b9438`.
- `pr910_soak_full_logprobs_v1.py`: capture driver.
- `compare_full_logprob_soaks_v2.py`: all-position streaming comparator.
- `compare_original_nlls_v1.py`: original-soak NLL comparator.
