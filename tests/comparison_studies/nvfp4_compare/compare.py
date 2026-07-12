#!/usr/bin/env python3
"""NVFP4-vs-bf16 spike: CPU comparison stage.

Consumes two ``score.py`` outputs that scored the *same* token sequences and
reports how far the NVFP4 sampler's per-token logprobs drift from the bf16
reference. This is the go/no-go number for Option A: the RL loop already
corrects a train/sample logprob gap via importance sampling (cispo), so the
question is purely "how wide does NVFP4 make that gap, and does the IS
correction stay healthy?"

Convention (matters for sign):
  behavior = the policy that actually produced the tokens  -> NVFP4 generate
  target   = the policy whose gradient we follow           -> bf16 rescore
  per-token log ratio  r = lp_target - lp_behavior
  per-token importance weight  w = exp(r) = pi_target / pi_behavior

Metrics:
  * logprob drift: mean/RMS/max |lp_target - lp_behavior| over completion tokens
  * KL estimators (Schulman):
      k1 = mean(-r)            ~ KL(behavior || target), low variance, biased
      k3 = mean(exp(-r) + r - 1) ~ KL(behavior || target), unbiased, nonneg
  * importance weights w: mean, p50/p95/p99, fraction outside clip bands
  * token-level effective sample size  ESS = (sum w)^2 / sum(w^2), and ESS/N
    (caveat: true RL ESS is per-sequence/group; this token-level proxy is a
    fast first read of how much mass the correction throws away)

A healthy result: small drift, k3 KL near zero, w concentrated near 1, tiny
clip fraction, ESS/N close to 1. A collapse: heavy clip fraction and ESS/N
far below 1 means NVFP4 drifted too far and Option A needs FP8 base or QLoRA.

Usage:
    python compare.py --behavior gen_nvfp4.json --target rescore_bf16.json \
        [--clip-low 0.8 --clip-high 1.2] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_seqs(path: Path) -> dict[Any, dict]:
    data = json.loads(path.read_text())
    by_id: dict[Any, dict] = {}
    for s in data["sequences"]:
        by_id[s["id"]] = s
    return by_id


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _collect_token_pairs(
    behavior: dict[Any, dict], target: dict[Any, dict]
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Return (behavior_logprobs, target_logprobs) aligned per completion token,
    plus a diagnostics dict. Skips tokens that are NaN in either source, and
    asserts token-id alignment so we never compare logprobs of different
    tokens."""
    lp_b: list[float] = []
    lp_t: list[float] = []
    n_seq_compared = 0
    n_seq_skipped = 0
    n_tok_dropped = 0
    misaligned: list[Any] = []

    for sid, sb in behavior.items():
        st = target.get(sid)
        if st is None:
            n_seq_skipped += 1
            continue
        if sb["completion_token_ids"] != st["completion_token_ids"]:
            misaligned.append(sid)
            continue
        n_seq_compared += 1
        for b_lp, t_lp in zip(sb["completion_logprobs"], st["completion_logprobs"]):
            if b_lp is None or t_lp is None or math.isnan(b_lp) or math.isnan(t_lp):
                n_tok_dropped += 1
                continue
            lp_b.append(float(b_lp))
            lp_t.append(float(t_lp))

    diag = {
        "sequences_compared": n_seq_compared,
        "sequences_skipped_missing": n_seq_skipped,
        "sequences_skipped_misaligned": misaligned,
        "tokens_dropped_nan": n_tok_dropped,
        "tokens_compared": len(lp_b),
    }
    return lp_b, lp_t, diag


def compute_metrics(
    lp_behavior: list[float],
    lp_target: list[float],
    *,
    clip_low: float,
    clip_high: float,
) -> dict[str, Any]:
    n = len(lp_behavior)
    if n == 0:
        return {"error": "no comparable tokens"}

    abs_diffs = [abs(t - b) for b, t in zip(lp_behavior, lp_target)]
    r = [t - b for b, t in zip(lp_behavior, lp_target)]  # log importance ratio
    w = [math.exp(min(ri, 50.0)) for ri in r]  # cap to avoid overflow on tails

    mean_abs = sum(abs_diffs) / n
    rms = math.sqrt(sum(d * d for d in abs_diffs) / n)
    max_abs = max(abs_diffs)

    k1 = sum(-ri for ri in r) / n
    k3 = sum(math.exp(-ri) + ri - 1 for ri in r) / n

    sum_w = sum(w)
    sum_w2 = sum(wi * wi for wi in w)
    ess = (sum_w * sum_w) / sum_w2 if sum_w2 > 0 else 0.0

    n_clipped = sum(1 for wi in w if wi < clip_low or wi > clip_high)
    w_sorted = sorted(w)

    return {
        "tokens": n,
        "logprob_drift": {
            "mean_abs": mean_abs,
            "rms": rms,
            "max_abs": max_abs,
        },
        "kl_estimators": {
            "k1_behavior_to_target": k1,
            "k3_behavior_to_target": k3,
        },
        "importance_weights": {
            "mean": sum_w / n,
            "p50": _percentile(w_sorted, 0.50),
            "p95": _percentile(w_sorted, 0.95),
            "p99": _percentile(w_sorted, 0.99),
            "min": w_sorted[0],
            "max": w_sorted[-1],
        },
        "clip": {
            "band": [clip_low, clip_high],
            "fraction_outside": n_clipped / n,
        },
        "effective_sample_size": {
            "ess": ess,
            "ess_over_n": ess / n,
            "note": "token-level proxy; true RL ESS is per-sequence/group",
        },
    }


def _verdict(metrics: dict[str, Any]) -> str:
    """Coarse heuristic read. Tune thresholds once we have a bf16-vs-bf16
    baseline run to subtract the pure engine-mismatch floor."""
    if "error" in metrics:
        return "NO DATA"
    clip = metrics["clip"]["fraction_outside"]
    ess = metrics["effective_sample_size"]["ess_over_n"]
    k3 = metrics["kl_estimators"]["k3_behavior_to_target"]
    if clip < 0.05 and ess > 0.8 and k3 < 0.02:
        return "HEALTHY (NVFP4 gap small; cispo correction comfortable)"
    if clip < 0.20 and ess > 0.5:
        return "MARGINAL (watch ESS/clip during a real RL run before trusting)"
    return "COLLAPSE RISK (NVFP4 drift large; consider FP8 base or QLoRA)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--behavior", type=Path, required=True, help="NVFP4 generate output"
    )
    ap.add_argument("--target", type=Path, required=True, help="bf16 rescore output")
    ap.add_argument("--clip-low", type=float, default=0.8)
    ap.add_argument("--clip-high", type=float, default=1.2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    behavior = _load_seqs(args.behavior)
    target = _load_seqs(args.target)
    lp_b, lp_t, diag = _collect_token_pairs(behavior, target)
    metrics = compute_metrics(
        lp_b, lp_t, clip_low=args.clip_low, clip_high=args.clip_high
    )
    verdict = _verdict(metrics)
    report = {
        "behavior_file": str(args.behavior),
        "target_file": str(args.target),
        "diagnostics": diag,
        "metrics": metrics,
        "verdict": verdict,
    }

    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    main()
    raise SystemExit(0)
