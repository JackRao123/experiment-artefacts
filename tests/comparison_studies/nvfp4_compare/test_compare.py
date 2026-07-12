#!/usr/bin/env python3
"""CPU unit tests for the NVFP4 spike compare-stage math.

Pure functions, no GPU/vLLM. Run:
    python -m pytest tests/comparison_studies/nvfp4_compare/test_compare.py
or:
    python tests/comparison_studies/nvfp4_compare/test_compare.py
"""

from __future__ import annotations

import math

import compare


def _seqs(records):
    return {r["id"]: r for r in records}


def test_identical_logprobs_are_perfectly_healthy():
    lp = [-0.1, -0.5, -1.2, -0.3]
    metrics = compare.compute_metrics(lp, list(lp), clip_low=0.8, clip_high=1.2)
    assert metrics["logprob_drift"]["mean_abs"] == 0.0
    assert metrics["logprob_drift"]["max_abs"] == 0.0
    assert abs(metrics["kl_estimators"]["k3_behavior_to_target"]) < 1e-12
    assert abs(metrics["importance_weights"]["mean"] - 1.0) < 1e-12
    assert metrics["clip"]["fraction_outside"] == 0.0
    assert abs(metrics["effective_sample_size"]["ess_over_n"] - 1.0) < 1e-12


def test_constant_shift_gives_constant_weight():
    # target uniformly higher by ln(1.1): every weight == 1.1, inside [0.8,1.2]
    shift = math.log(1.1)
    lp_b = [-0.4, -0.6, -0.9]
    lp_t = [x + shift for x in lp_b]
    m = compare.compute_metrics(lp_b, lp_t, clip_low=0.8, clip_high=1.2)
    assert abs(m["importance_weights"]["mean"] - 1.1) < 1e-9
    assert m["clip"]["fraction_outside"] == 0.0
    # constant weight => ESS/N == 1 exactly
    assert abs(m["effective_sample_size"]["ess_over_n"] - 1.0) < 1e-9
    # k3 KL is positive for any nonzero shift
    assert m["kl_estimators"]["k3_behavior_to_target"] > 0.0


def test_large_drift_trips_clip_and_drops_ess():
    # half the tokens have weight e^2 ~ 7.4 (way outside band), half ~1
    lp_b = [-1.0, -1.0, -1.0, -1.0]
    lp_t = [-1.0, 1.0, -1.0, 1.0]
    m = compare.compute_metrics(lp_b, lp_t, clip_low=0.8, clip_high=1.2)
    assert m["clip"]["fraction_outside"] == 0.5
    assert m["effective_sample_size"]["ess_over_n"] < 0.8


def test_misaligned_tokens_are_skipped_not_compared():
    behavior = _seqs(
        [
            {
                "id": 0,
                "completion_token_ids": [5, 6, 7],
                "completion_logprobs": [-0.1, -0.2, -0.3],
            }
        ]
    )
    target = _seqs(
        [
            {
                "id": 0,
                "completion_token_ids": [5, 6, 99],  # diverged token ids
                "completion_logprobs": [-0.1, -0.2, -0.3],
            }
        ]
    )
    _, _, diag = compare._collect_token_pairs(behavior, target)
    assert diag["sequences_skipped_misaligned"] == [0]
    assert diag["tokens_compared"] == 0


def test_nan_tokens_are_dropped():
    behavior = _seqs(
        [
            {
                "id": 1,
                "completion_token_ids": [1, 2],
                "completion_logprobs": [-0.1, float("nan")],
            }
        ]
    )
    target = _seqs(
        [{"id": 1, "completion_token_ids": [1, 2], "completion_logprobs": [-0.1, -0.5]}]
    )
    lp_b, lp_t, diag = compare._collect_token_pairs(behavior, target)
    assert diag["tokens_compared"] == 1
    assert diag["tokens_dropped_nan"] == 1
    assert lp_b == [-0.1] and lp_t == [-0.1]


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
