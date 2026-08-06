#!/usr/bin/env python3
"""Compare two full-position LPS-1003 seven-datum soak captures."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np


BASELINE = Path(sys.argv[1])
CURRENT = Path(sys.argv[2])
OUTPUT = Path(sys.argv[3])
LENGTHS = [41380, 30060, 20895, 12168, 60180, 34646, 55181]
PREFIX_LENGTHS = [41081, 29797, 20671, 12033, 60003, 34379, 54855]
TOTAL = sum(LENGTHS)


def percentiles(values):
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def load(path):
    count = 0
    mean = np.zeros(TOTAL, dtype=np.float64)
    m2 = np.zeros(TOTAL, dtype=np.float64)
    minimum = np.full(TOTAL, np.inf, dtype=np.float64)
    maximum = np.full(TOTAL, -np.inf, dtype=np.float64)
    nlls = []
    nonfinite = 0
    destroyed = []
    low = []
    truncated = False
    with gzip.open(path, "rt") as source:
        try:
            for line in source:
                record = json.loads(line)
                arrays = [np.asarray(row, dtype=np.float64) for row in record["logprobs"]]
                lengths = [len(row) for row in arrays]
                if lengths != LENGTHS:
                    raise ValueError(f"{path}: rep {record['rep']} lengths {lengths}")
                values = np.concatenate(arrays)
                nonfinite += int(np.count_nonzero(~np.isfinite(values)))
                count += 1
                delta = values - mean
                mean += delta / count
                m2 += delta * (values - mean)
                np.minimum(minimum, values, out=minimum)
                np.maximum(maximum, values, out=maximum)
                row_nlls = np.asarray(record["nlls"], dtype=np.float64)
                nlls.append(row_nlls)
                if np.any(row_nlls[4:] > 4.4):
                    destroyed.append(int(record["rep"]))
                if np.any(row_nlls[4:] < 3.2):
                    low.append(int(record["rep"]))
        except EOFError:
            truncated = True
    variance = m2 / (count - 1) if count > 1 else np.zeros(TOTAL)
    return {
        "count": count,
        "mean": mean,
        "variance": variance,
        "minimum": minimum,
        "maximum": maximum,
        "nlls": np.stack(nlls),
        "nonfinite": nonfinite,
        "destroyed": destroyed,
        "low": low,
        "truncated": truncated,
    }


def locate(position):
    offset = int(position)
    for doc, length in enumerate(LENGTHS):
        if offset < length:
            return {"doc": doc, "target_offset": offset}
        offset -= length
    raise ValueError(position)


baseline = load(BASELINE)
current = load(CURRENT)
delta = current["mean"] - baseline["mean"]
abs_delta = np.abs(delta)
pooled_std = np.sqrt((baseline["variance"] + current["variance"]) / 2)
effect = np.divide(abs_delta, pooled_std, out=np.zeros_like(abs_delta), where=pooled_std > 0)

prompt_mask = np.zeros(TOTAL, dtype=bool)
cursor = 0
for length, prefix in zip(LENGTHS, PREFIX_LENGTHS):
    prompt_mask[cursor : cursor + prefix - 1] = True
    cursor += length
response_mask = ~prompt_mask

nll_baseline = baseline["nlls"]
nll_current = current["nlls"]
nll_summary = []
for doc in range(len(LENGTHS)):
    base_values = nll_baseline[:, doc]
    current_values = nll_current[:, doc]
    nll_summary.append(
        {
            "doc": doc,
            "baseline_mean": float(np.mean(base_values)),
            "baseline_std": float(np.std(base_values, ddof=1)),
            "baseline_min": float(np.min(base_values)),
            "baseline_max": float(np.max(base_values)),
            "current_mean": float(np.mean(current_values)),
            "current_std": float(np.std(current_values, ddof=1)),
            "current_min": float(np.min(current_values)),
            "current_max": float(np.max(current_values)),
            "mean_delta": float(np.mean(current_values) - np.mean(base_values)),
        }
    )

max_position = int(np.argmax(abs_delta))
summary = {
    "baseline": {
        "path": str(BASELINE),
        "reps": baseline["count"],
        "nonfinite_logprobs": baseline["nonfinite"],
        "destroyed_reps": baseline["destroyed"],
        "low_reps": baseline["low"],
        "truncated_gzip": baseline["truncated"],
        "temporal_position_std": percentiles(np.sqrt(baseline["variance"])),
        "temporal_position_range": percentiles(baseline["maximum"] - baseline["minimum"]),
    },
    "current": {
        "path": str(CURRENT),
        "reps": current["count"],
        "nonfinite_logprobs": current["nonfinite"],
        "destroyed_reps": current["destroyed"],
        "low_reps": current["low"],
        "truncated_gzip": current["truncated"],
        "temporal_position_std": percentiles(np.sqrt(current["variance"])),
        "temporal_position_range": percentiles(current["maximum"] - current["minimum"]),
    },
    "all_positions": {
        "positions": TOTAL,
        "mean_logprob_abs_delta": percentiles(abs_delta),
        "pooled_std_effect_size": percentiles(effect),
        "fraction_abs_delta_gt_0_01": float(np.mean(abs_delta > 0.01)),
        "fraction_abs_delta_gt_0_05": float(np.mean(abs_delta > 0.05)),
        "fraction_abs_delta_gt_0_1": float(np.mean(abs_delta > 0.1)),
        "fraction_abs_delta_gt_0_5": float(np.mean(abs_delta > 0.5)),
        "current_mean_outside_baseline_range_fraction": float(
            np.mean((current["mean"] < baseline["minimum"]) | (current["mean"] > baseline["maximum"]))
        ),
        "max_abs_delta_location": locate(max_position),
        "max_abs_delta": float(abs_delta[max_position]),
    },
    "originally_unsupervised_prompt_positions": {
        "positions": int(np.count_nonzero(prompt_mask)),
        "mean_logprob_abs_delta": percentiles(abs_delta[prompt_mask]),
        "pooled_std_effect_size": percentiles(effect[prompt_mask]),
    },
    "originally_supervised_tail_positions": {
        "positions": int(np.count_nonzero(response_mask)),
        "mean_logprob_abs_delta": percentiles(abs_delta[response_mask]),
        "pooled_std_effect_size": percentiles(effect[response_mask]),
    },
    "per_datum_nll": nll_summary,
}

OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
