#!/usr/bin/env python3
"""Compare current NLLs with the complete original unfixed steady-state soak."""

import json
import statistics
import sys
from pathlib import Path


ORIGINAL = Path(sys.argv[1])
FULL_COMPARISON = Path(sys.argv[2])
OUTPUT = Path(sys.argv[3])

rows = [json.loads(line) for line in ORIGINAL.read_text().splitlines() if line.strip()]
valid = [row for row in rows if "nlls" in row]
current = json.loads(FULL_COMPARISON.read_text())

per_datum = []
for doc in range(7):
    values = [row["nlls"][doc] for row in valid]
    current_stats = current["per_datum_nll"][doc]
    per_datum.append(
        {
            "doc": doc,
            "original_mean": statistics.fmean(values),
            "original_median": statistics.median(values),
            "original_min": min(values),
            "original_max": max(values),
            "current_mean": current_stats["current_mean"],
            "current_minus_original_median": current_stats["current_mean"]
            - statistics.median(values),
        }
    )

summary = {
    "original_rows": len(valid),
    "original_destroyed_reps": [
        row["rep"] for row in valid if any(value > 4.4 for value in row["nlls"][4:])
    ],
    "original_low_reps": [
        row["rep"] for row in valid if any(value < 3.2 for value in row["nlls"][4:])
    ],
    "current_destroyed_reps": current["current"]["destroyed_reps"],
    "current_low_reps": current["current"]["low_reps"],
    "per_datum_nll": per_datum,
}
OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
