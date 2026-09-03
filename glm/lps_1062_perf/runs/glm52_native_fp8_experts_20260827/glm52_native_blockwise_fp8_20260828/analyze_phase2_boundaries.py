"""Compare sampled Phase 2 boundary tensors across all ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def compare(left: dict, right: dict) -> dict:
    result = {
        "shape_equal": left["shape"] == right["shape"],
        "dtype_equal": left["dtype"] == right["dtype"],
    }
    for field in ("sample", "tail_sample"):
        difference = right[field].float() - left[field].float()
        result[field] = {
            "equal": bool(torch.equal(left[field], right[field])),
            "max_absolute_error": float(difference.abs().max()),
            "mean_absolute_error": float(difference.abs().mean()),
        }
    for field in ("sum", "abs_sum", "square_sum", "minimum", "maximum"):
        result[f"{field}_absolute_error"] = float(
            (right[field].float() - left[field].float()).abs()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    result = {}
    for control_path in sorted(args.control.glob("rank*.pt")):
        candidate_path = args.candidate / control_path.name
        if not candidate_path.exists():
            raise FileNotFoundError(candidate_path)
        left = torch.load(control_path, map_location="cpu", weights_only=True)
        right = torch.load(candidate_path, map_location="cpu", weights_only=True)
        result[control_path.stem] = compare(left, right)

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
