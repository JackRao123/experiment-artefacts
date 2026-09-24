"""Summarize profile_driver outputs without treating debug-model MFU as valid."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for file in args.files:
        data = json.loads(file.read_text())
        controls = [w for w in data["windows"] if w["phase"] == "control"]
        results.append(
            {
                "label": data["label"],
                "seq_len": data["seq_len"],
                "num_gpus": data["num_gpus"],
                "tokens_per_step": data["tokens_per_step"],
                "forward_backward_tps_per_gpu": data["aggregates"][
                    "control_tps_per_gpu"
                ],
                "control_fb_seconds": [w["fb_elapsed_s"] for w in controls],
                "median_control_tps_per_gpu": statistics.median(
                    w["fb_tps_per_gpu"] for w in controls
                ),
                "optimizer_seconds_mean": data["aggregates"][
                    "control_optim_seconds_mean"
                ],
                "peak_allocated_gib": data["aggregates"]["peak_allocated_bytes"]
                / 2**30,
                "peak_reserved_gib": data["aggregates"]["peak_reserved_bytes"] / 2**30,
                "losses": [w["loss"] for w in controls],
                "grad_norms": [w["grad_norm"] for w in controls],
            }
        )
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
