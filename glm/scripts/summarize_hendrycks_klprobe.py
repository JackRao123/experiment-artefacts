#!/usr/bin/env python3
"""Render the GLM CP32 KL probe JSONL as a reproducible Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def weighted_mean(rows: list[dict], key: str) -> float:
    tokens = sum(row["tokens"] for row in rows)
    return (
        sum(row[key] * row["tokens"] for row in rows) / tokens
        if tokens
        else float("nan")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--trainer-image", default="source server/.venv")
    parser.add_argument("--sampler-image", default="source sampler/.venv")
    parser.add_argument("--trainer-config", required=True)
    parser.add_argument("--topology", required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.metrics).read_text().splitlines()
        if line.strip()
    ]
    metadata = next(record for record in records if record["event"] == "run_metadata")
    bootstrap = next(
        record for record in records if record["event"] == "policy_version_bootstrap"
    )
    preflight = next(record for record in records if record["event"] == "preflight")
    steps = sorted(
        (record for record in records if record["event"] == "step"),
        key=lambda record: record["step"],
    )
    if len(steps) != metadata["steps"]:
        raise RuntimeError(
            f"incomplete run: found {len(steps)}/{metadata['steps']} step records"
        )

    gate = metadata["gate"]
    clean_steps = [step for step in steps if step["k3"] < gate]
    total_tails = {
        threshold: sum(step["tail_counts"][f"abs_r_gt_{threshold}"] for step in steps)
        for threshold in (1, 2, 5, 10)
    }
    positive_tails = {
        threshold: sum(
            step["positive_tail_counts"][f"r_gt_{threshold}"] for step in steps
        )
        for threshold in (1, 2, 5, 10)
    }
    outliers = sum(step.get("n_outlier_records", 0) for step in steps)
    reloads_verified = all(step["sampler_reload_verified"] for step in steps)
    overall_pass = len(clean_steps) == len(steps)

    lines = [
        "# GLM-5.2 CP32 trainer/sampler KL probe",
        "",
        "Run metadata",
        f"- branch / SHA: {args.branch} / {args.sha}",
        f"- trainer image: {args.trainer_image}",
        f"- sampler image: {args.sampler_image}",
        f"- trainer config: {args.trainer_config}",
        "- model: zai-org/GLM-5.2-FP8",
        f"- topology: {args.topology}",
        "- dataset: PrimeIntellect/Hendrycks-Math/default/train; shuffle seed=999",
        f"- rollout: {metadata['steps']} steps; "
        f"{metadata['batch_size']} problems/step; "
        f"group={metadata['group_size']}; max_tokens={metadata['max_tokens']}; "
        f"T=1.0; top_p=1.0; sample seed={metadata['sample_seed']}",
        f"- gate: k3 < {gate}",
        "",
        "Preflight",
        "- version-zero bootstrap: "
        f"{bootstrap['initial_policy_version']}->{bootstrap['final_policy_version']}; "
        f"lr={bootstrap['learning_rate']}; grad_norm={bootstrap['grad_norm']:.1e}",
        f"- adapter policy version: required {preflight['required_policy_version']}; "
        f"observed {preflight['observed_policy_versions']}",
        f"- teacher-forced aligned tokens: {preflight['tokens']}",
        f"- preflight k3: {preflight['k3']:.6f}",
        "",
        "Per-step parity",
    ]
    for step in steps:
        tails = step["tail_counts"]
        lines.append(
            f"- step {step['step']:02d}: k3={step['k3']:.6f}  "
            f"mean_abs={step['mean_abs']:.6f}  max_abs={step['max_abs']:.4f}  "
            f"ESS/N={step['ess_over_n']:.4f}  clip={step['clip_fraction']:.4f}  "
            f"tokens={step['tokens']}  "
            "tails(|r|>1/2/5/10)="
            f"{tails['abs_r_gt_1']}/{tails['abs_r_gt_2']}/"
            f"{tails['abs_r_gt_5']}/{tails['abs_r_gt_10']}  "
            f"gate={'PASS' if step['gate_pass'] else 'FAIL'}"
        )

    clean_mean = (
        sum(step["k3"] for step in clean_steps) / len(clean_steps)
        if clean_steps
        else float("nan")
    )
    lines.extend(
        [
            "",
            "Summary",
            f"- steps below {gate}: {len(clean_steps)} / {len(steps)}",
            f"- maximum k3: {max(step['k3'] for step in steps):.6f}",
            f"- clean-step mean k3: {clean_mean:.6f}",
            f"- token-weighted mean absolute logprob delta: "
            f"{weighted_mean(steps, 'mean_abs'):.6f}",
            f"- sampler reloads verified: {'YES' if reloads_verified else 'NO'}",
            f"- overall parity verdict: {'PASS' if overall_pass else 'FAIL'}",
            "",
            "Tail analysis",
            f"- behavior/target logprob captures: {args.capture_dir}",
            f"- total scored tokens: {sum(step['tokens'] for step in steps)}",
            "- tails |r|>1/2/5/10: "
            f"{total_tails[1]}/{total_tails[2]}/{total_tails[5]}/{total_tails[10]}",
            f"- decoded outliers with |r| > 5: {outliers}",
            "- dangerous positive-r outliers r>1/2/5/10: "
            f"{positive_tails[1]}/{positive_tails[2]}/"
            f"{positive_tails[5]}/{positive_tails[10]}",
            "- interpretation: "
            + (
                "All steps passed the literal k3 gate; inspect mean_abs, ESS, "
                "clip fraction, and signed tails before concluding bulk parity."
                if overall_pass
                else "The literal gate failed on at least one step. Compare each "
                "failure with mean_abs, ESS, clip fraction, and signed tail counts "
                "to distinguish bulk mismatch from isolated tail domination."
            ),
            "",
        ]
    )
    Path(args.output).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
