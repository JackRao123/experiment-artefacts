#!/usr/bin/env python3
"""Render a rich GLM trainer/sampler KL-probe JSONL report."""

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


def _dataset_line(metadata: dict) -> str:
    name = metadata.get("dataset_name", metadata.get("dataset", "unknown"))
    levels = metadata.get("math_levels") or []
    suffix = f"; levels={', '.join(levels)}" if levels else ""
    return f"- dataset: {name}{suffix}; shuffle seed={metadata['data_seed']}"


def render_report(records: list[dict], args) -> str:
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
    gated_steps = [step for step in steps if step.get("gated", True)]
    passing_steps = [step for step in gated_steps if step["k3"] < gate]
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
    overall_pass = bool(gated_steps) and len(passing_steps) == len(gated_steps)

    lines = [
        "# GLM-5.2 trainer/sampler KL probe",
        "",
        "Run metadata",
        f"- branch / SHA: {args.branch} / {args.sha}",
        f"- trainer image: {args.trainer_image}",
        f"- sampler image: {args.sampler_image}",
        f"- trainer config: {args.trainer_config}",
        f"- model: {metadata['model']}",
        f"- topology: {args.topology}",
        _dataset_line(metadata),
        f"- thinking: {'ON' if metadata.get('enable_thinking') else 'OFF'}",
        f"- rollout: {metadata['steps']} steps; "
        f"{metadata['batch_size']} problems/step; "
        f"group={metadata['group_size']}; max_tokens={metadata['max_tokens']}; "
        f"T=1.0; top_p=1.0; sample seed={metadata.get('sample_seed')}",
        f"- gate: k3 < {gate}; warmup steps={metadata.get('warmup_steps', 0)}",
        "",
        "Preflight",
        "- version-zero bootstrap: "
        f"{bootstrap['initial_policy_version']}->{bootstrap['final_policy_version']}; "
        f"lr={bootstrap['learning_rate']}; grad_norm={bootstrap['grad_norm']:.1e}",
        f"- adapter policy version: required {preflight['required_policy_version']}; "
        f"observed {preflight['observed_policy_versions']}",
        f"- teacher-forced aligned tokens: {preflight['tokens']}",
        (
            "- preflight k3: not recomputable with the trainer formula "
            f"(legacy reverse-sign value was {preflight['k3']:.6f}; per-token "
            "preflight capture was not retained)"
            if preflight.get("k3_definition") == "legacy_reverse_sign"
            else f"- preflight k3: {preflight['k3']:.6f}"
        ),
        "",
        "Per-step parity",
        "| step | k3 | mean_abs | max_abs | ESS/N | clip | tokens | tails (\\|r\\|>1/2/5/10) | gate | step_s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for step in steps:
        tails = step["tail_counts"]
        gate_label = (
            "EXCLUDED"
            if not step.get("gated", True)
            else ("PASS" if step["gate_pass"] else "FAIL")
        )
        lines.append(
            f"| {step['step']:02d} | {step['k3']:.6f} | "
            f"{step['mean_abs']:.6f} | {step['max_abs']:.4f} | "
            f"{step['ess_over_n']:.4f} | {step['clip_fraction']:.4f} | "
            f"{step['tokens']} | {tails['abs_r_gt_1']}/{tails['abs_r_gt_2']}/"
            f"{tails['abs_r_gt_5']}/{tails['abs_r_gt_10']} | {gate_label} | "
            f"{step.get('timings', {}).get('step_s', float('nan')):.0f} |"
        )

    gated_mean = sum(step["k3"] for step in gated_steps) / len(gated_steps)
    gated_max = max(step["k3"] for step in gated_steps)
    failures = len(gated_steps) - len(passing_steps)
    lines.extend(
        [
            "",
            "Summary",
            f"- gated steps below {gate}: {len(passing_steps)} / {len(gated_steps)}",
            f"- maximum gated k3: {gated_max:.6f}",
            f"- mean gated k3: {gated_mean:.6f}",
            "- token-weighted mean absolute logprob delta: "
            f"{weighted_mean(steps, 'mean_abs'):.6f}",
            f"- sampler reloads verified: {'YES' if reloads_verified else 'NO'}",
            f"- overall parity verdict: {'PASS' if overall_pass else 'FAIL'}",
            "",
            "Compressed comparison",
            "| arm | gate | mean mismatch_kl | max | steps >= gate |",
            "|---|---:|---:|---:|---:|",
            f"| {args.comparison_arm} | {gate:.6f} | {gated_mean:.6f} | "
            f"{gated_max:.6f} | {failures}/{len(gated_steps)} |",
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
        ]
    )

    long_probe = next(
        (record for record in records if record["event"] == "long_probe"), None
    )
    if long_probe:
        lines.extend(
            [
                "",
                "Long-context observation",
                f"- prefix + decode: {long_probe['prefix_tokens']:,} + "
                f"{long_probe['decode_tokens']:,} tokens",
                f"- scored decode tokens: {long_probe['tokens']:,}",
                f"- k3: {long_probe['k3']:.6f}",
                f"- mean_abs / max_abs: {long_probe['mean_abs']:.6f} / "
                f"{long_probe['max_abs']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation",
            "- "
            + (
                "All gated steps passed the literal k3 threshold; inspect mean_abs, "
                "ESS, clip fraction, and signed tails before concluding bulk parity."
                if overall_pass
                else "At least one gated step failed the literal k3 threshold. Inspect "
                "the rich metrics and captures to distinguish bulk drift from tails."
            ),
            "",
        ]
    )
    return "\n".join(lines)


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
    parser.add_argument("--comparison-arm", default="run")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.metrics).read_text().splitlines()
        if line.strip()
    ]
    Path(args.output).write_text(render_report(records, args))


if __name__ == "__main__":
    main()
