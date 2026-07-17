from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]


def _load_config_module():
    path = SCRIPTS / "glm_klprobe_config.py"
    assert path.exists(), "generic GLM KL-probe configuration helper is missing"
    spec = importlib.util.spec_from_file_location("glm_klprobe_config", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_specs_cover_seeded_gsm8k_and_filtered_math() -> None:
    config = _load_config_module()

    gsm8k = config.dataset_spec("gsm8k", "", data_seed=999)
    assert gsm8k.dataset_name == "openai/gsm8k"
    assert gsm8k.dataset_config == "main"
    assert gsm8k.question_key == "question"
    assert gsm8k.answer_key == "answer"
    assert gsm8k.shuffle_seed == 999
    assert gsm8k.math_levels == ()

    math = config.dataset_spec(
        "math", "Level 4, Level 5", data_seed=16
    )
    assert math.dataset_name == "EleutherAI/hendrycks_math"
    assert math.question_key == "problem"
    assert math.answer_key == "solution"
    assert math.shuffle_seed == 16
    assert math.math_levels == ("Level 4", "Level 5")


def test_summarizer_renders_alpha_metadata_long_probe_and_comparison_row(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    output = tmp_path / "report.md"
    records = [
        {
            "event": "run_metadata",
            "run_id": "alpha",
            "model": "zai-org/GLM-5.2-FP8",
            "dataset": "math",
            "dataset_name": "EleutherAI/hendrycks_math",
            "math_levels": ["Level 4", "Level 5"],
            "enable_thinking": False,
            "steps": 2,
            "batch_size": 4,
            "group_size": 8,
            "max_tokens": 2048,
            "rank": 16,
            "data_seed": 16,
            "sample_seed": None,
            "gate": 0.015,
            "warmup_steps": 1,
        },
        {
            "event": "policy_version_bootstrap",
            "initial_policy_version": 0,
            "final_policy_version": 1,
            "learning_rate": 0.0,
            "grad_norm": 0.0,
        },
        {
            "event": "preflight",
            "required_policy_version": 1,
            "observed_policy_versions": [1],
            "tokens": 32,
            "k3": 0.001,
        },
        {
            "event": "step",
            "step": 0,
            "tokens": 100,
            "k3": 0.020,
            "mean_abs": 0.1,
            "max_abs": 1.0,
            "ess_over_n": 0.9,
            "clip_fraction": 0.1,
            "tail_counts": {"abs_r_gt_1": 0, "abs_r_gt_2": 0, "abs_r_gt_5": 0, "abs_r_gt_10": 0},
            "positive_tail_counts": {"r_gt_1": 0, "r_gt_2": 0, "r_gt_5": 0, "r_gt_10": 0},
            "n_outlier_records": 0,
            "sampler_reload_verified": True,
            "gated": False,
            "gate_pass": False,
            "timings": {"step_s": 1.0},
        },
        {
            "event": "step",
            "step": 1,
            "tokens": 200,
            "k3": 0.010,
            "mean_abs": 0.05,
            "max_abs": 0.5,
            "ess_over_n": 0.95,
            "clip_fraction": 0.05,
            "tail_counts": {"abs_r_gt_1": 0, "abs_r_gt_2": 0, "abs_r_gt_5": 0, "abs_r_gt_10": 0},
            "positive_tail_counts": {"r_gt_1": 0, "r_gt_2": 0, "r_gt_5": 0, "r_gt_10": 0},
            "n_outlier_records": 0,
            "sampler_reload_verified": True,
            "gated": True,
            "gate_pass": True,
            "timings": {"step_s": 2.0},
        },
        {
            "event": "long_probe",
            "prefix_tokens": 35000,
            "decode_tokens": 15000,
            "tokens": 14999,
            "k3": 0.0123,
            "mean_abs": 0.04,
            "max_abs": 0.9,
            "tail_counts": {"abs_r_gt_1": 0, "abs_r_gt_2": 0, "abs_r_gt_5": 0, "abs_r_gt_10": 0},
        },
    ]
    metrics.write_text("".join(json.dumps(record) + "\n" for record in records))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "summarize_hendrycks_klprobe.py"),
            "--metrics", str(metrics),
            "--output", str(output),
            "--capture-dir", str(tmp_path / "capture"),
            "--branch", "main",
            "--sha", "abc123",
            "--trainer-config", "trainer.json",
            "--topology", "4x8 CP32 + 1x8 TP8",
            "--comparison-arm", "Alpha",
        ],
        check=True,
    )

    report = output.read_text()
    assert "EleutherAI/hendrycks_math" in report
    assert "Level 4, Level 5" in report
    assert "thinking: OFF" in report
    assert "step 00" in report and "EXCLUDED" in report
    assert "35,000 + 15,000" in report
    assert "| Alpha | 0.015000 | 0.010000 | 0.010000 | 0/1 |" in report
