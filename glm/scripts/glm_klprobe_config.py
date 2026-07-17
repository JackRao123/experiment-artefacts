"""Dependency-free configuration helpers for the GLM KL probes."""

from __future__ import annotations

from typing import NamedTuple


MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


class DatasetSpec(NamedTuple):
    key: str
    dataset_name: str
    dataset_config: str | None
    question_key: str
    answer_key: str
    shuffle_seed: int
    math_levels: tuple[str, ...]


def dataset_spec(dataset: str, math_levels: str, *, data_seed: int) -> DatasetSpec:
    levels = tuple(level.strip() for level in math_levels.split(",") if level.strip())
    if dataset == "gsm8k":
        return DatasetSpec(
            key="gsm8k",
            dataset_name="openai/gsm8k",
            dataset_config="main",
            question_key="question",
            answer_key="answer",
            shuffle_seed=data_seed,
            math_levels=(),
        )
    if dataset == "math":
        return DatasetSpec(
            key="math",
            dataset_name="EleutherAI/hendrycks_math",
            dataset_config=None,
            question_key="problem",
            answer_key="solution",
            shuffle_seed=data_seed,
            math_levels=levels,
        )
    raise ValueError(f"unsupported dataset: {dataset}")


def load_dataset_from_spec(spec: DatasetSpec):
    """Load and deterministically order the selected probe dataset."""
    from datasets import concatenate_datasets, load_dataset

    if spec.key == "gsm8k":
        return load_dataset(
            spec.dataset_name, spec.dataset_config, split="train"
        ).shuffle(seed=spec.shuffle_seed)

    parts = [
        load_dataset(spec.dataset_name, subject, split="train")
        for subject in MATH_SUBJECTS
    ]
    data = concatenate_datasets(parts)
    if spec.math_levels:
        keep = set(spec.math_levels)
        data = data.filter(lambda row: row["level"] in keep)
    return data.shuffle(seed=spec.shuffle_seed)
