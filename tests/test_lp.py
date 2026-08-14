# tests/test_lp.py
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.lp import (
    LearningProblem,
    load_learning_problems,
    save_learning_problems,
    split_learning_problems,
)


def test_degenerate_problem_is_rejected() -> None:
    with pytest.raises(ValueError, match="no negative examples"):
        LearningProblem(
            id="x", target_concept="male", pos_example=["a"], neg_example=[],
            complexity=1,
        )


def test_nces_datapoint_uses_local_names(problems) -> None:
    name, examples = problems[0].as_nces_datapoint()
    assert name == "male"
    assert examples["positive examples"] == ["stefan", "markus"]
    assert examples["negative examples"] == ["anna", "michelle"]


def test_round_trip_grouped_by_complexity(problems, tmp_path: Path) -> None:
    path = tmp_path / "learning_problems.json"
    save_learning_problems(problems, path)
    loaded = load_learning_problems(path)
    assert {p.id for p in loaded} == {p.id for p in problems}


def test_split_is_disjoint_and_deterministic(problems) -> None:
    many = [
        LearningProblem(
            id=f"lp_{i:04d}",
            target_concept=f"C{i}",
            pos_example=["a"],
            neg_example=["b"],
            complexity=1,
        )
        for i in range(20)
    ]
    first = split_learning_problems(many, seed=1)
    second = split_learning_problems(many, seed=1)
    assert [p.id for p in first["test"]] == [p.id for p in second["test"]]

    ids = [p.id for group in first.values() for p in group]
    assert len(ids) == len(set(ids)) == 20
