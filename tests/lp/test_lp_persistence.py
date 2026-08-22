# tests/lp/test_lp_persistence.py
"""Grouped artifact and split persistence."""

from __future__ import annotations

import json
from pathlib import Path

from test_lp_schema import blank_complexity, make_problem

from src.data.lp import (
    LearningProblem,
    load_learning_problems,
    save_learning_problems,
    save_split,
)


def test_grouping_key_is_dl_length(tmp_path: Path) -> None:
    problems = [
        make_problem(0, complexity=blank_complexity(dl_length=1)),
        make_problem(1, complexity=blank_complexity(dl_length=4)),
        make_problem(2, complexity=blank_complexity(dl_length=4)),
    ]
    path = tmp_path / "learning_problems.json"
    save_learning_problems(problems, path)

    grouped = json.loads(path.read_text(encoding="utf-8"))
    assert set(grouped) == {"1", "4"}
    assert len(grouped["4"]) == 2


def test_round_trip_is_lossless(tmp_path: Path) -> None:
    problems = [
        make_problem(i, complexity=blank_complexity(dl_length=i % 3 + 1))
        for i in range(9)
    ]
    path = tmp_path / "learning_problems.json"
    save_learning_problems(problems, path)
    assert sorted(load_learning_problems(path), key=lambda p: p.id) == sorted(
        problems, key=lambda p: p.id
    )


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "learning_problems.json"
    save_learning_problems([make_problem()], path)
    assert path.is_file()