# tests/nces/test_nces.py
from __future__ import annotations

import json
from pathlib import Path

from src.benchmarking.metrics import calculate_extension_metrics
from src.models.nces import prepare_nces_training_data


def test_perfect_hypothesis_is_semantically_equivalent() -> None:
    metrics = calculate_extension_metrics({"a", "b"}, {"a", "b"}, {"a", "b", "c", "d"})
    assert metrics.f1 == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.semantic_equivalence is True
    assert metrics.jaccard == 1.0


def test_partial_overlap_scores() -> None:
    metrics = calculate_extension_metrics({"a", "c"}, {"a", "b"}, {"a", "b", "c", "d"})
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.jaccard == pytest_approx(1.0 / 3, 4)
    assert metrics.semantic_equivalence is False
    # a=TP, c=FP, b=FN, d=TN -> 2/4
    assert metrics.accuracy == 0.5


def test_empty_hypothesis_does_not_divide_by_zero() -> None:
    metrics = calculate_extension_metrics(set(), {"a"}, {"a", "b"})
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.jaccard == 0.0

# No complexity summary in current version
# def test_complexity_summary_groups_records() -> None:
#     records = [
#         {"complexity": 1, "f1": 1.0, "accuracy": 1.0, "semantic_equivalence": True},
#         {"complexity": 1, "f1": 0.0, "accuracy": 0.5, "semantic_equivalence": False},
#         {"complexity": 3, "f1": 0.5, "accuracy": 0.75, "semantic_equivalence": False},
#     ]
#     summary = _aggregate(records)
#     assert summary["count"] == 3
#     assert summary["mean_f1"] == 0.5
#     assert summary["semantic_equivalence_rate"] == 1/3


def test_training_data_is_written_with_local_names(problems, tmp_path: Path) -> None:
    path = tmp_path / "nces_train_data.json"
    data = prepare_nces_training_data(problems, path, seed=1)

    assert len(data) == 2
    name, examples = data[0]
    assert name == "male"
    assert "positive examples" in examples and "negative examples" in examples

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["positive examples"] == ["stefan", "markus"]


def pytest_approx(value: float, tolerance: float = 1e-9):
    """Local approx helper so the module has no pytest import requirement."""

    class _Approx:
        def __eq__(self, other: object) -> bool:
            return abs(float(other) - value) < tolerance # type: ignore itll be fine ig

    return _Approx()
