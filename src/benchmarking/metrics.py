# src/benchmarking/metrics.py
"""Extension-based metrics for evaluating NCES hypotheses."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExtensionMetrics:
    """Metrics comparing a hypothesis extension to a target extension."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    jaccard: float
    semantic_equivalence: bool
    intersection: int
    union: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_metrics(
    predicted: Collection[str],
    target: Collection[str],
    all_individuals: Collection[str],
) -> ExtensionMetrics:
    """Compare a hypothesis extension against the target extension.

    Accuracy is measured over every individual in the knowledge base, so a
    hypothesis that is merely consistent with the sampled examples cannot
    score perfectly by accident.
    """
    predicted_set = set(predicted)
    target_set = set(target)
    universe = set(all_individuals) | predicted_set | target_set

    true_positive = len(predicted_set & target_set)
    false_positive = len(predicted_set - target_set)
    false_negative = len(target_set - predicted_set)
    true_negative = len(universe) - true_positive - false_positive - false_negative

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    union = len(predicted_set | target_set)

    return ExtensionMetrics(
        accuracy=_ratio(true_positive + true_negative, len(universe)),
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        jaccard=_ratio(true_positive, union),
        semantic_equivalence=predicted_set == target_set,
        intersection=true_positive,
        union=union,
    )


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def aggregate_by_complexity(
    records: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Build the ``complexity_summary``: per-complexity aggregate metrics."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get("complexity", 0)), []).append(record)

    summary: dict[str, dict[str, float]] = {}
    for complexity, group in sorted(buckets.items(), key=lambda kv: int(kv[0])):
        summary[complexity] = {
            "count": len(group),
            "mean_f1": _mean(group, "f1"),
            "mean_accuracy": _mean(group, "accuracy"),
            "mean_precision": _mean(group, "precision"),
            "mean_recall": _mean(group, "recall"),
            "mean_jaccard": _mean(group, "jaccard"),
            "semantic_equivalence_rate": _mean(group, "semantic_equivalence"),
        }
    return summary


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0)) for record in records]
    return sum(values) / len(values) if values else 0.0
