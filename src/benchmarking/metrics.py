# src/benchmarking/metrics.py
"""Extension-based metrics for evaluating NCES hypotheses."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.data.complexity import Complexity


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

#: Bucketings reported in every ``complexity_summary``.
COMPLEXITY_AXES = {
    "by_dl_length": lambda c: c.dl_length,
    "by_depth": lambda c: c.depth,
    "by_expressivity": lambda c: c.expressivity,
    "by_extension_ratio": lambda c: _ratio_bucket(c.extension_ratio),
}

def _ratio_bucket(ratio: float | None) -> str:
    """Bucket an extension ratio into a coarse class-balance band."""
    if ratio is None:
        return "unknown"
    if ratio < 0.05:
        return "rare"
    if ratio < 0.25:
        return "uncommon"
    if ratio < 0.75:
        return "balanced"
    return "dominant"


def compute_lift(f1: float, complexity: Complexity) -> float | None:
    """F1 relative to the best single atomic class.

    ``None`` when the learning problem carries no hardness annotation, since
    the floor is then unknown. Negative values mean the hypothesis
    underperformed a trivial atomic concept.
    """
    if complexity.atomic_baseline_f1 is None:
        return None
    return f1 - complexity.atomic_baseline_f1


def summarize_by_complexity(results: Sequence[dict]) -> dict[str, dict]:
    """Aggregate metrics along every complexity axis."""
    summary: dict[str, dict] = {}
    scored = [r for r in results if "error" not in r]

    for axis_name, key_fn in COMPLEXITY_AXES.items():
        buckets: dict[str, list[dict]] = {}
        for result in scored:
            complexity = Complexity.from_dict(result["complexity"])
            buckets.setdefault(str(key_fn(complexity)), []).append(result)
        summary[axis_name] = {
            bucket: _aggregate_by_complexity(entries) for bucket, entries in sorted(buckets.items())
        }
    return summary

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


def _aggregate_by_complexity(
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
