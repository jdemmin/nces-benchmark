# src/benchmarking/metrics.py
"""Extension-based metrics for evaluating NCES hypotheses."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.data.complexity import Complexity
from src.data.results import (
    EmbeddingResult,
    LearningProblemResult,
    MeanMetricsResult,
    MetricsResult,
)


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
    "dl_length": lambda c: c.dl_length,
    "depth": lambda c: c.depth,
    "expressivity": lambda c: c.expressivity,
    "constructors": lambda c: len(c.num_constructors),
    "num_atomic_classes": lambda c: c.num_atomic_classes,
    "num_roles": lambda c: c.num_roles,
    "extension_ratio": lambda c: _ratio_bucket(c.hardness.extension_ratio),
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
    if complexity.hardness.atomic_baseline_f1 is None:
        return None
    return f1 - complexity.hardness.atomic_baseline_f1

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


def mean_embeddings_results(reports: Sequence[EmbeddingResult]) -> MeanMetricsResult:
    """Compute the mean of the metrics across multiple embedding results."""
    metrics_list = [report.mean_metrics for report in reports if report.mean_metrics is not None]
    return mean_results(metrics_list)

def mean_results(records: list[MetricsResult] | list[MeanMetricsResult]) -> MeanMetricsResult:
    """Compute the mean of the metrics across multiple metrics results."""
    
    tmp_records: list[MeanMetricsResult] = []
    if all(isinstance(record, MetricsResult) for record in records):
        for report in records:
            if report is not None:
                tmp_records.append(report.to_mean_metrics()) # type: ignore
    else:
        tmp_records = [report for report in records if report is not None] # type: ignore
    none_reports = [report for report in tmp_records if report is None]
    keys = {
        "mean_accuracy": 0.0,
        "mean_precision": 0.0,
        "mean_recall": 0.0,
        "mean_f1_score": 0.0,
        "mean_jaccard": 0.0,
        "mean_semantic_equivalence": 0.0,
        "mean_intersection": 0.0,
        "mean_union": 0.0,
        "mean_lift": 0.0,
    }
    for entry in tmp_records:
        for key in keys:
            if hasattr(entry, key) and getattr(entry, key) is not None:
                keys[key] += getattr(entry, key)
    for key in keys:
        keys[key] /= (len(tmp_records) - len(none_reports)) if (len(tmp_records)) > 0 else 1
    return MeanMetricsResult(
        mean_accuracy=keys["mean_accuracy"],
        mean_precision=keys["mean_precision"],
        mean_recall=keys["mean_recall"],
        mean_f1_score=keys["mean_f1_score"],
        mean_jaccard=keys["mean_jaccard"],
        mean_semantic_equivalence=keys["mean_semantic_equivalence"],
        mean_intersection=keys["mean_intersection"],
        mean_union=keys["mean_union"],
        mean_lift=keys["mean_lift"]
    )

def _mean(records: list[LearningProblemResult], key: str) -> float:
    """
    Compute the mean of a numeric key in a list of LearningProblemResults.
    Ignores missing keys.
    """
    values = []
    for entry in records:
        entry = getattr(entry.metrics, key, None)
        if entry is not None:
            values.append(float(entry))
    return sum(values) / len(values) if values else 0.0

def _meanSemanticEquivalence(items: list[LearningProblemResult]) -> float:
    """
    Compute the mean of the semantic_equivalence metric in a list of LearningProblemResults.
    Ignores missing keys.
    """
    empty_values = 0
    if items is None or len(items) == 0:
        return 0.0
    value: int = 0
    for entry in items:
        if (
            entry.metrics is not None
            and entry.metrics.semantic_equivalence
        ):
            value += 1
        else:
            empty_values += 1
    return value / (len(items) - empty_values) if (len(items) - empty_values) > 0 else 0.0


def get_complexity_summary(records: list[LearningProblemResult]) -> dict[str, dict[str, MeanMetricsResult]]:
    """
    Compute the mean of each metric in a list of LearningProblemResults,
    grouped by complexity axes.
    """
    summary: dict[str, dict[str, MeanMetricsResult]] = {}
    for axis_name in COMPLEXITY_AXES:
        group = _group_by_complexity(records, axis_name)
        for complexity_value, group_records in group.items():
            mean_group_metrics = mean_results(
                [r.metrics for r in group_records if r.metrics is not None]
            )
            summary.setdefault(axis_name, {}).setdefault(complexity_value, mean_group_metrics)
    return summary


def _group_by_complexity(records: list[LearningProblemResult], axis_name: str) -> dict[Any, list[LearningProblemResult]]:
    """
    Group a list of LearningProblemResults by a complexity axis.
    """
    if axis_name not in COMPLEXITY_AXES:
        raise ValueError(f"Unknown complexity axis: {axis_name}")
    axis_func = COMPLEXITY_AXES[axis_name]
    grouped: dict[Any, list[LearningProblemResult]] = {}
    for record in records:
        key = axis_func(record.learning_problem.complexity)
        grouped.setdefault(key, []).append(record)
    return grouped
