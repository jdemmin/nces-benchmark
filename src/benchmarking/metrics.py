# src/benchmarking/metrics.py
"""Extension-based metrics for evaluating NCES hypotheses."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.data.complexity import Complexity
from src.data.results import (
    ComplexityStratum,
    EmbeddingResult,
    LearningProblemResult,
    MeanMetricsResult,
    MetricsResult,
    SingleMetric,
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
    "constructors": lambda c: len(c.constructors),
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

def calculate_extension_metrics(
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

    accuracy = round(_ratio(true_positive + true_negative, len(universe)), 4)
    precision = round(_ratio(true_positive, true_positive + false_positive), 4)
    recall = round(_ratio(true_positive, true_positive + false_negative), 4)
    f1 = round(_ratio(2 * precision * recall, precision + recall), 4)
    union = len(predicted_set | target_set)
    jaccard = round(_ratio(true_positive, union), 4)
    semantic_equivalence = predicted_set == target_set

    return ExtensionMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        semantic_equivalence=semantic_equivalence,
        intersection=true_positive,
        union=union,
    )

def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def mean_embeddings_results(reports: Sequence[EmbeddingResult]) -> MeanMetricsResult:
    """Compute the mean of the metrics across multiple embedding results."""
    metrics_list = [report.mean_metrics for report in reports if report.mean_metrics is not None]
    return calculate_metrics(metrics_list)

def calculate_metrics(records: list[MetricsResult] | list[MeanMetricsResult]) -> MeanMetricsResult:
    """Compute the mean of the metrics across multiple metrics results.
    Now, also contains the variance and standard deviation of each metric.
    
    Metric Results are converted to Mean Results so that there are no
    troubles computing the mean, variance, and standard deviation of
    each metric including bools.
    """
    
    tmp_records: list[MeanMetricsResult] = []
    for record in records:
        if record is not None:
            if isinstance(record, MetricsResult):
                tmp_records.append(record.to_mean_metrics()) # type: ignore
            elif isinstance(record, MeanMetricsResult):
                tmp_records.append(record) # type: ignore
    metrics = (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "jaccard",
        "semantic_equivalence_rate",
        "intersection",
        "union",
        "lift",
    )
    counts = {key: 0 for key in metrics}
    means = {key: 0.0 for key in metrics}
    m2 = {key: 0.0 for key in metrics}
    for result in tmp_records:
        result_dict = result.to_dict()
        for key in metrics:
            value = result_dict.get(key)
            if value is None or value["mean"] is None:
                continue
            # Increment number of observations
            counts[key] += 1
            # Difference between new value and old mean
            delta = value["mean"] - means[key]
            # Update mean
            means[key] += delta / counts[key]
            # Difference between new value and updated mean
            delta2 = value["mean"] - means[key]
            # Update sum of squared deviations
            m2[key] += delta * delta2

    # Pass counts to _get_single_metric
    variance = {
        key: m2[key] / (counts[key] - 1)
        if counts[key] > 1
        else 0.0
        for key in metrics
    }
    return MeanMetricsResult(
        accuracy=_get_single_metric(key="accuracy", mean=means, variance=variance, counts=counts),
        precision=_get_single_metric(key="precision", mean=means, variance=variance, counts=counts),
        recall=_get_single_metric(key="recall", mean=means, variance=variance, counts=counts),
        f1_score=_get_single_metric(key="f1_score", mean=means, variance=variance, counts=counts),
        jaccard=_get_single_metric(key="jaccard", mean=means, variance=variance, counts=counts),
        semantic_equivalence_rate=_get_single_metric(key="semantic_equivalence_rate", mean=means, variance=variance, counts=counts),
        intersection=_get_single_metric(key="intersection", mean=means, variance=variance, counts=counts),
        union=_get_single_metric(key="union", mean=means, variance=variance, counts=counts),
        lift=_get_single_metric(key="lift", mean=means, variance=variance, counts=counts),
        lp_count=len(tmp_records)
    )

def _get_single_metric(key: str, mean: dict[str, float], variance: dict[str, float], counts: dict[str, int]) -> SingleMetric:
    return SingleMetric(
        identifier=key,
        mean=round(mean[key], 4),
        variance=round(variance[key], 4),
        std_dev=round(variance[key] ** 0.5, 4),
        # n=counts[key],
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
            mean_group_metrics = calculate_metrics(
                [r.metrics for r in group_records if r.metrics is not None]
            )
            summary.setdefault(axis_name, {}).setdefault(complexity_value, mean_group_metrics)
    return summary

def build_complexity_stratum(stratum_name: str, group: dict[str, MeanMetricsResult]) -> ComplexityStratum:
    return ComplexityStratum(
        stratum_name=stratum_name,
        # The maximum number of learning problems in any bucket for this stratum.
        # Can we arbitrarily choose the maximum as the bucket size?
        bucket_size=max((m.lp_count for m in group.values()), default=0),
        aggragate_per_bucket_value=group if group else None
    )


def build_complexity_strata(group: dict[str, dict[str, MeanMetricsResult]]) -> list[ComplexityStratum]:
    strata: list[ComplexityStratum] = []
    for stratum_name, stratum_group in group.items():
        stratum = build_complexity_stratum(stratum_name, stratum_group)
        if stratum is not None:
            strata.append(stratum)
    return strata


def _group_by_complexity(records: list[LearningProblemResult], axis_name: str) -> dict[Any, list[LearningProblemResult]]:
    """
    Group a list of LearningProblemResults by a complexity axis.
    """
    if axis_name not in COMPLEXITY_AXES:
        raise ValueError(f"Unknown complexity axis: {axis_name}")
    axis_func = COMPLEXITY_AXES[axis_name]
    grouped: dict[Any, list[LearningProblemResult]] = {}
    for record in records:
        if (
            record.learning_problem.complexity is not None
            and record.error is None
        ):
            key = axis_func(record.learning_problem.complexity)
            grouped.setdefault(key, []).append(record)
        if record.metrics is None:
            grouped.setdefault(key, [])
    return grouped
