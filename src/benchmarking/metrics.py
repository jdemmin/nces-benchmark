# src/benchmarking/metrics.py
"""Extension-based metrics for evaluating NCES hypotheses."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.data.complexity import Complexity
from src.data.results import EmbeddingResult, MeanMetricsResult


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
    "by_extension_ratio": lambda c: _ratio_bucket(c.hardness.extension_ratio),
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


def _mean_embeddings_results(reports: Sequence[EmbeddingResult]) -> MeanMetricsResult:
    """Compute the mean of the metrics across multiple embedding results."""
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
    for entry in reports:
        metric = entry.mean_metrics if entry.mean_metrics else None
        if metric is None:
            continue
        for key in keys:
            keys[key] += getattr(metric, key, 0.0)
    for key in keys:
        keys[key] /= len(reports) if reports else 1
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
