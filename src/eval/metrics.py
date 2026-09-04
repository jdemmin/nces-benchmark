# src/eval/metrics.py
"""Extension-based outcome measurement.

Scoring is performed on reasoner-computed extensions over all individuals,
never on the sampled example sets: a hypothesis that merely enumerates the
provided positives would score perfectly while bearing no relation to the
target concept.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Above this atomic baseline, headroom-normalised ABL is numerically unstable.
ABL_NORM_GUARD = 0.95


def _ratio(numerator: float, denominator: float) -> float:
    """Return 0.0 on an empty denominator rather than raising."""
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class ConfusionMatrix:
    """Per-problem confusion matrix reconstructed from extension counts."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def consistent(self) -> bool:
        """A matrix with any negative cell is internally inconsistent."""
        return min(self.tp, self.fp, self.fn, self.tn) >= 0

    def to_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


def confusion_matrix(
    *,
    hypothesis_extension: Collection[str],
    target_extension: Collection[str],
    universe_size: int,
) -> ConfusionMatrix:
    """Reconstruct the confusion matrix; never clamped.

    Clamping a negative cell would silently manufacture a plausible-looking
    result from a detectable error, so the caller is expected to check
    :attr:`ConfusionMatrix.consistent` and skip inconsistent matrices.
    """
    predicted = frozenset(hypothesis_extension)
    target = frozenset(target_extension)
    tp = len(predicted & target)
    fp = len(predicted) - tp
    fn = len(target) - tp
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=universe_size - tp - fp - fn)


@dataclass(frozen=True)
class ExtensionMetrics:
    """All outcomes for one evaluated learning problem."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    semantic_equivalence: bool
    intersection: int
    union: int
    hypothesis_extension_size: int
    target_extension_size: int
    universe_size: int
    atomic_baseline_f1: float | None
    abl: float | None
    abl_norm: float | None
    empty_hypothesis: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "semantic_equivalence": self.semantic_equivalence,
            "intersection": self.intersection,
            "union": self.union,
            "hypothesis_extension_size": self.hypothesis_extension_size,
            "target_extension_size": self.target_extension_size,
            "universe_size": self.universe_size,
            "atomic_baseline_f1": self.atomic_baseline_f1,
            "abl": self.abl,
            "abl_norm": self.abl_norm,
            "empty_hypothesis": self.empty_hypothesis,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExtensionMetrics:
        return cls(
            precision=float(payload["precision"]),
            recall=float(payload["recall"]),
            f1=float(payload["f1"]),
            accuracy=float(payload["accuracy"]),
            semantic_equivalence=bool(payload["semantic_equivalence"]),
            intersection=int(payload["intersection"]),
            union=int(payload["union"]),
            hypothesis_extension_size=int(payload["hypothesis_extension_size"]),
            target_extension_size=int(payload["target_extension_size"]),
            universe_size=int(payload["universe_size"]),
            atomic_baseline_f1=payload.get("atomic_baseline_f1"),
            abl=payload.get("abl"),
            abl_norm=payload.get("abl_norm"),
            empty_hypothesis=bool(payload["empty_hypothesis"]),
        )


def score_extensions(
    *,
    hypothesis_extension: Collection[str],
    target_extension: Collection[str],
    universe_size: int,
    atomic_baseline: float | None,
) -> ExtensionMetrics:
    """Score one hypothesis against one target over the whole universe.

    ``atomic_baseline`` is a property of the problem and the knowledge base
    alone, so it is identical across embedding conditions and cancels exactly
    in the paired difference.
    """
    predicted = frozenset(hypothesis_extension)
    target = frozenset(target_extension)

    intersection = len(predicted & target)
    union = len(predicted | target)

    precision = _ratio(intersection, len(predicted))
    recall = _ratio(intersection, len(target))
    f1 = _ratio(2 * precision * recall, precision + recall)

    matrix = confusion_matrix(
        hypothesis_extension=predicted,
        target_extension=target,
        universe_size=universe_size,
    )
    accuracy = _ratio(matrix.tp + matrix.tn, universe_size)

    abl = None if atomic_baseline is None else f1 - atomic_baseline
    abl_norm = None
    if atomic_baseline is not None and atomic_baseline <= ABL_NORM_GUARD:
        abl_norm = _ratio(f1 - atomic_baseline, 1.0 - atomic_baseline)

    return ExtensionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        semantic_equivalence=predicted == target,
        intersection=intersection,
        union=union,
        hypothesis_extension_size=len(predicted),
        target_extension_size=len(target),
        universe_size=universe_size,
        atomic_baseline_f1=atomic_baseline,
        abl=abl,
        abl_norm=abl_norm,
        empty_hypothesis=not predicted,
    )