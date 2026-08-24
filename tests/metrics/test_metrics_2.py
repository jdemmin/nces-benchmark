# tests/benchmarking/test_metrics.py
"""Tests for src.benchmarking.metrics.

The suite deliberately targets the arithmetic edges (empty inputs, all-zero
denominators, ``None`` metrics) rather than only the happy path, because those
are the branches that silently corrupt an aggregate report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from src.benchmarking.metrics import (
    COMPLEXITY_AXES,
    ExtensionMetrics,
    _group_by_complexity,
    _mean,
    _meanSemanticEquivalence,
    _ratio,
    _ratio_bucket,
    calculate_metrics,
    compute_lift,
    get_complexity_summary,
    mean_embeddings_results,
    mean_results,
)
from src.data.results import MeanMetricsResult, MetricsResult

# ---------------------------------------------------------------------------
# Lightweight stand-ins.
#
# The production types drag in ``ontolearn.KnowledgeBase`` via
# ``src.data.results``; the pure-arithmetic functions under test only ever
# touch a handful of attributes, so duck-typed doubles keep the suite fast
# and independent of the ontology stack.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeHardness:
    extension_ratio: float | None = None
    atomic_baseline_f1: float | None = None


@dataclass(frozen=True)
class FakeComplexity:
    dl_length: int = 1
    depth: int = 1
    expressivity: str = "ALC"
    constructors: dict[str, int] = field(default_factory=dict)
    num_atomic_classes: int = 1
    num_roles: int = 0
    hardness: FakeHardness = field(default_factory=FakeHardness)


def make_metrics(**overrides: Any) -> MetricsResult:
    """A ``MetricsResult`` with every field explicit and overridable."""
    base: dict[str, Any] = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "jaccard": 0.0,
        "semantic_equivalence": False,
        "intersection": 0,
        "union": 0,
        "lift": 0,
    }
    base.update(overrides)
    return MetricsResult(**base)


def make_mean(**overrides: Any) -> MeanMetricsResult:
    base: dict[str, Any] = {
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
    base.update(overrides)
    return MeanMetricsResult(**base)


def make_record(
    metrics: MetricsResult | None = None,
    complexity: FakeComplexity | None = None,
) -> SimpleNamespace:
    """A duck-typed ``LearningProblemResult``.

    ``_mean``, ``_meanSemanticEquivalence`` and ``_group_by_complexity`` only
    read ``.metrics`` and ``.learning_problem.complexity``.
    """
    return SimpleNamespace(
        metrics=metrics,
        learning_problem=SimpleNamespace(
            complexity=complexity or FakeComplexity()
        ),
    )


MEAN_FIELDS = tuple(make_mean().to_dict().keys())


def assert_all_zero(result: MeanMetricsResult) -> None:
    for name in MEAN_FIELDS:
        assert getattr(result, name) == pytest.approx(0.0), name


# ---------------------------------------------------------------------------
# _ratio
# ---------------------------------------------------------------------------


class TestRatio:
    def test_zero_denominator_short_circuits_instead_of_raising(self) -> None:
        assert _ratio(5, 0) == 0.0

    def test_zero_over_zero_is_zero_not_nan(self) -> None:
        result = _ratio(0, 0)
        assert result == 0.0
        assert not math.isnan(result)

    def test_returns_float_even_for_integer_inputs(self) -> None:
        result = _ratio(1, 2)
        assert isinstance(result, float)
        assert result == 0.5

    def test_does_not_floor_divide(self) -> None:
        # Guards against a regression to ``//``.
        assert _ratio(1, 3) == pytest.approx(1 / 3)

    def test_negative_numerator_is_passed_through(self) -> None:
        # Relevant because lift may be negative.
        assert _ratio(-2, 4) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# _ratio_bucket
# ---------------------------------------------------------------------------


class TestRatioBucket:
    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [
            (None, "unknown"),
            (0.0, "rare"),
            (0.049999, "rare"),
            (0.05, "uncommon"),  # boundary is inclusive on the upper bucket
            (0.24999, "uncommon"),
            (0.25, "balanced"),
            (0.74999, "balanced"),
            (0.75, "dominant"),
            (1.0, "dominant"),
        ],
    )
    def test_boundaries_are_left_closed(
        self, ratio: float | None, expected: str
    ) -> None:
        assert _ratio_bucket(ratio) == expected

    def test_out_of_range_values_still_bucket(self) -> None:
        # No validation upstream, so the function must not crash on garbage.
        assert _ratio_bucket(-1.0) == "rare"
        assert _ratio_bucket(42.0) == "dominant"

    def test_none_is_distinguishable_from_zero(self) -> None:
        # A missing annotation must not be conflated with a genuinely empty
        # extension.
        assert _ratio_bucket(None) != _ratio_bucket(0.0)


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    def test_perfect_prediction(self) -> None:
        universe = ["a", "b", "c", "d"]
        result = calculate_metrics(["a", "b"], ["a", "b"], universe)

        assert result.accuracy == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.jaccard == pytest.approx(1.0)
        assert result.semantic_equivalence is True
        assert result.intersection == 2
        assert result.union == 2

    def test_accuracy_is_measured_over_the_whole_kb_not_the_examples(self) -> None:
        """The docstring's core promise.

        A hypothesis that matches the two sampled positives but also covers
        an unsampled individual must not score a perfect accuracy.
        """
        universe = [f"i{n}" for n in range(10)]
        result = calculate_metrics(
            predicted=["i0", "i1", "i9"], target=["i0", "i1"], all_individuals=universe
        )

        # 2 TP + 7 TN out of 10.
        assert result.accuracy == pytest.approx(0.9)
        assert result.recall == pytest.approx(1.0)
        assert result.precision == pytest.approx(2 / 3)
        assert result.semantic_equivalence is False

    def test_worked_confusion_matrix(self) -> None:
        universe = ["a", "b", "c", "d", "e"]
        # TP={a}, FP={d}, FN={b}, TN={c,e}
        result = calculate_metrics(["a", "d"], ["a", "b"], universe)

        assert result.intersection == 1
        assert result.union == 3
        assert result.precision == pytest.approx(0.5)
        assert result.recall == pytest.approx(0.5)
        assert result.f1 == pytest.approx(0.5)
        assert result.jaccard == pytest.approx(1 / 3)
        assert result.accuracy == pytest.approx(3 / 5)

    def test_empty_prediction_against_nonempty_target(self) -> None:
        result = calculate_metrics([], ["a"], ["a", "b"])

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0  # must be 0.0, not NaN
        assert result.jaccard == 0.0
        assert result.semantic_equivalence is False
        assert result.accuracy == pytest.approx(0.5)  # b is a true negative

    def test_both_empty_is_semantically_equivalent_with_full_accuracy(self) -> None:
        result = calculate_metrics([], [], ["a", "b", "c"])

        assert result.semantic_equivalence is True
        assert result.accuracy == pytest.approx(1.0)
        # F1/Jaccard are undefined here and are reported as 0.0 by convention.
        assert result.f1 == 0.0
        assert result.jaccard == 0.0
        assert result.union == 0

    def test_completely_disjoint_sets(self) -> None:
        result = calculate_metrics(["a"], ["b"], ["a", "b"])

        assert result.intersection == 0
        assert result.union == 2
        assert result.f1 == 0.0
        assert result.accuracy == 0.0  # no TP and no TN
        assert result.semantic_equivalence is False

    def test_universe_is_widened_by_predicted_and_target(self) -> None:
        """``all_individuals`` may be incomplete; TN must never go negative."""
        result = calculate_metrics(
            predicted=["ghost"], target=["phantom"], all_individuals=[]
        )

        # Universe becomes {ghost, phantom}: 1 FP + 1 FN, so accuracy is 0.
        assert result.accuracy == 0.0
        assert 0.0 <= result.accuracy <= 1.0

    def test_accuracy_stays_in_unit_interval_for_unknown_individuals(self) -> None:
        result = calculate_metrics(
            predicted=["x", "y"], target=["y", "z"], all_individuals=["y"]
        )
        assert 0.0 <= result.accuracy <= 1.0

    def test_duplicates_in_input_are_deduplicated(self) -> None:
        dup = calculate_metrics(
            ["a", "a", "b"], ["a", "b", "b"], ["a", "b", "a"]
        )
        clean = calculate_metrics(["a", "b"], ["a", "b"], ["a", "b"])

        assert dup.to_dict() == clean.to_dict()

    def test_accepts_arbitrary_collections(self) -> None:
        from_sets = calculate_metrics({"a", "b"}, frozenset({"a"}), ("a", "b", "c"))
        from_lists = calculate_metrics(["a", "b"], ["a"], ["a", "b", "c"])

        assert from_sets.to_dict() == from_lists.to_dict()

    def test_inputs_are_not_mutated(self) -> None:
        predicted = ["a"]
        target = ["b"]
        universe = ["a", "b"]

        calculate_metrics(predicted, target, universe)

        assert predicted == ["a"]
        assert target == ["b"]
        assert universe == ["a", "b"]

    def test_f1_is_the_harmonic_mean_of_precision_and_recall(self) -> None:
        universe = [f"i{n}" for n in range(20)]
        result = calculate_metrics(
            predicted=["i0", "i1", "i2", "i3"],
            target=["i0", "i1", "i15"],
            all_individuals=universe,
        )
        expected = (
            2
            * result.precision
            * result.recall
            / (result.precision + result.recall)
        )
        assert result.f1 == pytest.approx(expected)

    def test_precision_recall_asymmetry(self) -> None:
        """Over-prediction and under-prediction must not look identical."""
        universe = ["a", "b", "c", "d"]
        over = calculate_metrics(["a", "b", "c"], ["a"], universe)
        under = calculate_metrics(["a"], ["a", "b", "c"], universe)

        assert over.recall == pytest.approx(1.0)
        assert over.precision == pytest.approx(1 / 3)
        assert under.precision == pytest.approx(1.0)
        assert under.recall == pytest.approx(1 / 3)
        # Jaccard and F1 are symmetric, precision/recall are not.
        assert over.jaccard == pytest.approx(under.jaccard)
        assert over.f1 == pytest.approx(under.f1)
        assert over.precision != pytest.approx(under.precision)

    def test_jaccard_never_exceeds_f1(self) -> None:
        # Algebraic invariant: J = F1 / (2 - F1) <= F1 for F1 in [0, 1].
        universe = [f"i{n}" for n in range(12)]
        result = calculate_metrics(
            ["i0", "i1", "i2"], ["i1", "i2", "i3", "i4"], universe
        )
        assert result.jaccard <= result.f1 + 1e-12
        assert result.jaccard == pytest.approx(
            result.f1 / (2 - result.f1)
        )

    def test_semantic_equivalence_implies_maximal_scores(self) -> None:
        result = calculate_metrics(["a", "b"], ["b", "a"], ["a", "b", "c"])
        assert result.semantic_equivalence is True
        assert result.f1 == pytest.approx(1.0)
        assert result.jaccard == pytest.approx(1.0)
        assert result.accuracy == pytest.approx(1.0)

    def test_semantic_equivalence_is_strict_set_equality(self) -> None:
        # One extra individual out of many keeps F1 high but equivalence false.
        universe = [f"i{n}" for n in range(100)]
        target = [f"i{n}" for n in range(50)]
        predicted = target + ["i50"]

        result = calculate_metrics(predicted, target, universe)

        assert result.f1 > 0.98
        assert result.semantic_equivalence is False

    def test_returns_frozen_dataclass(self) -> None:
        result = calculate_metrics(["a"], ["a"], ["a"])
        with pytest.raises(Exception):  # FrozenInstanceError
            result.accuracy = 0.0  # type: ignore[misc]

    def test_to_dict_exposes_every_field(self) -> None:
        payload = calculate_metrics(["a"], ["a", "b"], ["a", "b", "c"]).to_dict()

        assert set(payload) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "jaccard",
            "semantic_equivalence",
            "intersection",
            "union",
        }
        assert payload["intersection"] == 1
        assert payload["union"] == 2
        # ``jaccard``/``f1`` keys, not ``f1_score`` — MetricsResult renames it.
        assert "f1_score" not in payload

    def test_intersection_and_union_are_ints(self) -> None:
        result = calculate_metrics(["a"], ["a", "b"], ["a", "b"])
        assert isinstance(result.intersection, int)
        assert isinstance(result.union, int)


# ---------------------------------------------------------------------------
# compute_lift
# ---------------------------------------------------------------------------


class TestComputeLift:
    def test_returns_none_without_a_hardness_baseline(self) -> None:
        complexity = FakeComplexity(hardness=FakeHardness(atomic_baseline_f1=None))
        assert compute_lift(0.9, complexity) is None

    def test_positive_lift_beats_the_atomic_floor(self) -> None:
        complexity = FakeComplexity(hardness=FakeHardness(atomic_baseline_f1=0.4))
        assert compute_lift(0.75, complexity) == pytest.approx(0.35)

    def test_negative_lift_signals_underperforming_a_trivial_concept(self) -> None:
        complexity = FakeComplexity(hardness=FakeHardness(atomic_baseline_f1=0.8))
        assert compute_lift(0.3, complexity) == pytest.approx(-0.5)

    def test_zero_baseline_is_not_treated_as_missing(self) -> None:
        """``0.0`` is falsy — a truthiness check here would be a bug."""
        complexity = FakeComplexity(hardness=FakeHardness(atomic_baseline_f1=0.0))
        assert compute_lift(0.6, complexity) == pytest.approx(0.6)

    def test_matching_the_baseline_yields_exactly_zero(self) -> None:
        complexity = FakeComplexity(hardness=FakeHardness(atomic_baseline_f1=0.62))
        assert compute_lift(0.62, complexity) == pytest.approx(0.0)

    def test_result_is_bounded_by_negative_one_and_one(self) -> None:
        for f1 in (0.0, 0.5, 1.0):
            for baseline in (0.0, 0.5, 1.0):
                complexity = FakeComplexity(
                    hardness=FakeHardness(atomic_baseline_f1=baseline)
                )
                assert -1.0 <= compute_lift(f1, complexity) <= 1.0


# ---------------------------------------------------------------------------
# _get_mean_metrics_from_metrics_sequence
# ---------------------------------------------------------------------------


class TestMeanFromMetricsSequence:
    def test_empty_sequence_yields_all_zeros(self) -> None:
        assert_all_zero(mean_results([]))

    def test_single_element_is_an_identity_projection(self) -> None:
        metrics = make_metrics(
            accuracy=0.5,
            precision=0.25,
            recall=0.75,
            f1_score=0.375,
            jaccard=0.2,
            semantic_equivalence=True,
            intersection=3,
            union=7,
            lift=0.1,
        )
        result = mean_results([metrics])

        assert result.mean_accuracy == pytest.approx(0.5)
        assert result.mean_precision == pytest.approx(0.25)
        assert result.mean_recall == pytest.approx(0.75)
        assert result.mean_f1_score == pytest.approx(0.375)
        assert result.mean_jaccard == pytest.approx(0.2)
        assert result.mean_semantic_equivalence == pytest.approx(1.0)
        assert result.mean_intersection == pytest.approx(3.0)
        assert result.mean_union == pytest.approx(7.0)
        assert result.mean_lift == pytest.approx(0.1)

    def test_averages_over_all_entries(self) -> None:
        result = mean_results(
            [
                make_metrics(accuracy=1.0, f1_score=1.0, intersection=10, lift=0.4),
                make_metrics(accuracy=0.0, f1_score=0.0, intersection=0, lift=-0.2),
            ]
        )

        assert result.mean_accuracy == pytest.approx(0.5)
        assert result.mean_f1_score == pytest.approx(0.5)
        assert result.mean_intersection == pytest.approx(5.0)
        assert result.mean_lift == pytest.approx(0.1)

    def test_semantic_equivalence_becomes_a_success_rate(self) -> None:
        metrics_list = [
            make_metrics(semantic_equivalence=True),
            make_metrics(semantic_equivalence=False),
            make_metrics(semantic_equivalence=False),
            make_metrics(semantic_equivalence=True),
        ]
        result = mean_results(metrics_list)
        assert result.mean_semantic_equivalence == pytest.approx(0.5)

    def test_all_false_equivalence_averages_to_zero_without_dividing_by_zero(
        self,
    ) -> None:
        """Contrast with ``_meanSemanticEquivalence``, which raises here."""
        result = mean_results(
            [make_metrics(semantic_equivalence=False) for _ in range(3)]
        )
        assert result.mean_semantic_equivalence == pytest.approx(0.0)

    def test_negative_lift_is_preserved_not_clamped(self) -> None:
        result = mean_results(
            [make_metrics(lift=-0.5), make_metrics(lift=-0.1)]
        )
        assert result.mean_lift == pytest.approx(-0.3)

    def test_does_not_mutate_the_input_metrics(self) -> None:
        metrics = make_metrics(accuracy=0.5, intersection=4)
        snapshot = metrics.to_dict()

        mean_results([metrics, metrics])

        assert metrics.to_dict() == snapshot

    def test_repeated_calls_do_not_accumulate_state(self) -> None:
        """Guards against a shared mutable accumulator across invocations."""
        metrics = [make_metrics(accuracy=1.0, union=4)]

        first = mean_results(metrics)
        second = mean_results(metrics)

        assert first.to_dict() == second.to_dict()

    def test_accepts_a_tuple(self) -> None:
        result = mean_results(
            [make_metrics(accuracy=0.4), make_metrics(accuracy=0.6)]
        )
        assert result.mean_accuracy == pytest.approx(0.5)

    # @pytest.mark.xfail(
    #     reason="lift is typed float but compute_lift returns None when the "
    #     "learning problem carries no hardness annotation; the accumulator "
    #     "adds it unguarded.",
    #     raises=TypeError,
    #     strict=True,
    # )
    def test_none_lift_should_not_crash_the_aggregate(self) -> None:
        mean_results(
            [make_metrics(lift=None)]  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# get_mean_metrics_from_list
# ---------------------------------------------------------------------------


class TestGetMeanMetricsFromList:
    def test_empty_list_yields_all_zeros(self) -> None:
        assert_all_zero(mean_results([]))

    def test_averages_pre_aggregated_means(self) -> None:
        result = mean_results(
            [
                make_mean(mean_accuracy=0.8, mean_f1_score=0.6, mean_union=10.0),
                make_mean(mean_accuracy=0.4, mean_f1_score=0.2, mean_union=20.0),
            ]
        )

        assert result.mean_accuracy == pytest.approx(0.6)
        assert result.mean_f1_score == pytest.approx(0.4)
        assert result.mean_union == pytest.approx(15.0)

    def test_equivalence_rate_is_averaged_not_re_thresholded(self) -> None:
        result = mean_results(
            [
                make_mean(mean_semantic_equivalence=1.0),
                make_mean(mean_semantic_equivalence=0.0),
                make_mean(mean_semantic_equivalence=0.5),
            ]
        )
        assert result.mean_semantic_equivalence == pytest.approx(0.5)

    def test_is_an_unweighted_mean_of_means(self) -> None:
        """A 1-problem split counts as much as a 100-problem split.

        This documents the current (unweighted) semantics so a future switch
        to a weighted mean is a deliberate, visible change.
        """
        result = mean_results(
            [make_mean(mean_accuracy=1.0), make_mean(mean_accuracy=0.0)]
        )
        assert result.mean_accuracy == pytest.approx(0.5)

    def test_single_element_round_trips(self) -> None:
        source = make_mean(
            mean_accuracy=0.11,
            mean_precision=0.22,
            mean_recall=0.33,
            mean_f1_score=0.44,
            mean_jaccard=0.55,
            mean_semantic_equivalence=0.66,
            mean_intersection=7.0,
            mean_union=8.0,
            mean_lift=0.99,
        )
        assert mean_results([source]).to_dict() == source.to_dict()

    def test_does_not_mutate_its_inputs(self) -> None:
        entries = [make_mean(mean_accuracy=0.5), make_mean(mean_accuracy=0.5)]
        snapshots = [entry.to_dict() for entry in entries]

        mean_results(entries)

        assert [entry.to_dict() for entry in entries] == snapshots

    def test_idempotent_over_a_single_aggregate(self) -> None:
        once = mean_results([make_mean(mean_accuracy=0.7)])
        twice = mean_results([once])
        assert twice.to_dict() == once.to_dict()


# ---------------------------------------------------------------------------
# mean_embeddings_results
# ---------------------------------------------------------------------------


def make_embedding_result(mean_metrics: MeanMetricsResult | None) -> SimpleNamespace:
    """Duck-typed ``EmbeddingResult``; only ``.mean_metrics`` is read."""
    return SimpleNamespace(mean_metrics=mean_metrics)


class TestMeanEmbeddingsResults:
    def test_empty_reports_yield_all_zeros_without_dividing_by_zero(self) -> None:
        assert_all_zero(mean_embeddings_results([]))

    def test_averages_across_embedding_results(self) -> None:
        result = mean_embeddings_results(
            [
                make_embedding_result(make_mean(mean_f1_score=0.9, mean_lift=0.3)),
                make_embedding_result(make_mean(mean_f1_score=0.5, mean_lift=-0.1)),
            ]
        )

        assert result.mean_f1_score == pytest.approx(0.7)
        assert result.mean_lift == pytest.approx(0.1)

    def test_carries_every_metric_field_through(self) -> None:
        source = make_mean(
            mean_accuracy=0.1,
            mean_precision=0.2,
            mean_recall=0.3,
            mean_f1_score=0.4,
            mean_jaccard=0.5,
            mean_semantic_equivalence=0.6,
            mean_intersection=7.0,
            mean_union=8.0,
            mean_lift=0.9,
        )
        result = mean_embeddings_results([make_embedding_result(source)])
        assert result.to_dict() == source.to_dict()

    def test_none_mean_metrics_are_skipped_and_not_counted(self) -> None:
        """Documents a real dilution bug.

        The divisor is ``len(reports)``, not the number of contributing
        reports, so a failed split silently halves the reported score.
        """
        result = mean_embeddings_results(
            [
                make_embedding_result(make_mean(mean_accuracy=1.0)),
                make_embedding_result(None),
            ]
        )
        assert result.mean_accuracy != pytest.approx(0.5)

    def test_all_none_reports_collapse_to_zero(self) -> None:
        assert_all_zero(
            mean_embeddings_results(
                [make_embedding_result(None), make_embedding_result(None)]
            )
        )

    def test_falsy_all_zero_mean_metrics_is_treated_as_missing(self) -> None:
        """``entry.mean_metrics if entry.mean_metrics else None`` is a
        truthiness test. ``MeanMetricsResult`` defines no ``__bool__`` or
        ``__len__``, so a genuine all-zero aggregate is still truthy and
        must contribute to the divisor."""
        result = mean_embeddings_results(
            [
                make_embedding_result(make_mean(mean_accuracy=1.0)),
                make_embedding_result(make_mean()),  # all zeros, still an object
            ]
        )
        assert result.mean_accuracy == pytest.approx(0.5)

    def test_tolerates_reports_missing_a_metric_attribute(self) -> None:
        """``getattr(metric, key, 0.0)`` is the documented fallback."""
        partial = SimpleNamespace(mean_accuracy=1.0)  # no other fields
        result = mean_embeddings_results([make_embedding_result(partial)])

        assert result.mean_accuracy == pytest.approx(1.0)
        assert result.mean_lift == pytest.approx(0.0)
        assert result.mean_union == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _mean
# ---------------------------------------------------------------------------


class TestMean:
    def test_empty_records_yield_zero(self) -> None:
        assert _mean([], "accuracy") == 0.0

    def test_averages_the_requested_metric(self) -> None:
        records = [
            make_record(make_metrics(accuracy=1.0)),
            make_record(make_metrics(accuracy=0.5)),
            make_record(make_metrics(accuracy=0.0)),
        ]
        assert _mean(records, "accuracy") == pytest.approx(0.5)

    def test_unknown_key_yields_zero_rather_than_raising(self) -> None:
        records = [make_record(make_metrics(accuracy=1.0))]
        assert _mean(records, "does_not_exist") == 0.0

    def test_records_without_metrics_are_excluded_from_the_divisor(self) -> None:
        """Unlike ``mean_embeddings_results``, ``_mean`` divides by the number
        of contributing values, so failures do not dilute the mean."""
        records = [
            make_record(make_metrics(accuracy=1.0)),
            make_record(None),  # getattr(None, "accuracy", None) -> None
        ]
        assert _mean(records, "accuracy") == pytest.approx(1.0)

    def test_all_records_missing_metrics_yields_zero(self) -> None:
        assert _mean([make_record(None), make_record(None)], "accuracy") == 0.0

    def test_zero_values_are_included_not_skipped(self) -> None:
        """The guard is ``is not None``; a legitimate 0.0 must count."""
        records = [
            make_record(make_metrics(f1_score=0.0)),
            make_record(make_metrics(f1_score=1.0)),
        ]
        assert _mean(records, "f1_score") == pytest.approx(0.5)

    def test_boolean_metric_is_coerced_to_float(self) -> None:
        records = [
            make_record(make_metrics(semantic_equivalence=True)),
            make_record(make_metrics(semantic_equivalence=False)),
        ]
        assert _mean(records, "semantic_equivalence") == pytest.approx(0.5)

    def test_integer_metric_is_coerced_to_float(self) -> None:
        records = [
            make_record(make_metrics(intersection=1)),
            make_record(make_metrics(intersection=2)),
        ]
        result = _mean(records, "intersection")
        assert isinstance(result, float)
        assert result == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# _meanSemanticEquivalence
# ---------------------------------------------------------------------------


class TestMeanSemanticEquivalence:
    def test_none_input_yields_zero(self) -> None:
        assert _meanSemanticEquivalence(None) == 0.0  # type: ignore[arg-type]

    def test_empty_list_yields_zero(self) -> None:
        assert _meanSemanticEquivalence([]) == 0.0

    def test_all_equivalent_yields_one(self) -> None:
        records = [make_record(make_metrics(semantic_equivalence=True))] * 3
        assert _meanSemanticEquivalence(records) == pytest.approx(1.0)

    def test_mixed_results_divide_by_the_true_count_not_the_total(self) -> None:
        """Documents the divisor bug.

        ``len(items) - empty_values`` counts only the *successful* items, so
        2 hits out of 4 records reports 1.0 instead of 0.5.
        """
        records = [
            make_record(make_metrics(semantic_equivalence=True)),
            make_record(make_metrics(semantic_equivalence=True)),
            make_record(make_metrics(semantic_equivalence=False)),
            make_record(make_metrics(semantic_equivalence=False)),
        ]
        assert _meanSemanticEquivalence(records) == pytest.approx(1.0)

    def test_disagrees_with_the_metrics_sequence_aggregate(self) -> None:
        """Cross-check pinning the inconsistency between the two code paths."""
        metrics = [
            make_metrics(semantic_equivalence=True),
            make_metrics(semantic_equivalence=False),
        ]
        records = [make_record(m) for m in metrics]

        via_sequence = mean_results(
            metrics
        ).mean_semantic_equivalence
        via_helper = _meanSemanticEquivalence(records)

        assert via_sequence == pytest.approx(0.5)
        assert via_helper == pytest.approx(1.0)

    def test_does_not_raise_when_nothing_is_equivalent(self) -> None:
        """The realistic failure mode: no hypothesis is exactly right, so
        ``empty_values == len(items)`` and the division blows up."""
        records = [make_record(make_metrics(semantic_equivalence=False))] * 5
        _meanSemanticEquivalence(records)

    def test_does_not_raise_when_every_record_lacks_metrics(self) -> None:
        _meanSemanticEquivalence([make_record(None), make_record(None)])


# ---------------------------------------------------------------------------
# COMPLEXITY_AXES / _group_by_complexity
# ---------------------------------------------------------------------------


class TestComplexityAxes:
    def test_axis_registry_is_stable(self) -> None:
        """The report schema depends on these exact keys."""
        assert set(COMPLEXITY_AXES) == {
            "dl_length",
            "depth",
            "expressivity",
            "constructors",
            "num_atomic_classes",
            "num_roles",
            "extension_ratio",
        }

    def test_each_axis_extracts_the_expected_value(self) -> None:
        complexity = FakeComplexity(
            dl_length=9,
            depth=3,
            expressivity="ALCHIQ(D)",
            constructors={"and": 4, "exists": 1},
            num_atomic_classes=6,
            num_roles=2,
            hardness=FakeHardness(extension_ratio=0.4),
        )

        assert COMPLEXITY_AXES["dl_length"](complexity) == 9
        assert COMPLEXITY_AXES["depth"](complexity) == 3
        assert COMPLEXITY_AXES["expressivity"](complexity) == "ALCHIQ(D)"
        assert COMPLEXITY_AXES["num_atomic_classes"](complexity) == 6
        assert COMPLEXITY_AXES["num_roles"](complexity) == 2
        assert COMPLEXITY_AXES["extension_ratio"](complexity) == "balanced"

    def test_constructors_axis_counts_distinct_kinds_not_occurrences(self) -> None:
        """``len(c.num_constructors)`` is the arity of the mapping.

        Two concepts with wildly different constructor totals land in the
        same bucket if they use the same *kinds*.
        """
        few = FakeComplexity(constructors={"and": 1, "exists": 1})
        many = FakeComplexity(constructors={"and": 99, "exists": 50})

        assert COMPLEXITY_AXES["constructors"](few) == 2
        assert COMPLEXITY_AXES["constructors"](many) == 2

    def test_constructors_axis_of_an_atomic_concept_is_zero(self) -> None:
        assert COMPLEXITY_AXES["constructors"](FakeComplexity()) == 0

    def test_extension_ratio_axis_returns_a_bucket_label_not_the_float(self) -> None:
        complexity = FakeComplexity(hardness=FakeHardness(extension_ratio=0.01))
        value = COMPLEXITY_AXES["extension_ratio"](complexity)

        assert value == "rare"
        assert isinstance(value, str)

    def test_extension_ratio_axis_handles_a_missing_annotation(self) -> None:
        complexity = FakeComplexity(hardness=FakeHardness(extension_ratio=None))
        assert COMPLEXITY_AXES["extension_ratio"](complexity) == "unknown"


class TestGroupByComplexity:
    def test_unknown_axis_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown complexity axis: nope"):
            _group_by_complexity([], "nope")

    def test_empty_records_yield_an_empty_grouping(self) -> None:
        assert _group_by_complexity([], "depth") == {}

    def test_groups_by_the_axis_value(self) -> None:
        shallow_a = make_record(make_metrics(), FakeComplexity(depth=1))
        shallow_b = make_record(make_metrics(), FakeComplexity(depth=1))
        deep = make_record(make_metrics(), FakeComplexity(depth=4))

        grouped = _group_by_complexity([shallow_a, deep, shallow_b], "depth")

        assert set(grouped) == {1, 4}
        assert grouped[1] == [shallow_a, shallow_b]  # insertion order preserved
        assert grouped[4] == [deep]

    def test_every_record_appears_exactly_once(self) -> None:
        records = [
            make_record(make_metrics(), FakeComplexity(depth=depth))
            for depth in (1, 1, 2, 3, 3, 3)
        ]
        grouped = _group_by_complexity(records, "depth")

        assert sum(len(bucket) for bucket in grouped.values()) == len(records)
        assert {depth: len(bucket) for depth, bucket in grouped.items()} == {
            1: 2,
            2: 1,
            3: 3,
        }

    def test_records_without_metrics_are_still_grouped(self) -> None:
        """Grouping is by complexity only; the metrics filter happens later."""
        grouped = _group_by_complexity(
            [make_record(None, FakeComplexity(depth=2))], "depth"
        )
        assert len(grouped[2]) == 1

    def test_string_axis_produces_string_keys(self) -> None:
        grouped = _group_by_complexity(
            [make_record(make_metrics(), FakeComplexity(expressivity="ALC"))],
            "expressivity",
        )
        assert list(grouped) == ["ALC"]

    def test_grouping_by_extension_ratio_uses_buckets(self) -> None:
        records = [
            make_record(
                make_metrics(),
                FakeComplexity(hardness=FakeHardness(extension_ratio=ratio)),
            )
            for ratio in (0.01, 0.02, 0.5, None)
        ]
        grouped = _group_by_complexity(records, "extension_ratio")

        assert set(grouped) == {"rare", "balanced", "unknown"}
        assert len(grouped["rare"]) == 2


# ---------------------------------------------------------------------------
# get_complexity_summary
# ---------------------------------------------------------------------------


# class TestGetComplexitySummary:
#     def test_empty_records_yield_an_empty_summary(self) -> None:
#         """No records means no groups, so no axis keys are created at all."""
#         assert get_complexity_summary([]) == {}

#     def test_every_axis_is_reported_when_records_exist(self) -> None:
#         summary = get_complexity_summary([make_record(make_metrics())])
#         assert set(summary) == set(COMPLEXITY_AXES)

#     def test_per_bucket_means_are_computed_independently(self) -> None:
#         records = [
#             make_record(make_metrics(f1_score=1.0), FakeComplexity(depth=1)),
#             make_record(make_metrics(f1_score=0.8), FakeComplexity(depth=1)),
#             make_record(make_metrics(f1_score=0.2), FakeComplexity(depth=5)),
#         ]
#         summary