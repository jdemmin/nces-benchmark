# tests/test_metrics.py
"""Tests for extension-based hypothesis metrics."""

from __future__ import annotations

import math

import pytest

from src.benchmarking.metrics import (
    COMPLEXITY_AXES,
    ExtensionMetrics,
    _ratio,
    _ratio_bucket,
    calculate_metrics,
    compute_lift,
)
from src.data.complexity import Complexity, Hardness, structural_complexity

UNIVERSE = [f"i{n}" for n in range(10)]


def make_complexity(
    *,
    dl_length: int = 3,
    depth: int = 1,
    expressivity: str = "EL",
    extension_ratio: float | None = 0.5,
    atomic_baseline_f1: float | None = 0.4,
    extension_size: int | None = 5,
    redundant: bool | None = False,
) -> Complexity:
    return Complexity(
        dl_length=dl_length,
        depth=depth,
        constructors={},
        num_atomic_classes=1,
        num_roles=0,
        expressivity=expressivity,
        hardness=Hardness(
            extension_size=extension_size,
            extension_ratio=extension_ratio,
            atomic_baseline_f1=atomic_baseline_f1,
            redundant=redundant,
        ),
    )


class TestRatio:
    def test_normal_division(self):
        assert _ratio(1, 4) == 0.25

    def test_zero_denominator_returns_zero(self):
        assert _ratio(5, 0) == 0.0

    def test_zero_numerator(self):
        assert _ratio(0, 7) == 0.0

    def test_returns_float_for_integer_inputs(self):
        result = _ratio(3, 6)
        assert isinstance(result, float)


class TestCalculateMetrics:
    def test_perfect_prediction(self):
        target = {"i0", "i1", "i2"}
        metrics = calculate_metrics(target, target, UNIVERSE)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.jaccard == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.semantic_equivalence is True
        assert metrics.intersection == 3
        assert metrics.union == 3

    def test_disjoint_prediction(self):
        metrics = calculate_metrics({"i0", "i1"}, {"i2", "i3"}, UNIVERSE)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.jaccard == 0.0
        assert metrics.intersection == 0
        assert metrics.union == 4
        assert metrics.semantic_equivalence is False

    def test_partial_overlap(self):
        # predicted = 3, target = 2, overlap = 1
        metrics = calculate_metrics({"i0", "i1", "i2"}, {"i2", "i3"}, UNIVERSE)

        assert metrics.precision == pytest.approx(1 / 3)
        assert metrics.recall == pytest.approx(1 / 2)
        assert metrics.f1 == pytest.approx(0.4)
        assert metrics.jaccard == pytest.approx(1 / 4)
        assert metrics.intersection == 1
        assert metrics.union == 4

    def test_accuracy_counts_true_negatives_over_full_universe(self):
        # tp=1, fp=1, fn=1, universe=10 -> tn=7
        metrics = calculate_metrics({"i0", "i1"}, {"i0", "i2"}, UNIVERSE)
        assert metrics.accuracy == pytest.approx(8 / 10)

    def test_accuracy_penalises_overprediction_outside_examples(self):
        """A hypothesis consistent with samples but too broad must not score 1.0."""
        target = {"i0"}
        broad = set(UNIVERSE)
        metrics = calculate_metrics(broad, target, UNIVERSE)

        assert metrics.recall == 1.0
        assert metrics.accuracy == pytest.approx(1 / 10)
        assert metrics.semantic_equivalence is False

    def test_empty_prediction(self):
        metrics = calculate_metrics(set(), {"i0", "i1"}, UNIVERSE)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.jaccard == 0.0
        assert metrics.accuracy == pytest.approx(8 / 10)

    def test_empty_target(self):
        metrics = calculate_metrics({"i0"}, set(), UNIVERSE)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.union == 1

    def test_both_empty_is_semantically_equivalent(self):
        metrics = calculate_metrics(set(), set(), UNIVERSE)

        assert metrics.semantic_equivalence is True
        assert metrics.accuracy == 1.0
        assert metrics.f1 == 0.0, "F1 is undefined and floors to 0 for empty sets"
        assert metrics.jaccard == 0.0
        assert metrics.union == 0

    def test_empty_universe_falls_back_to_predicted_and_target(self):
        metrics = calculate_metrics({"a"}, {"a"}, [])

        assert metrics.accuracy == 1.0
        assert metrics.f1 == 1.0

    def test_individuals_outside_universe_are_absorbed(self):
        """Unknown individuals extend the universe rather than breaking accuracy."""
        metrics = calculate_metrics({"ghost"}, {"ghost"}, UNIVERSE)

        assert metrics.accuracy == 1.0
        assert metrics.intersection == 1

    def test_accepts_sequences_with_duplicates(self):
        metrics = calculate_metrics(["i0", "i0", "i1"], ["i0", "i1", "i1"], UNIVERSE)

        assert metrics.intersection == 2
        assert metrics.union == 2
        assert metrics.f1 == 1.0

    def test_accepts_generators_are_not_required_but_tuples_work(self):
        metrics = calculate_metrics(("i0",), ("i0",), tuple(UNIVERSE))
        assert metrics.f1 == 1.0

    def test_f1_is_harmonic_mean_of_precision_and_recall(self):
        metrics = calculate_metrics({"i0", "i1", "i2", "i3"}, {"i0", "i1"}, UNIVERSE)
        expected = (
            2
            * metrics.precision
            * metrics.recall
            / (metrics.precision + metrics.recall)
        )
        assert metrics.f1 == pytest.approx(expected)

    def test_all_metrics_within_unit_interval(self):
        metrics = calculate_metrics({"i0", "i5", "i9"}, {"i1", "i5"}, UNIVERSE)
        for value in (
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.jaccard,
        ):
            assert 0.0 <= value <= 1.0

    def test_metrics_are_frozen(self):
        metrics = calculate_metrics({"i0"}, {"i0"}, UNIVERSE)
        with pytest.raises(Exception): # type: ignore[misc]
            metrics.f1 = 0.0  

    def test_to_dict_round_trip(self):
        metrics = calculate_metrics({"i0", "i1"}, {"i1"}, UNIVERSE)
        payload = metrics.to_dict()

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
        assert ExtensionMetrics(**payload) == metrics


class TestRatioBucket:
    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [
            (None, "unknown"),
            (0.0, "rare"),
            (0.049, "rare"),
            (0.05, "uncommon"),
            (0.24, "uncommon"),
            (0.25, "balanced"),
            (0.74, "balanced"),
            (0.75, "dominant"),
            (1.0, "dominant"),
        ],
    )
    def test_boundaries(self, ratio, expected):
        assert _ratio_bucket(ratio) == expected

    def test_buckets_are_exhaustive_and_ordered(self):
        labels = [_ratio_bucket(r) for r in (0.01, 0.1, 0.5, 0.9)]
        assert labels == ["rare", "uncommon", "balanced", "dominant"]


class TestComputeLift:
    def test_none_when_unannotated(self):
        complexity = make_complexity(atomic_baseline_f1=None)
        assert compute_lift(0.9, complexity) is None

    def test_positive_lift(self):
        complexity = make_complexity(atomic_baseline_f1=0.4)
        assert compute_lift(0.9, complexity) == pytest.approx(0.5)

    def test_negative_lift_when_beaten_by_atomic_class(self):
        complexity = make_complexity(atomic_baseline_f1=0.8)
        lift = compute_lift(0.3, complexity)
        assert lift != None
        assert lift == pytest.approx(-0.5)
        assert lift < 0

    def test_zero_lift_when_matching_baseline(self):
        complexity = make_complexity(atomic_baseline_f1=0.6)
        assert compute_lift(0.6, complexity) == pytest.approx(0.0)

    def test_zero_baseline_is_not_treated_as_missing(self):
        complexity = make_complexity(atomic_baseline_f1=0.0)
        assert compute_lift(0.5, complexity) == pytest.approx(0.5)

    def test_uses_real_complexity_from_string_expression(self):
        complexity = structural_complexity("Male ⊓ ∃ hasChild.Female")
        assert complexity.hardness.atomic_baseline_f1 is None
        assert compute_lift(1.0, complexity) is None


# class TestMean:
#     def test_mean_of_floats(self):
#         assert _mean([{"f1": 0.2}, {"f1": 0.4}], "f1") == pytest.approx(0.3)

#     def test_missing_key_defaults_to_zero(self):
#         assert _mean([{"f1": 1.0}, {}], "f1") == pytest.approx(0.5)

#     def test_empty_records(self):
#         assert _mean([], "f1") == 0.0

#     def test_booleans_average_as_a_rate(self):
#         records = [
#             {"semantic_equivalence": True},
#             {"semantic_equivalence": False},
#             {"semantic_equivalence": True},
#             {"semantic_equivalence": False},
#         ]
#         assert _mean(records, "semantic_equivalence") == pytest.approx(0.5)



class TestComplexityAxes:

    def test_key_functions_read_the_right_field(self):
        complexity = make_complexity(
            dl_length=7, depth=3, expressivity="ALC", extension_ratio=0.1
        )
        assert COMPLEXITY_AXES["dl_length"](complexity) == 7
        assert COMPLEXITY_AXES["depth"](complexity) == 3
        assert COMPLEXITY_AXES["expressivity"](complexity) == "ALC"
        assert COMPLEXITY_AXES["extension_ratio"](complexity) == "uncommon"

    def test_extension_ratio_axis_reads_hardness_via_property(self):
        """The axis uses ``c.extension_ratio``, not ``c.hardness.extension_ratio``."""
        complexity = make_complexity(extension_ratio=0.9)
        try:
            bucket = COMPLEXITY_AXES["extension_ratio"](complexity)
        except AttributeError:
            pytest.fail(
                "Complexity exposes no 'extension_ratio' attribute; the axis "
                "should read complexity.hardness.extension_ratio"
            )
        assert bucket == "dominant"

        


# class TestSummarizeByComplexity:
#     @staticmethod
#     def _result(complexity: Complexity, **metrics) -> dict:
#         payload = {
#             "f1": 0.5,
#             "accuracy": 0.5,
#             "precision": 0.5,
#             "recall": 0.5,
#             "jaccard": 0.5,
#             "semantic_equivalence": False,
#         }
#         payload.update(metrics)
#         payload["complexity"] = complexity.to_dict()
#         return payload

#     def test_errored_results_are_excluded(self):
#         results = [{"error": "timeout", "complexity": {"dl_length": 3}}]
#         summary = summarize_by_complexity(results)

#         assert set(summary) == set(COMPLEXITY_AXES)
#         assert all(axis == {} for axis in summary.values())

#     def test_empty_input_yields_empty_axes(self):
#         summary = summarize_by_complexity([])
#         assert summary == {axis: {} for axis in COMPLEXITY_AXES}

#     def test_all_axes_present_for_scored_results(self):
#         results = [self._result(make_complexity())]
#         summary = summarize_by_complexity(results)
#         assert set(summary) == set(COMPLEXITY_AXES)

#     def test_buckets_scored_results_by_dl_length(self):
#         results = [
#             self._result(make_complexity(dl_length=3), f1=1.0),
#             self._result(make_complexity(dl_length=3), f1=0.0),
#             self._result(make_complexity(dl_length=8), f1=0.5),
#         ]
#         summary = summarize_by_complexity(results)
#         assert set(summary["by_dl_length"]) == {"3", "8"}

#     def test_mixed_errored_and_scored(self):
#         results = [
#             self._result(make_complexity(dl_length=4)),
#             {"error": "reasoner failure"},
#         ]
#         summary = summarize_by_complexity(results)
#         assert set(summary["by_dl_length"]) == {"4"}

#     def test_v1_bare_integer_complexity_is_accepted(self):
#         results = [{"complexity": 5, "f1": 1.0}]
#         summary = summarize_by_complexity(results)
#         assert "5" in summary["by_dl_length"]
#         assert summary["by_expressivity"]["unknown"]
#         assert "unknown" in summary["by_extension_ratio"]

#     def test_inner_aggregation_regroups_on_raw_complexity_field(self):
#         """Documents the double-bucketing defect.

#         Outer buckets are keyed by the axis, but ``_aggregate_by_complexity``
#         re-buckets each group on ``record["complexity"]`` — a dict under schema
#         v2 — and then calls ``int()`` on the stringified key.
#         """
#         results = [self._result(make_complexity(dl_length=3))]
#         try:
#             summarize_by_complexity(results)
#         except ValueError:
#             pytest.fail()

#     def test_buckets_map_directly_to_aggregates(self):
#         results = [
#             self._result(make_complexity(dl_length=3), f1=1.0),
#             self._result(make_complexity(dl_length=3), f1=0.0),
#         ]
#         summary = summarize_by_complexity(results)

#         bucket = summary["by_dl_length"]["3"]
#         assert bucket["count"] == 2
#         assert bucket["mean_f1"] == pytest.approx(0.5)


class TestMetricsIntegration:
    def test_metrics_feed_straight_into_lift(self):
        complexity = make_complexity(atomic_baseline_f1=0.5)
        metrics = calculate_metrics({"i0", "i1"}, {"i0", "i1"}, UNIVERSE)

        assert compute_lift(metrics.f1, complexity) == pytest.approx(0.5)

    def test_no_nan_leaks_from_degenerate_inputs(self):
        metrics = calculate_metrics([], [], [])
        for value in metrics.to_dict().values():
            if isinstance(value, float):
                assert not math.isnan(value)