"""Tests for extension-based outcome measurement.

The properties asserted here mirror the methodology's load-bearing
guarantees: the confusion matrix is never clamped, ratios never raise on
empty denominators, and the atomic baseline cancels exactly in the paired
difference that constitutes the estimand.
"""

from __future__ import annotations

import math

import pytest

from src.eval.metrics import (
    ABL_NORM_GUARD,
    ConfusionMatrix,
    ExtensionMetrics,
    _ratio,
    confusion_matrix,
    score_extensions,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

UNIVERSE = 100


def score(
    predicted: set[str] | list[str],
    target: set[str] | list[str],
    *,
    universe_size: int = UNIVERSE,
    baseline: float | None = None,
) -> ExtensionMetrics:
    return score_extensions(
        hypothesis_extension=predicted,
        target_extension=target,
        universe_size=universe_size,
        atomic_baseline=baseline,
    )


def individuals(prefix: str, count: int) -> set[str]:
    return {f"http://example.org/kb#{prefix}{i}" for i in range(count)}


# --------------------------------------------------------------------------
# _ratio
# --------------------------------------------------------------------------


class TestRatio:
    """An empty denominator must yield 0.0, never an exception."""

    @pytest.mark.parametrize(
        ("numerator", "denominator", "expected"),
        [
            (1.0, 2.0, 0.5),
            (0.0, 4.0, 0.0),
            (3.0, 3.0, 1.0),
            (7.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (-2.0, 4.0, -0.5),
        ],
    )
    def test_returns_expected_value(
        self, numerator: float, denominator: float, expected: float
    ) -> None:
        assert _ratio(numerator, denominator) == pytest.approx(expected)

    def test_zero_denominator_does_not_raise(self) -> None:
        # Empty hypotheses are a measured outcome, not an error condition.
        assert _ratio(1.0, 0) == 0.0

    def test_negative_baseline_denominator_is_supported(self) -> None:
        # abl_norm passes 1.0 - baseline, which is > 0 under the guard, but
        # the primitive itself must remain sign-agnostic.
        assert _ratio(1.0, -2.0) == pytest.approx(-0.5)


# --------------------------------------------------------------------------
# ConfusionMatrix / confusion_matrix
# --------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_cells_are_reconstructed_from_counts(self) -> None:
        matrix = confusion_matrix(
            hypothesis_extension={"a", "b", "c"},
            target_extension={"b", "c", "d"},
            universe_size=10,
        )
        assert matrix.to_dict() == {"tp": 2, "fp": 1, "fn": 1, "tn": 6}

    def test_cells_sum_to_universe_size(self) -> None:
        matrix = confusion_matrix(
            hypothesis_extension=individuals("p", 7),
            target_extension=individuals("p", 4) | individuals("q", 3),
            universe_size=UNIVERSE,
        )
        assert matrix.tp + matrix.fp + matrix.fn + matrix.tn == UNIVERSE

    def test_perfect_hypothesis_has_no_errors(self) -> None:
        extension = individuals("x", 5)
        matrix = confusion_matrix(
            hypothesis_extension=extension,
            target_extension=extension,
            universe_size=UNIVERSE,
        )
        assert (matrix.tp, matrix.fp, matrix.fn) == (5, 0, 0)
        assert matrix.tn == UNIVERSE - 5

    def test_empty_hypothesis_yields_only_fn_and_tn(self) -> None:
        matrix = confusion_matrix(
            hypothesis_extension=set(),
            target_extension=individuals("t", 8),
            universe_size=UNIVERSE,
        )
        assert (matrix.tp, matrix.fp, matrix.fn, matrix.tn) == (
            0,
            0,
            8,
            UNIVERSE - 8,
        )

    def test_empty_target_yields_only_fp_and_tn(self) -> None:
        matrix = confusion_matrix(
            hypothesis_extension=individuals("h", 3),
            target_extension=set(),
            universe_size=UNIVERSE,
        )
        assert (matrix.tp, matrix.fp, matrix.fn, matrix.tn) == (
            0,
            3,
            0,
            UNIVERSE - 3,
        )

    def test_duplicate_iris_are_deduplicated(self) -> None:
        # Extensions are sets; a repeated IRI must not inflate |P|.
        matrix = confusion_matrix(
            hypothesis_extension=["a", "a", "b"],
            target_extension=["b", "b"],
            universe_size=10,
        )
        assert matrix.to_dict() == {"tp": 1, "fp": 1, "fn": 0, "tn": 8}

    def test_disjoint_extensions_have_zero_tp(self) -> None:
        matrix = confusion_matrix(
            hypothesis_extension=individuals("h", 4),
            target_extension=individuals("t", 6),
            universe_size=UNIVERSE,
        )
        assert matrix.tp == 0
        assert (matrix.fp, matrix.fn) == (4, 6)


class TestConfusionMatrixConsistency:
    """`consistent` is the caller's gate; the matrix is never clamped."""

    def test_well_formed_matrix_is_consistent(self) -> None:
        assert ConfusionMatrix(tp=1, fp=2, fn=3, tn=4).consistent

    def test_all_zero_matrix_is_consistent(self) -> None:
        assert ConfusionMatrix(tp=0, fp=0, fn=0, tn=0).consistent

    @pytest.mark.parametrize(
        "cells",
        [
            {"tp": -1, "fp": 0, "fn": 0, "tn": 0},
            {"tp": 0, "fp": -1, "fn": 0, "tn": 0},
            {"tp": 0, "fp": 0, "fn": -1, "tn": 0},
            {"tp": 0, "fp": 0, "fn": 0, "tn": -1},
        ],
    )
    def test_any_negative_cell_is_inconsistent(self, cells: dict) -> None:
        assert not ConfusionMatrix(**cells).consistent

    def test_undersized_universe_produces_negative_tn_unclamped(self) -> None:
        # A universe smaller than the extensions is a detectable error and
        # must surface as a negative cell rather than a plausible zero.
        matrix = confusion_matrix(
            hypothesis_extension=individuals("h", 10),
            target_extension=individuals("t", 10),
            universe_size=5,
        )
        assert matrix.tn == 5 - 0 - 10 - 10
        assert matrix.tn < 0
        assert not matrix.consistent

    def test_zero_universe_with_nonempty_extension_is_inconsistent(self) -> None:
        matrix = confusion_matrix(
            hypothesis_extension={"a"},
            target_extension={"a"},
            universe_size=0,
        )
        assert not matrix.consistent


class TestConfusionMatrixImmutability:
    def test_is_frozen(self) -> None:
        matrix = ConfusionMatrix(tp=1, fp=1, fn=1, tn=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            matrix.tp = 5  # type: ignore[misc]

    def test_to_dict_key_order_is_stable(self) -> None:
        keys = list(ConfusionMatrix(tp=1, fp=2, fn=3, tn=4).to_dict())
        assert keys == ["tp", "fp", "fn", "tn"]


# --------------------------------------------------------------------------
# score_extensions — core rates
# --------------------------------------------------------------------------


class TestScoreExtensionsCoreRates:
    def test_perfect_hypothesis_scores_one_everywhere(self) -> None:
        extension = individuals("x", 20)
        metrics = score(extension, extension)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.semantic_equivalence is True
        assert metrics.empty_hypothesis is False

    def test_disjoint_hypothesis_scores_zero_rates(self) -> None:
        metrics = score(individuals("h", 10), individuals("t", 10))
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.semantic_equivalence is False

    def test_precision_and_recall_use_correct_denominators(self) -> None:
        # |P| = 4, |T| = 2, |P n T| = 2
        metrics = score({"a", "b", "c", "d"}, {"a", "b"})
        assert metrics.precision == pytest.approx(0.5)
        assert metrics.recall == pytest.approx(1.0)

    def test_f1_is_the_harmonic_mean_of_precision_and_recall(self) -> None:
        metrics = score({"a", "b", "c", "d"}, {"a", "b"})
        expected = (
            2
            * metrics.precision
            * metrics.recall
            / (metrics.precision + metrics.recall)
        )
        assert metrics.f1 == pytest.approx(expected)

    def test_accuracy_counts_true_negatives(self) -> None:
        # One FP and one FN out of 100 individuals.
        metrics = score({"a", "b", "x"}, {"a", "b", "y"})
        assert metrics.accuracy == pytest.approx(98 / 100)

    def test_accuracy_is_tn_dominated_for_small_extension_ratio(self) -> None:
        # A trivially empty hypothesis still scores high accuracy when the
        # target extension is tiny -- the documented reason ABL is primary.
        metrics = score(set(), individuals("t", 1), universe_size=10_000)
        assert metrics.f1 == 0.0
        assert metrics.accuracy > 0.999

    def test_intersection_and_union_are_reported(self) -> None:
        metrics = score({"a", "b", "c"}, {"b", "c", "d"})
        assert metrics.intersection == 2
        assert metrics.union == 4

    def test_extension_sizes_are_deduplicated_counts(self) -> None:
        metrics = score(["a", "a", "b"], ["b", "b", "c"])
        assert metrics.hypothesis_extension_size == 2
        assert metrics.target_extension_size == 2

    def test_universe_size_is_passed_through(self) -> None:
        assert score({"a"}, {"a"}, universe_size=42).universe_size == 42


class TestScoreExtensionsDegenerateCases:
    def test_empty_hypothesis_is_flagged_and_scores_zero(self) -> None:
        metrics = score(set(), individuals("t", 5))
        assert metrics.empty_hypothesis is True
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.intersection == 0
        assert metrics.hypothesis_extension_size == 0

    def test_empty_target_does_not_raise(self) -> None:
        metrics = score(individuals("h", 3), set())
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.target_extension_size == 0

    def test_both_empty_is_semantically_equivalent(self) -> None:
        # owl:Nothing-like target: P == T == {} is genuine equivalence, and
        # every rate collapses to 0.0 without raising.
        metrics = score(set(), set())
        assert metrics.semantic_equivalence is True
        assert metrics.empty_hypothesis is True
        assert metrics.f1 == 0.0
        assert metrics.accuracy == 1.0

    def test_zero_universe_does_not_raise(self) -> None:
        metrics = score(set(), set(), universe_size=0)
        assert metrics.accuracy == 0.0

    def test_semantic_equivalence_is_exact_set_equality(self) -> None:
        target = individuals("t", 5)
        near_miss = set(target)
        near_miss.pop()
        assert score(target, target).semantic_equivalence is True
        assert score(near_miss, target).semantic_equivalence is False

    def test_semantic_equivalence_ignores_input_ordering(self) -> None:
        assert score(["b", "a"], ["a", "b"]).semantic_equivalence is True


class TestScoreExtensionsAcceptsAnyCollection:
    """Extensions arrive as frozensets from the oracle, lists from JSON."""

    @pytest.mark.parametrize("factory", [set, frozenset, list, tuple])
    def test_collection_type_does_not_change_result(self, factory) -> None:
        predicted = factory(["a", "b", "c"])
        target = factory(["b", "c", "d"])
        metrics = score(predicted, target)
        assert metrics.intersection == 2
        assert metrics.f1 == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# ABL and ABL_norm
# --------------------------------------------------------------------------


class TestAtomicBaselineLift:
    def test_abl_is_f1_minus_baseline(self) -> None:
        metrics = score({"a", "b", "c", "d"}, {"a", "b"}, baseline=0.25)
        assert metrics.f1 == pytest.approx(2 / 3)
        assert metrics.abl == pytest.approx(2 / 3 - 0.25)

    def test_abl_is_negative_when_baseline_beats_hypothesis(self) -> None:
        # A hypothesis worse than the best single atomic class must show a
        # negative lift, not a floored zero.
        metrics = score(individuals("h", 5), individuals("t", 5), baseline=0.8)
        assert metrics.f1 == 0.0
        assert metrics.abl == pytest.approx(-0.8)

    def test_abl_is_none_when_baseline_is_missing(self) -> None:
        # Missing annotation triggers the documented raw-F1 substitution;
        # it must never be silently coerced to 0.0.
        metrics = score({"a"}, {"a"}, baseline=None)
        assert metrics.abl is None
        assert metrics.abl_norm is None
        assert metrics.atomic_baseline_f1 is None

    def test_baseline_is_echoed_for_the_analysis_layer(self) -> None:
        assert score({"a"}, {"a"}, baseline=0.5).atomic_baseline_f1 == 0.5

    def test_zero_baseline_leaves_abl_equal_to_f1(self) -> None:
        metrics = score({"a", "b"}, {"a", "b"}, baseline=0.0)
        assert metrics.abl == pytest.approx(metrics.f1)
        assert metrics.abl_norm == pytest.approx(metrics.f1)


class TestAtomicBaselineLiftNormalised:
    def test_abl_norm_is_headroom_normalised(self) -> None:
        metrics = score({"a", "b", "c", "d"}, {"a", "b"}, baseline=0.5)
        assert metrics.abl_norm == pytest.approx((2 / 3 - 0.5) / 0.5)

    def test_abl_norm_is_one_for_a_perfect_hypothesis(self) -> None:
        extension = individuals("x", 10)
        metrics = score(extension, extension, baseline=0.4)
        assert metrics.f1 == 1.0
        assert metrics.abl_norm == pytest.approx(1.0)

    def test_abl_norm_is_computed_at_the_guard_boundary(self) -> None:
        # The guard is inclusive: b == 0.95 is still numerically usable.
        metrics = score({"a"}, {"a"}, baseline=ABL_NORM_GUARD)
        assert metrics.abl_norm is not None
        assert metrics.abl_norm == pytest.approx(1.0)

    def test_abl_norm_is_suppressed_above_the_guard(self) -> None:
        metrics = score({"a"}, {"a"}, baseline=0.96)
        assert metrics.abl_norm is None
        assert metrics.abl is not None  # raw lift stays available

    def test_abl_norm_is_suppressed_at_unit_baseline(self) -> None:
        # Undefined at b == 1; must not raise a ZeroDivisionError.
        metrics = score({"a"}, {"a"}, baseline=1.0)
        assert metrics.abl_norm is None
        assert metrics.abl == pytest.approx(0.0)

    def test_guard_constant_matches_documented_threshold(self) -> None:
        assert ABL_NORM_GUARD == 0.95

    @pytest.mark.parametrize("baseline", [0.0, 0.3, 0.5, 0.9, 0.95])
    def test_abl_norm_is_bounded_above_by_one(self, baseline: float) -> None:
        extension = individuals("x", 7)
        metrics = score(extension, extension, baseline=baseline)
        assert metrics.abl_norm is not None
        assert metrics.abl_norm <= 1.0 + 1e-12


# --------------------------------------------------------------------------
# properties relied upon by the analysis layer
# --------------------------------------------------------------------------


class TestEstimandProperties:
    """Properties the paired design depends on."""

    def test_atomic_baseline_cancels_exactly_in_the_paired_difference(
        self,
    ) -> None:
        # d = m_dice - m_random. The baseline is a property of the problem
        # and KB alone, so it must cancel bit-exactly -- otherwise every
        # contrast carries a baseline-dependent residual.
        target = individuals("t", 37)
        treatment = individuals("t", 30) | individuals("f", 4)
        control = individuals("t", 11) | individuals("f", 19)

        for baseline in (0.0, 0.1234567890123, 0.5, 0.9499999999):
            dice = score(treatment, target, baseline=baseline)
            random = score(control, target, baseline=baseline)
            assert dice.abl is not None and random.abl is not None
            assert dice.abl - random.abl == pytest.approx(dice.f1 - random.f1, abs=1e-15)

    def test_abl_norm_difference_is_a_scaled_f1_difference(self) -> None:
        target = individuals("t", 20)
        treatment = individuals("t", 18)
        control = individuals("t", 6)
        baseline = 0.6

        dice = score(treatment, target, baseline=baseline)
        random = score(control, target, baseline=baseline)
        assert dice.abl_norm is not None and random.abl_norm is not None
        assert dice.abl_norm - random.abl_norm == pytest.approx(
            (dice.f1 - random.f1) / (1.0 - baseline)
        )

    def test_metrics_are_independent_of_the_atomic_baseline(self) -> None:
        # Only abl/abl_norm/atomic_baseline_f1 may vary with the baseline.
        target = individuals("t", 12)
        predicted = individuals("t", 9) | individuals("f", 2)
        varying = {"atomic_baseline_f1", "abl", "abl_norm"}

        low = score(predicted, target, baseline=0.1).to_dict()
        high = score(predicted, target, baseline=0.9).to_dict()
        for key in low:
            if key not in varying:
                assert low[key] == high[key], key

    def test_jaccard_is_a_monotone_reparameterisation_of_f1(self) -> None:
        # Justifies not reporting Jaccard: J = F1 / (2 - F1) carries no
        # information beyond F1.
        target = individuals("t", 25)
        for kept, extra in [(25, 0), (20, 5), (13, 12), (4, 30), (0, 9)]:
            predicted = set(list(target)[:kept]) | individuals("f", extra)
            metrics = score(predicted, target)
            jaccard = (
                metrics.intersection / metrics.union if metrics.union else 0.0
            )
            assert jaccard == pytest.approx(metrics.f1 / (2 - metrics.f1))

    def test_semantic_equivalence_implies_unit_f1_for_nonempty_targets(
        self,
    ) -> None:
        extension = individuals("x", 9)
        metrics = score(extension, extension)
        assert metrics.semantic_equivalence
        assert metrics.f1 == 1.0

    @pytest.mark.parametrize(
        ("kept", "extra"),
        [(30, 0), (25, 5), (15, 15), (1, 40), (0, 3), (30, 70)],
    )
    def test_all_rates_lie_in_the_unit_interval(
        self, kept: int, extra: int
    ) -> None:
        target = individuals("t", 30)
        predicted = set(list(target)[:kept]) | individuals("f", extra)
        metrics = score(predicted, target)
        for value in (
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.accuracy,
        ):
            assert 0.0 <= value <= 1.0
            assert not math.isnan(value)

    def test_f1_is_bounded_by_precision_and_recall(self) -> None:
        target = individuals("t", 18)
        predicted = individuals("t", 6) | individuals("f", 30)
        metrics = score(predicted, target)
        lower = min(metrics.precision, metrics.recall)
        upper = max(metrics.precision, metrics.recall)
        assert lower <= metrics.f1 <= upper


class TestMechanismLayerFields:
    """Fields the interpretive 'breadth hypothesis' layer reads."""

    def test_broad_hypothesis_shows_recall_gain_with_precision_loss(
        self,
    ) -> None:
        target = individuals("t", 10)
        narrow = individuals("t", 5)
        broad = set(target) | individuals("f", 40)

        narrow_metrics = score(narrow, target)
        broad_metrics = score(broad, target)

        assert broad_metrics.recall > narrow_metrics.recall
        assert broad_metrics.precision < narrow_metrics.precision
        assert (
            broad_metrics.hypothesis_extension_size
            > narrow_metrics.hypothesis_extension_size
        )

    def test_hypothesis_extension_size_is_a_raw_count(self) -> None:
        # Documented as raw-scale and readable within a KB only.
        metrics = score(individuals("h", 17), individuals("t", 3))
        assert metrics.hypothesis_extension_size == 17
        assert isinstance(metrics.hypothesis_extension_size, int)


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


class TestExtensionMetricsSerialisation:
    def test_to_dict_exposes_the_full_schema(self) -> None:
        expected = {
            "precision",
            "recall",
            "f1",
            "accuracy",
            "semantic_equivalence",
            "intersection",
            "union",
            "hypothesis_extension_size",
            "target_extension_size",
            "universe_size",
            "atomic_baseline_f1",
            "abl",
            "abl_norm",
            "empty_hypothesis",
        }
        assert set(score({"a"}, {"a"}, baseline=0.5).to_dict()) == expected

    def test_round_trip_preserves_every_field(self) -> None:
        original = score(
            individuals("t", 6) | individuals("f", 2),
            individuals("t", 9),
            baseline=0.42,
        )
        assert ExtensionMetrics.from_dict(original.to_dict()) == original

    def test_round_trip_preserves_absent_baseline_as_none(self) -> None:
        original = score({"a"}, {"a", "b"}, baseline=None)
        restored = ExtensionMetrics.from_dict(original.to_dict())
        assert restored.abl is None
        assert restored.abl_norm is None
        assert restored.atomic_baseline_f1 is None
        assert restored == original

    def test_round_trip_preserves_suppressed_abl_norm(self) -> None:
        original = score({"a"}, {"a"}, baseline=0.99)
        restored = ExtensionMetrics.from_dict(original.to_dict())
        assert restored.abl is not None
        assert restored.abl_norm is None
        assert restored == original

    def test_round_trip_survives_json(self) -> None:
        import json

        original = score(
            individuals("t", 4) | individuals("f", 3),
            individuals("t", 5),
            baseline=0.31,
        )
        restored = ExtensionMetrics.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )
        assert restored == original

    def test_from_dict_coerces_persisted_types(self) -> None:
        # JSON may widen ints to floats and bools to 0/1.
        payload = {
            "precision": 1,
            "recall": "0.5",
            "f1": 0.6666666666666666,
            "accuracy": 1,
            "semantic_equivalence": 0,
            "intersection": 2.0,
            "union": 4.0,
            "hypothesis_extension_size": 4.0,
            "target_extension_size": 2.0,
            "universe_size": 100.0,
            "atomic_baseline_f1": 0.25,
            "abl": 0.4166666666666666,
            "abl_norm": 0.5555555555555555,
            "empty_hypothesis": 0,
        }
        restored = ExtensionMetrics.from_dict(payload)
        assert restored.precision == 1.0
        assert restored.recall == 0.5
        assert restored.semantic_equivalence is False
        assert restored.empty_hypothesis is False
        assert restored.intersection == 2
        assert isinstance(restored.intersection, int)

    @pytest.mark.parametrize(
        "missing",
        ["precision", "recall", "f1", "accuracy", "empty_hypothesis"],
    )
    def test_from_dict_rejects_missing_required_fields(
        self, missing: str
    ) -> None:
        payload = score({"a"}, {"a"}, baseline=0.5).to_dict()
        del payload[missing]
        with pytest.raises(KeyError):
            ExtensionMetrics.from_dict(payload)

    def test_optional_fields_may_be_absent_entirely(self) -> None:
        # `payload.get` is used for the three optional fields.
        payload = score({"a"}, {"a"}, baseline=None).to_dict()
        for key in ("atomic_baseline_f1", "abl", "abl_norm"):
            del payload[key]
        restored = ExtensionMetrics.from_dict(payload)
        assert restored.abl is None


class TestExtensionMetricsImmutability:
    def test_is_frozen(self) -> None:
        metrics = score({"a"}, {"a"})
        with pytest.raises(Exception):  # FrozenInstanceError
            metrics.f1 = 0.0  # type: ignore[misc]

    def test_equal_scores_compare_equal(self) -> None:
        assert score({"a", "b"}, {"a"}, baseline=0.2) == score(
            {"b", "a"}, {"a"}, baseline=0.2
        )