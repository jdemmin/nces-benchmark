# tests/benchmarking/test_metrics.py
"""Tests for src.benchmarking.metrics.

The suite focuses on the non-obvious behaviour:

* accuracy is measured over the *whole* universe, not the sampled examples;
* ``calculate_metrics`` uses Welford's algorithm with a per-metric
  observation count, so ``None`` values must not shift the mean;
* ``MetricsResult.to_mean_metrics`` maps ``semantic_equivalence`` (a bool)
  onto ``semantic_equivalence_rate`` (a float);
* complexity bucketing boundaries are half-open on the right.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from src.benchmarking.metrics import (
    COMPLEXITY_AXES,
    _group_by_complexity,
    _mean,
    _meanSemanticEquivalence,
    _ratio_bucket,
    calculate_extension_metrics,
    calculate_metrics,
    compute_lift,
    get_complexity_summary,
    mean_embeddings_results,
)
from src.config import EmbeddingSettings
from src.data.results import (
    EmbeddingResult,
    LearningProblemResult,
    MetricsResult,
    NCESStats,
)

# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------


@pytest.fixture
def universe() -> list[str]:
    return [f"ex:i{n}" for n in range(10)]


def make_metrics(**overrides) -> MetricsResult:
    """A MetricsResult with distinguishable, non-default values."""
    base = {
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.7,
        "f1_score": 0.75,
        "jaccard": 0.6,
        "semantic_equivalence": False,
        "intersection": 3,
        "union": 5,
        "lift": 0.25,
    }
    base.update(overrides)
    return MetricsResult(**base)  # type: ignore[arg-type]


def make_lp_result(monkeypatch_free_lp, metrics: MetricsResult | None):
    """Build a LearningProblemResult without touching ontolearn."""
    return LearningProblemResult(
        learning_problem=monkeypatch_free_lp,
        metrics=metrics,
    )

# ---------------------------------------------------------------------------
# lightweight stand-ins for the LearningProblem/Complexity graph
# ---------------------------------------------------------------------------


class FakeHardness:
    def __init__(self, extension_ratio=None, atomic_baseline_f1=None):
        self.extension_ratio = extension_ratio
        self.atomic_baseline_f1 = atomic_baseline_f1


class FakeComplexity:
    def __init__(
        self,
        dl_length=3,
        depth=1,
        expressivity="ALC",
        constructors=("AND",),
        num_atomic_classes=2,
        num_roles=0,
        hardness=None,
    ):
        self.dl_length = dl_length
        self.depth = depth
        self.expressivity = expressivity
        self.constructors = constructors
        self.num_atomic_classes = num_atomic_classes
        self.num_roles = num_roles
        self.hardness = hardness or FakeHardness()


class FakeTruncatedLP:
    def __init__(self, name, complexity):
        self.name = name
        self.complexity = complexity

    def to_dict(self):
        return {"name": self.name}


class FakeLP:
    """Quacks like LearningProblem for the two attributes metrics.py needs."""

    def __init__(self, name="lp", complexity=None):
        self.name = name
        self.complexity = complexity or FakeComplexity()

    def to_truncated(self):
        return FakeTruncatedLP(self.name, self.complexity)


def lp_result(metrics: MetricsResult | None = None, **complexity_kwargs):
    return LearningProblemResult(
        learning_problem=FakeLP(complexity=FakeComplexity(**complexity_kwargs)),
        metrics=metrics if metrics is not None else make_metrics(),
    )

class TestCalculateExtensionMetrics:
    def test_perfect_match_is_semantically_equivalent(self, universe):
        target = universe[:4]
        m = calculate_extension_metrics(target, target, universe)

        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.jaccard == 1.0
        assert m.semantic_equivalence is True
        assert m.intersection == 4
        assert m.union == 4

    def test_accuracy_is_measured_over_the_full_universe(self, universe):
        """A hypothesis that nails 2/2 positives but is silent elsewhere
        must not be rewarded with a perfect score, and a hypothesis that
        over-generalises must be punished on the true negatives."""
        target = universe[:2]
        over_general = universe  # predicts everything

        m = calculate_extension_metrics(over_general, target, universe)

        assert m.recall == 1.0
        assert m.precision == pytest.approx(0.2)
        # 2 TP + 0 TN out of 10 individuals
        assert m.accuracy == pytest.approx(0.2)

    def test_partial_overlap_matches_hand_computed_values(self, universe):
        target = {"ex:i0", "ex:i1", "ex:i2", "ex:i3"}
        predicted = {"ex:i2", "ex:i3", "ex:i4"}

        m = calculate_extension_metrics(predicted, target, universe)

        # TP=2, FP=1, FN=2, TN=10-2-1-2=5
        assert m.intersection == 2
        assert m.union == 5
        assert m.precision == pytest.approx(2 / 3,)
        assert m.recall == pytest.approx(0.5)
        assert m.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
        assert m.jaccard == pytest.approx(2 / 5)
        assert m.accuracy == pytest.approx(7 / 10)
        assert m.semantic_equivalence is False

    def test_individuals_outside_all_individuals_extend_the_universe(self):
        """The universe is the union of the three inputs, so an unknown
        predicted individual must still count as a false positive without
        making the denominator inconsistent."""
        m = calculate_extension_metrics(
            predicted={"a", "ghost"},
            target={"a", "b"},
            all_individuals={"a", "b"},
        )

        # universe = {a, b, ghost} -> TP=1, FP=1, FN=1, TN=0
        assert m.accuracy == pytest.approx(1 / 3)
        assert m.union == 3

    def test_empty_prediction_yields_zeroes_not_division_error(self, universe):
        m = calculate_extension_metrics([], universe[:3], universe)

        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.jaccard == 0.0
        assert m.accuracy == pytest.approx(0.7)  # 7 true negatives
        assert m.semantic_equivalence is False

    def test_both_empty_is_equivalent_but_scores_zero_on_f1(self, universe):
        """Degenerate but important: the empty concept equals the empty
        target, yet precision/recall are undefined and clamp to 0."""
        m = calculate_extension_metrics([], [], universe)

        assert m.semantic_equivalence is True
        assert m.accuracy == 1.0
        assert m.f1 == 0.0
        assert m.union == 0

    def test_duplicates_in_input_are_ignored(self, universe):
        dupes = ["ex:i0", "ex:i0", "ex:i1"]
        deduped = ["ex:i0", "ex:i1"]

        assert calculate_extension_metrics(
            dupes, universe[:2], universe
        ) == calculate_extension_metrics(deduped, universe[:2], universe)

    def test_empty_universe_does_not_raise(self):
        m = calculate_extension_metrics([], [], [])
        assert m.accuracy == 0.0  # 0/0 guarded by _ratio

    def test_to_dict_round_trips_all_fields(self, universe):
        m = calculate_extension_metrics(universe[:2], universe[:3], universe)
        d = m.to_dict()

        assert set(d) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "jaccard",
            "semantic_equivalence",
            "intersection",
            "union",
        }
        assert d["intersection"] == 2


class TestComputeLift:
    def test_none_when_baseline_missing(self):
        c = FakeComplexity(hardness=FakeHardness(atomic_baseline_f1=None))
        assert compute_lift(0.9, c) is None

    def test_positive_lift_over_baseline(self):
        c = FakeComplexity(hardness=FakeHardness(atomic_baseline_f1=0.4))
        assert compute_lift(0.9, c) == pytest.approx(0.5)

    def test_negative_lift_signals_worse_than_atomic_concept(self):
        c = FakeComplexity(hardness=FakeHardness(atomic_baseline_f1=0.8))
        assert compute_lift(0.3, c) == pytest.approx(-0.5)

    def test_zero_baseline_is_not_treated_as_missing(self):
        """0.0 is falsy; the implementation must check for None explicitly."""
        c = FakeComplexity(hardness=FakeHardness(atomic_baseline_f1=0.0))
        assert compute_lift(0.6, c) == pytest.approx(0.6)

class TestRatioBucket:
    @pytest.mark.parametrize(
        "ratio,expected",
        [
            (None, "unknown"),
            (0.0, "rare"),
            (0.049, "rare"),
            (0.05, "uncommon"),   # boundary belongs to the upper bucket
            (0.249, "uncommon"),
            (0.25, "balanced"),
            (0.749, "balanced"),
            (0.75, "dominant"),
            (1.0, "dominant"),
        ],
    )
    def test_boundaries_are_half_open(self, ratio, expected):
        assert _ratio_bucket(ratio) == expected


class TestCalculateMetrics:
    def test_single_record_has_zero_variance(self):
        result = calculate_metrics([make_metrics(accuracy=0.5)])

        assert result.lp_count == 1
        assert result.accuracy.mean == pytest.approx(0.5)
        assert result.accuracy.variance == 0.0
        assert result.accuracy.std_dev == 0.0

    def test_mean_and_sample_variance_match_closed_form(self):
        values = [0.2, 0.4, 0.9]
        records = [make_metrics(f1_score=v) for v in values]

        result = calculate_metrics(records)

        mean = round(sum(values) / len(values), 4)
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)

        assert result.f1_score.mean == pytest.approx(mean)
        assert result.f1_score.variance == pytest.approx(var)
        assert result.f1_score.std_dev == pytest.approx(round(math.sqrt(var), 4))
        assert result.lp_count == 3

    def test_variance_is_sample_not_population(self):
        """Two observations 0.0 and 1.0: sample variance 0.5, population 0.25."""
        result = calculate_metrics(
            [make_metrics(jaccard=0.0), make_metrics(jaccard=1.0)]
        )
        assert result.jaccard.variance == pytest.approx(0.5)

    def test_identifiers_are_set_per_metric(self):
        result = calculate_metrics([make_metrics()])

        assert result.accuracy.identifier == "accuracy"
        assert result.f1_score.identifier == "f1_score"
        assert result.semantic_equivalence_rate.identifier == (
            "semantic_equivalence_rate"
        )

    def test_bools_are_averaged_into_an_equivalence_rate(self):
        records = [
            make_metrics(semantic_equivalence=True),
            make_metrics(semantic_equivalence=True),
            make_metrics(semantic_equivalence=False),
            make_metrics(semantic_equivalence=False),
        ]

        result = calculate_metrics(records)

        assert result.semantic_equivalence_rate.mean == pytest.approx(0.5)

    def test_empty_input_returns_zeroed_aggregate(self):
        result = calculate_metrics([])

        assert result.lp_count == 0
        assert result.accuracy.mean == 0.0
        assert result.accuracy.variance == 0.0

    def test_welford_is_numerically_stable_for_large_offsets(self):
        """A naive sum-of-squares accumulator loses all precision here."""
        base = 1e8
        values = [base + 1, base + 2, base + 3, base + 4]
        result = calculate_metrics([make_metrics(union=v) for v in values])

        assert result.union.mean == pytest.approx(round(base + 2.5, 4))
        assert result.union.variance == pytest.approx(round(5 / 3, 4), rel=1e-6)

    def test_accepts_mean_metrics_input_for_two_stage_aggregation(self):
        stage_one = [
            calculate_metrics([make_metrics(recall=0.2)]),
            calculate_metrics([make_metrics(recall=0.8)]),
        ]

        stage_two = calculate_metrics(stage_one)

        assert stage_two.lp_count == 2
        assert stage_two.recall.mean == pytest.approx(0.5)

    def test_none_lift_is_skipped_without_biasing_the_mean(self):
        """Per-metric counters mean a missing lift must not be read as 0.0."""
        records = [
            calculate_metrics([make_metrics(lift=0.6)]),
            calculate_metrics([make_metrics(lift=0.4)]),
        ]
        records.append(replace(records[0], lift=None))

        result = calculate_metrics(records)

        assert result.lp_count == 3  # the record still counts as a problem
        assert result.lift.mean == pytest.approx(0.5)  # ...but not toward lift
        assert result.accuracy.mean == pytest.approx(0.9)

    def test_none_entries_in_the_sequence_are_dropped(self):
        result = calculate_metrics([make_metrics(accuracy=0.4), None])  # type: ignore[list-item]

        assert result.lp_count == 1
        assert result.accuracy.mean == pytest.approx(0.4)

    def test_order_of_records_does_not_change_the_aggregate(self):
        values = [0.1, 0.35, 0.7, 0.95]
        forward = calculate_metrics([make_metrics(precision=v) for v in values])
        backward = calculate_metrics(
            [make_metrics(precision=v) for v in reversed(values)]
        )

        assert forward.precision.mean == pytest.approx(backward.precision.mean)
        assert forward.precision.variance == pytest.approx(
            backward.precision.variance
        )

def make_embedding_result(metrics_list, split_name="test"):
    return EmbeddingResult(
        split_name=split_name,
        learning_problem_results=[],
        embedding_settings=EmbeddingSettings(),
        nces_stats=NCESStats("GRU", 1.0, False),
        number_of_problems=len(metrics_list),
        number_of_successful_problems=len(metrics_list),
        mean_metrics=calculate_metrics(metrics_list) if metrics_list else None,
    )


class TestMeanEmbeddingsResults:
    def test_averages_across_reports(self):
        reports = [
            make_embedding_result([make_metrics(f1_score=0.4)]),
            make_embedding_result([make_metrics(f1_score=0.8)]),
        ]

        result = mean_embeddings_results(reports)

        assert result.lp_count == 2
        assert result.f1_score.mean == pytest.approx(0.6)

    def test_reports_without_metrics_are_excluded(self):
        reports = [
            make_embedding_result([make_metrics(f1_score=0.4)]),
            make_embedding_result([]),  # mean_metrics is None
        ]

        result = mean_embeddings_results(reports)

        assert result.lp_count == 1
        assert result.f1_score.mean == pytest.approx(0.4)

    def test_all_reports_empty_returns_zeroed_aggregate(self):
        result = mean_embeddings_results([make_embedding_result([])])
        assert result.lp_count == 0


class TestGroupByComplexity:
    def test_groups_by_the_requested_axis(self):
        records = [
            lp_result(dl_length=1),
            lp_result(dl_length=1),
            lp_result(dl_length=4),
        ]

        grouped = _group_by_complexity(records, "dl_length")

        assert sorted(grouped) == [1, 4]
        assert len(grouped[1]) == 2
        assert len(grouped[4]) == 1

    def test_constructors_axis_uses_the_cardinality(self):
        records = [
            lp_result(constructors=("AND", "SOME")),
            lp_result(constructors=("AND",)),
        ]

        grouped = _group_by_complexity(records, "constructors")

        assert sorted(grouped) == [1, 2]

    def test_extension_ratio_axis_uses_the_bucket_label(self):
        records = [
            lp_result(hardness=FakeHardness(extension_ratio=0.01)),
            lp_result(hardness=FakeHardness(extension_ratio=0.9)),
            lp_result(hardness=FakeHardness(extension_ratio=None)),
        ]

        grouped = _group_by_complexity(records, "extension_ratio")

        assert set(grouped) == {"rare", "dominant", "unknown"}

    def test_unknown_axis_raises(self):
        with pytest.raises(ValueError, match="Unknown complexity axis"):
            _group_by_complexity([lp_result()], "not_an_axis")

    def test_no_records_yields_no_groups(self):
        assert _group_by_complexity([], "depth") == {}


class TestComplexitySummary:
    def test_every_axis_is_present(self):
        summary = get_complexity_summary([lp_result(), lp_result()])
        assert set(summary) == set(COMPLEXITY_AXES)

    def test_buckets_aggregate_only_their_own_members(self):
        records = [
            lp_result(make_metrics(f1_score=0.2), dl_length=1),
            lp_result(make_metrics(f1_score=0.4), dl_length=1),
            lp_result(make_metrics(f1_score=1.0), dl_length=7),
        ]

        by_length = get_complexity_summary(records)["dl_length"]

        assert by_length[1].lp_count == 2
        assert by_length[1].f1_score.mean == pytest.approx(0.3)
        assert by_length[7].lp_count == 1
        assert by_length[7].f1_score.mean == pytest.approx(1.0)

    def test_failed_problems_are_excluded_from_their_bucket(self):
        """A problem with metrics=None still forms a bucket key, but must
        not contribute an observation."""
        lp_without_metrics = lp_result(None, dl_length=2)
        lp_without_metrics.metrics = None
        records = [
            lp_result(make_metrics(f1_score=0.5), dl_length=2),
            lp_without_metrics,
        ]

        by_length = get_complexity_summary(records)["dl_length"]

        assert by_length[2].lp_count == 1
        assert by_length[2].f1_score.mean == pytest.approx(0.5)

    def test_bucket_with_only_failures_is_zeroed_not_missing(self):
        lp_without_metrics = lp_result(None, dl_length=9)
        lp_without_metrics.metrics = None
        by_length = get_complexity_summary([lp_without_metrics])[
            "dl_length"
        ]

        assert 9 in by_length
        assert by_length[9].lp_count == 0

    def test_empty_records_produce_empty_axis_maps(self):
        assert get_complexity_summary([]) == {}


class TestLegacyMeanHelpers:
    def test_mean_ignores_records_without_the_key(self):
        records = [
            lp_result(make_metrics(f1_score=0.4)),
            lp_result(make_metrics(f1_score=0.6)),
        ]
        assert _mean(records, "f1_score") == pytest.approx(0.5)

    def test_mean_of_unknown_key_is_zero(self):
        assert _mean([lp_result()], "does_not_exist") == 0.0

    def test_mean_of_empty_list_is_zero(self):
        assert _mean([], "f1_score") == 0.0

    def test_semantic_equivalence_rate(self):
        records = [
            lp_result(make_metrics(semantic_equivalence=True)),
            lp_result(make_metrics(semantic_equivalence=True)),
            lp_result(make_metrics(semantic_equivalence=False)),
        ]
        # Note: the helper divides by the number of *truthy-or-present*
        # entries, treating False as "empty".
        assert _meanSemanticEquivalence(records) == pytest.approx(1.0)

    def test_semantic_equivalence_of_empty_list_is_zero(self):
        assert _meanSemanticEquivalence([]) == 0.0
        assert _meanSemanticEquivalence(None) == 0.0  # type: ignore[arg-type]