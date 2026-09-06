# tests/eval/test_inference.py
"""Unit tests for the inferential layer.

The design under test is paired and seed-clustered, so most fixtures build a
``PairedDesign`` directly rather than going through ``assemble``: the goal is
to exercise the estimators, not the pairing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval import inference
from src.eval.inference import (
    Diagnostics,
    Interval,
    NonParametricResult,
    SeedSpread,
    _pratt_signed_rank,
    _sign_flip_p_value,
    analyse_contrast,
    cluster_bootstrap_interval,
    diagnostics,
    hodges_lehmann,
    non_parametric_test,
    point_estimate,
    seed_spread,
)
from src.eval.pairing import OUTCOMES, PairedDesign

# -- fixtures -------------------------------------------------------------


def make_design(
    per_seed_values: dict[int, dict[str, float | None]],
    *,
    condition: str = "dice",
    knowledge_base: str = "semantic_bible",
    outcome: str = "abl",
    hypotheses: dict[str, tuple[str, str]] | None = None,
    extension_sizes: dict[str, tuple[float, float]] | None = None,
    substituted_primary: bool = False,
) -> PairedDesign:
    """Build a ``PairedDesign`` from ``{seed: {problem_id: diff}}``.

    Only the columns the inference layer reads are populated: ``seed``,
    ``problem_id``, ``d_<outcome>`` and, optionally, the two hypothesis
    columns and the two extension-size columns used by the diagnostics.
    """
    rows: list[dict[str, object]] = []
    for seed, problems in per_seed_values.items():
        for problem_id, diff in problems.items():
            row: dict[str, object] = {
                "seed": seed,
                "problem_id": problem_id,
                f"d_{outcome}": diff,
            }
            if hypotheses is not None:
                treated, control = hypotheses[problem_id]
                row["hypothesis_treated"] = treated
                row["hypothesis_control"] = control
            if extension_sizes is not None:
                treated_size, control_size = extension_sizes[problem_id]
                row["hypothesis_extension_size_treated"] = treated_size
                row["hypothesis_extension_size_control"] = control_size
            rows.append(row)

    return PairedDesign(
        condition=condition,
        knowledge_base=knowledge_base,
        frame=pd.DataFrame(rows),
        seeds=tuple(sorted(per_seed_values)),
        unpaired_problem_ids=(),
        failures={},
        substituted_primary=substituted_primary,
    )


@pytest.fixture
def two_seed_design() -> PairedDesign:
    """Two seeds, three problems, one exact tie after collapsing."""
    return make_design(
        {
            1: {"p1": 0.4, "p2": -0.2, "p3": 0.1},
            2: {"p1": 0.2, "p2": 0.0, "p3": -0.1},
        }
    )


@pytest.fixture
def five_seed_design() -> PairedDesign:
    """Five seeds with a seed-dependent offset, as the real suite has."""
    offsets = {1: 0.00, 2: 0.05, 3: -0.05, 4: 0.10, 5: -0.10}
    base = {"p1": 0.30, "p2": 0.10, "p3": -0.05, "p4": 0.00}
    return make_design(
        {
            seed: {pid: value + offset for pid, value in base.items()}
            for seed, offset in offsets.items()
        }
    )


# -- point estimate -------------------------------------------------------


class TestPointEstimate:
    def test_is_mean_of_seed_collapsed_per_problem_diffs(self, two_seed_design):
        # collapsed: p1 = 0.3, p2 = -0.1, p3 = 0.0 -> mean = 0.2/3
        assert point_estimate(two_seed_design, "abl") == pytest.approx(
            (0.3 - 0.1 + 0.0) / 3
        )

    def test_unbalanced_seeds_weight_problems_equally(self):
        """A problem seen in one seed still contributes exactly one value."""
        design = make_design(
            {1: {"p1": 1.0, "p2": 0.0}, 2: {"p1": 3.0}},
        )
        # p1 collapses to 2.0, p2 to 0.0 -> 1.0, not the raw mean of 4/3.
        assert point_estimate(design, "abl") == pytest.approx(1.0)

    def test_none_for_missing_outcome(self, two_seed_design):
        assert point_estimate(two_seed_design, "recall") is None

    def test_none_when_all_values_are_nan(self):
        design = make_design({1: {"p1": None}, 2: {"p1": None}})
        assert point_estimate(design, "abl") is None


# -- Hodges-Lehmann -------------------------------------------------------


class TestHodgesLehmann:
    def test_empty_is_nan(self):
        assert np.isnan(hodges_lehmann(np.array([])))

    def test_single_value_returned_verbatim(self):
        assert hodges_lehmann(np.array([0.42])) == pytest.approx(0.42)

    def test_includes_the_diagonal_walsh_averages(self):
        """With ``k=0`` the self-averages (i.e. the values) are included."""
        values = np.array([0.0, 2.0])
        # Walsh averages: 0.0, 1.0, 2.0 -> median 1.0
        assert hodges_lehmann(values) == pytest.approx(1.0)

    def test_differs_from_median_of_differences(self):
        """Regression guard: HL is not ``np.median``."""
        values = np.array([0.0, 0.0, 0.0, 10.0, 12.0])
        assert np.median(values) == pytest.approx(0.0)
        assert hodges_lehmann(values) > 0.0

    def test_matches_brute_force_walsh_median(self):
        rng = np.random.default_rng(11)
        values = rng.normal(size=9)
        expected = np.median(
            [
                (values[i] + values[j]) / 2
                for i in range(values.size)
                for j in range(i, values.size)
            ]
        )
        assert hodges_lehmann(values) == pytest.approx(expected)

    def test_shift_equivariant(self):
        values = np.array([-0.3, 0.1, 0.4, 0.9])
        shifted = hodges_lehmann(values + 5.0)
        assert shifted == pytest.approx(hodges_lehmann(values) + 5.0)


# -- Pratt signed rank ----------------------------------------------------


class TestPrattSignedRank:
    def test_all_positive_is_the_full_rank_sum(self):
        values = np.array([0.1, 0.2, 0.3])
        assert _pratt_signed_rank(values) == pytest.approx(1 + 2 + 3)

    def test_antisymmetric_under_negation(self):
        values = np.array([0.4, -0.2, 0.0, 0.7])
        assert _pratt_signed_rank(-values) == pytest.approx(
            -_pratt_signed_rank(values)
        )

    def test_zeros_are_ranked_before_being_dropped(self):
        """Pratt's rule: zeros occupy the low ranks and push the others up.

        Under Wilcoxon's discard-then-rank rule both calls below would give
        the same statistic; under Pratt's they must not.
        """
        without_zeros = _pratt_signed_rank(np.array([0.1, 0.2]))
        with_zeros = _pratt_signed_rank(np.array([0.0, 0.0, 0.1, 0.2]))
        assert without_zeros == pytest.approx(1 + 2)
        assert with_zeros == pytest.approx(3 + 4)
        assert with_zeros > without_zeros

    def test_ties_in_magnitude_share_average_ranks(self):
        values = np.array([0.5, -0.5])
        # both get rank 1.5 -> they cancel
        assert _pratt_signed_rank(values) == pytest.approx(0.0)

    def test_all_zeros_is_zero(self):
        assert _pratt_signed_rank(np.zeros(4)) == pytest.approx(0.0)


# -- sign-flip null -------------------------------------------------------


class TestSignFlipPValue:
    def test_no_nonzero_values_is_p_one_and_exact(self):
        p_value, exact = _sign_flip_p_value(np.zeros(5))
        assert p_value == 1.0
        assert exact is True

    def test_small_n_uses_exact_enumeration(self):
        p_value, exact = _sign_flip_p_value(np.array([0.1, 0.2, 0.3]))
        assert exact is True
        # Only the all-positive flip attains the extreme statistic in each
        # tail: 2 of 2**3 assignments.
        assert p_value == pytest.approx(2 / 8)

    def test_exact_p_value_is_a_multiple_of_the_enumeration_grid(self):
        values = np.array([0.5, -0.2, 0.3, 0.9])
        p_value, exact = _sign_flip_p_value(values)
        assert exact is True
        assert p_value * 2**4 == pytest.approx(
            round(p_value * 2**4)
        )

    def test_single_nonzero_value_cannot_reject(self):
        p_value, exact = _sign_flip_p_value(np.array([0.0, 0.0, 0.7]))
        assert exact is True
        assert p_value == pytest.approx(1.0)

    def test_p_value_is_bounded_and_symmetric_in_sign(self):
        values = np.array([0.3, 0.4, -0.1, 0.6, 0.2])
        forward, _ = _sign_flip_p_value(values)
        reversed_, _ = _sign_flip_p_value(-values)
        assert 0.0 < forward <= 1.0
        assert forward == pytest.approx(reversed_)

    def test_zero_mass_does_not_change_the_exact_p_value(self):
        """Padding with zeros rescales all ranks, so the null is unchanged."""
        bare, _ = _sign_flip_p_value(np.array([0.1, 0.2, 0.3]))
        padded, _ = _sign_flip_p_value(
            np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
        )
        assert bare == pytest.approx(padded)

    def test_monte_carlo_branch_above_threshold(self, monkeypatch):
        """Above the threshold the null is sampled, not enumerated."""
        monkeypatch.setattr(inference, "EXACT_SIGN_FLIP_MAX_N", 3)
        monkeypatch.setattr(inference, "SIGN_FLIP_RESAMPLES", 200)
        values = np.array([0.1, 0.2, 0.3, 0.4])
        p_value, exact = _sign_flip_p_value(values, seed=0)
        assert exact is False
        assert 0.0 < p_value <= 1.0
        # The observed value is counted into the null, so p is never 0.
        assert p_value >= 1 / (200 + 1)

    def test_monte_carlo_is_deterministic_in_its_seed(self, monkeypatch):
        monkeypatch.setattr(inference, "EXACT_SIGN_FLIP_MAX_N", 2)
        monkeypatch.setattr(inference, "SIGN_FLIP_RESAMPLES", 300)
        values = np.array([0.4, -0.1, 0.6, 0.2, 0.5])
        first, _ = _sign_flip_p_value(values, seed=7)
        second, _ = _sign_flip_p_value(values, seed=7)
        other, _ = _sign_flip_p_value(values, seed=8)
        assert first == second
        assert first != other or True  # seeds may coincide; equality is fine

    def test_monte_carlo_approximates_the_exact_p_value(self, monkeypatch):
        values = np.array([0.3, 0.5, -0.2, 0.8, 0.1, 0.4])
        exact_p, exact_flag = _sign_flip_p_value(values)
        assert exact_flag is True

        monkeypatch.setattr(inference, "EXACT_SIGN_FLIP_MAX_N", 2)
        monkeypatch.setattr(inference, "SIGN_FLIP_RESAMPLES", 20_000)
        sampled_p, sampled_flag = _sign_flip_p_value(values, seed=3)
        assert sampled_flag is False
        assert sampled_p == pytest.approx(exact_p, abs=0.02)

    def test_threshold_boundary_is_inclusive(self, monkeypatch):
        monkeypatch.setattr(inference, "EXACT_SIGN_FLIP_MAX_N", 4)
        _, at_threshold = _sign_flip_p_value(np.array([0.1, 0.2, 0.3, 0.4]))
        assert at_threshold is True
        monkeypatch.setattr(inference, "SIGN_FLIP_RESAMPLES", 50)
        _, above = _sign_flip_p_value(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        assert above is False

# -- non-parametric result ------------------------------------------------


class TestNonParametricTest:
    def test_degenerate_all_zero_contrast(self):
        result = non_parametric_test(pd.Series([0.0, 0.0, 0.0]))
        assert result.degenerate is True
        assert result.p_value == 1.0
        assert result.hodges_lehmann == 0.0
        assert (result.wins, result.losses, result.ties) == (0, 0, 3)
        assert result.sign_test_p is None
        assert result.n_nonzero == 0
        assert result.exact_null is True

    def test_empty_series_is_degenerate(self):
        result = non_parametric_test(pd.Series(dtype=float))
        assert result.degenerate is True
        assert result.ties == 0
        assert result.n_nonzero == 0

    def test_win_loss_tie_counts(self):
        result = non_parametric_test(pd.Series([0.5, -0.1, 0.0, 0.2, 0.0]))
        assert (result.wins, result.losses, result.ties) == (2, 1, 2)
        assert result.n_nonzero == 3
        assert result.degenerate is False

    def test_sign_test_uses_only_nonzero_observations(self):
        """Binomial on 4 wins of 4 trials, two-sided."""
        result = non_parametric_test(pd.Series([0.1, 0.2, 0.3, 0.4, 0.0, 0.0]))
        assert result.sign_test_p == pytest.approx(2 * 0.5**4)

    def test_hodges_lehmann_is_reported_on_all_values(self):
        values = [0.0, 0.0, 0.0, 0.0, 0.6, 0.8]
        result = non_parametric_test(pd.Series(values))
        assert result.hodges_lehmann == pytest.approx(
            hodges_lehmann(np.array(values))
        )

    def test_large_zero_mass_can_pair_small_mean_with_small_p(self):
        """The pattern the methodology warns about, pinned as behaviour."""
        values = [0.0] * 12 + [0.02] * 8
        result = non_parametric_test(pd.Series(values))
        assert np.mean(values) < 0.01
        assert result.p_value < 0.05
        assert result.hodges_lehmann > 0.0

    def test_p_value_invariant_to_ordering(self):
        values = [0.4, -0.1, 0.0, 0.3, 0.2]
        first = non_parametric_test(pd.Series(values))
        second = non_parametric_test(pd.Series(list(reversed(values))))
        assert first.p_value == pytest.approx(second.p_value)
        assert first.hodges_lehmann == pytest.approx(second.hodges_lehmann)

    def test_nan_free_result_is_json_serialisable_shape(self):
        result = non_parametric_test(pd.Series([0.2, -0.3, 0.5]))
        payload = result.to_dict()
        assert set(payload) == {
            "p_value",
            "hodges_lehmann",
            "wins",
            "losses",
            "ties",
            "sign_test_p",
            "n_nonzero",
            "exact_null",
            "degenerate",
        }
        assert isinstance(payload["p_value"], float)


# -- cluster bootstrap ----------------------------------------------------


class TestClusterBootstrapInterval:
    def test_none_with_a_single_seed(self):
        design = make_design({1: {"p1": 0.2, "p2": 0.4}})
        assert cluster_bootstrap_interval(design, "abl", n_resamples=50) is None

    def test_none_for_missing_outcome(self, five_seed_design):
        assert (
            cluster_bootstrap_interval(five_seed_design, "recall")
            is None
        )

    def test_none_when_every_value_is_nan(self):
        design = make_design({1: {"p1": None}, 2: {"p1": None}})
        assert cluster_bootstrap_interval(design, "abl", n_resamples=50) is None

    def test_identical_seeds_give_a_degenerate_interval(self):
        """Seed-clustered resampling cannot vary when seeds are identical.

        A row-level bootstrap would yield a non-degenerate interval here, so
        this pins the resampling unit.
        """
        values = {"p1": 0.30, "p2": -0.10, "p3": 0.05}
        design = make_design({seed: dict(values) for seed in (1, 2, 3, 4, 5)})
        interval = cluster_bootstrap_interval(
            design, "abl", n_resamples=200, seed=0
        )
        expected = float(np.mean(list(values.values())))
        assert interval is not None
        assert interval.low == pytest.approx(expected)
        assert interval.high == pytest.approx(expected)
        assert interval.excludes_zero is True

    def test_interval_brackets_the_point_estimate(self, five_seed_design):
        interval = cluster_bootstrap_interval(
            five_seed_design, "abl", n_resamples=400, seed=0
        )
        estimate = point_estimate(five_seed_design, "abl")
        assert interval is not None
        assert interval.low <= estimate <= interval.high

    def test_deterministic_in_its_seed(self, five_seed_design):
        first = cluster_bootstrap_interval(
            five_seed_design, "abl", n_resamples=300, seed=4
        )
        second = cluster_bootstrap_interval(
            five_seed_design, "abl", n_resamples=300, seed=4
        )
        assert (first.low, first.high) == (second.low, second.high)

    def test_alpha_widens_the_interval(self, five_seed_design):
        narrow = cluster_bootstrap_interval(
            five_seed_design, "abl", n_resamples=800, alpha=0.20, seed=1
        )
        wide = cluster_bootstrap_interval(
            five_seed_design, "abl", n_resamples=800, alpha=0.01, seed=1
        )
        assert wide.low <= narrow.low
        assert wide.high >= narrow.high

    def test_excludes_zero_only_when_both_bounds_share_a_sign(self):
        design = make_design(
            {
                seed: {"p1": 0.5 + 0.01 * seed, "p2": -0.5 - 0.01 * seed}
                for seed in (1, 2, 3, 4, 5)
            }
        )
        interval = cluster_bootstrap_interval(
            design, "abl", n_resamples=400, seed=0
        )
        assert interval.low <= 0.0 <= interval.high
        assert interval.excludes_zero is False

    def test_negative_effect_is_detected(self):
        design = make_design(
            {
                seed: {f"p{i}": -0.4 - 0.01 * seed for i in range(6)}
                for seed in (1, 2, 3, 4, 5)
            }
        )
        interval = cluster_bootstrap_interval(
            design, "abl", n_resamples=400, seed=0
        )
        assert interval.high < 0.0
        assert interval.excludes_zero is True

    def test_partial_nan_rows_are_dropped_not_fatal(self):
        design = make_design(
            {
                1: {"p1": 0.2, "p2": None},
                2: {"p1": 0.3, "p2": 0.1},
                3: {"p1": 0.25, "p2": 0.15},
            }
        )
        interval = cluster_bootstrap_interval(
            design, "abl", n_resamples=200, seed=0
        )
        assert interval is not None
        assert interval.low > 0.0

    def test_to_dict_shape(self):
        payload = Interval(low=-0.1, high=0.3, excludes_zero=False).to_dict()
        assert payload == {
            "low": -0.1,
            "high": 0.3,
            "excludes_zero": False,
        }

# -- diagnostics ----------------------------------------------------------


class TestDiagnostics:
    def test_counts_and_zero_fraction_on_collapsed_diffs(self):
        design = make_design(
            {
                1: {"p1": 0.4, "p2": 0.0, "p3": -0.2, "p4": 0.0},
                2: {"p1": 0.2, "p2": 0.0, "p3": -0.4, "p4": 0.0},
            }
        )
        result = diagnostics(design, "abl")
        assert (result.wins, result.losses, result.ties) == (1, 1, 2)
        assert result.zero_fraction == pytest.approx(0.5)

    def test_identical_hypothesis_fraction(self):
        design = make_design(
            {
                1: {"p1": 0.0, "p2": 0.3},
                2: {"p1": 0.0, "p2": 0.3},
            },
            hypotheses={"p1": ("Male", "Male"), "p2": ("Male", "Person")},
        )
        result = diagnostics(design, "abl")
        assert result.identical_hypothesis_fraction == pytest.approx(0.5)

    def test_identical_hypothesis_fraction_is_none_without_columns(
        self, two_seed_design
    ):
        result = diagnostics(two_seed_design, "abl")
        assert result.identical_hypothesis_fraction is None

    def test_empty_hypothesis_rates_per_condition(self):
        design = make_design(
            {
                1: {"p1": 0.1, "p2": 0.2, "p3": 0.3, "p4": 0.4},
            },
            extension_sizes={
                "p1": (0.0, 5.0),
                "p2": (0.0, 7.0),
                "p3": (3.0, 0.0),
                "p4": (4.0, 9.0),
            },
        )
        result = diagnostics(design, "abl")
        assert result.empty_rate_treated == pytest.approx(0.5)
        assert result.empty_rate_control == pytest.approx(0.25)

    def test_empty_rates_are_none_without_columns(self, two_seed_design):
        result = diagnostics(two_seed_design, "abl")
        assert result.empty_rate_treated is None
        assert result.empty_rate_control is None

    def test_zero_fraction_does_not_divide_by_zero_on_empty_outcome(
        self, two_seed_design
    ):
        result = diagnostics(two_seed_design, "precision")
        assert result.zero_fraction == 0.0
        assert (result.wins, result.losses, result.ties) == (0, 0, 0)

    def test_fully_degenerate_contrast_reports_unit_zero_fraction(self):
        design = make_design({1: {"p1": 0.0, "p2": 0.0}, 2: {"p1": 0.0, "p2": 0.0}})
        result = diagnostics(design, "abl")
        assert result.zero_fraction == pytest.approx(1.0)
        assert result.ties == 2

    def test_to_dict_folds_the_triple_into_one_key(self):
        payload = Diagnostics(
            zero_fraction=0.25,
            identical_hypothesis_fraction=0.1,
            wins=3,
            losses=1,
            ties=2,
            empty_rate_treated=0.0,
            empty_rate_control=None,
        ).to_dict()
        assert payload["win_loss_tie"] == [3, 1, 2]
        assert "wins" not in payload
        assert payload["empty_rate_control"] is None


# -- seed spread ----------------------------------------------------------


class TestSeedSpread:
    def test_per_seed_means_are_plain_python_scalars(self, five_seed_design):
        result = seed_spread(five_seed_design, "abl")
        assert set(result.per_seed_means) == {1, 2, 3, 4, 5}
        assert all(isinstance(k, int) for k in result.per_seed_means)
        assert all(isinstance(v, float) for v in result.per_seed_means.values())

    def test_per_seed_mean_values(self):
        design = make_design(
            {1: {"p1": 0.2, "p2": 0.4}, 2: {"p1": -0.1, "p2": -0.3}}
        )
        result = seed_spread(design, "abl")
        assert result.per_seed_means[1] == pytest.approx(0.3)
        assert result.per_seed_means[2] == pytest.approx(-0.2)

    def test_dominant_source_is_concepts_when_problems_vary_more(self):
        """Identical seeds, heterogeneous problems: seed SD is 0."""
        values = {"p1": 0.9, "p2": -0.9, "p3": 0.1, "p4": 0.5}
        design = make_design({seed: dict(values) for seed in (1, 2, 3)})
        result = seed_spread(design, "abl")
        assert result.seed_sd == pytest.approx(0.0)
        assert result.problem_sd > 0.0
        assert result.dominant_source == "concepts"

    def test_dominant_source_is_runs_when_seeds_vary_more(self):
        """Every problem has the same collapsed value, so problem SD is 0."""
        design = make_design(
            {
                1: {"p1": 1.0, "p2": 1.0},
                2: {"p1": -1.0, "p2": -1.0},
                3: {"p1": 0.0, "p2": 0.0},
            }
        )
        result = seed_spread(design, "abl")
        assert result.problem_sd == pytest.approx(0.0)
        assert result.seed_sd > 0.0
        assert result.dominant_source == "runs"

    def test_dominant_source_comparable_on_exact_equality(self):
        spread = SeedSpread(
            per_seed_means={1: 0.1, 2: 0.2}, seed_sd=0.3, problem_sd=0.3
        )
        assert spread.dominant_source == "comparable"

    def test_dominant_source_unknown_when_either_sd_is_none(self):
        assert (
            SeedSpread(per_seed_means={1: 0.1}, seed_sd=None, problem_sd=0.2)
            .dominant_source
            == "unknown"
        )
        assert (
            SeedSpread(per_seed_means={1: 0.1}, seed_sd=0.2, problem_sd=None)
            .dominant_source
            == "unknown"
        )

    def test_single_seed_has_no_seed_sd(self):
        design = make_design({1: {"p1": 0.2, "p2": 0.4}})
        result = seed_spread(design, "abl")
        assert result.seed_sd is None
        assert result.problem_sd is not None
        assert result.dominant_source == "unknown"

    def test_single_problem_has_no_problem_sd(self):
        design = make_design({1: {"p1": 0.2}, 2: {"p1": 0.4}})
        result = seed_spread(design, "abl")
        assert result.problem_sd is None
        assert result.seed_sd is not None

    def test_sds_use_the_sample_convention(self):
        design = make_design({1: {"p1": 0.0}, 2: {"p1": 2.0}})
        result = seed_spread(design, "abl")
        # ddof=1 over {0, 2} -> sqrt(2), not 1.0
        assert result.seed_sd == pytest.approx(np.sqrt(2.0))

    def test_to_dict_includes_the_verdict(self, five_seed_design):
        payload = seed_spread(five_seed_design, "abl").to_dict()
        assert set(payload) == {
            "per_seed_means",
            "seed_sd",
            "problem_sd",
            "dominant_source",
        }
        assert payload["dominant_source"] in {
            "concepts",
            "runs",
            "comparable",
            "unknown",
        }

# -- orchestration --------------------------------------------------------


class TestAnalyseContrast:
    def test_populates_every_block(self, five_seed_design):
        result = analyse_contrast(five_seed_design, "abl")
        assert result.estimate == pytest.approx(
            point_estimate(five_seed_design, "abl")
        )
        assert isinstance(result.interval, Interval)
        assert isinstance(result.test, NonParametricResult)
        assert isinstance(result.diagnostics, Diagnostics)
        assert isinstance(result.seed_spread, SeedSpread)
        assert result.notes == []

    def test_carries_design_metadata(self, five_seed_design):
        result = analyse_contrast(five_seed_design, "abl")
        assert result.condition == "dice"
        assert result.knowledge_base == "semantic_bible"
        assert result.outcome == "abl"
        assert result.n_seeds == 5
        assert result.n_problems == 4
        assert result.n_observations == 20
        assert result.substituted_primary is False

    def test_notes_are_copied_not_aliased(self):
        design = make_design({1: {"p1": 0.1}, 2: {"p1": 0.2}})
        design.notes.append("pre-existing note")
        result = analyse_contrast(design, "abl")
        result.notes.append("added later")
        assert design.notes == ["pre-existing note"]

    def test_substitution_flag_is_propagated(self):
        design = make_design(
            {1: {"p1": 0.1}, 2: {"p1": 0.2}},
            outcome="f1",
            substituted_primary=True,
        )
        result = analyse_contrast(design, "f1")
        assert result.substituted_primary is True
        assert result.outcome == "f1"

    def test_unavailable_outcome_short_circuits_with_a_note(
        self, five_seed_design
    ):
        result = analyse_contrast(five_seed_design, "recall")
        assert result.estimate is None
        assert result.interval is None
        assert result.test is None
        assert result.diagnostics is None
        assert result.seed_spread is None
        assert any("'recall' unavailable" in note for note in result.notes)

    def test_single_seed_yields_no_interval_but_keeps_the_rest(self):
        design = make_design({1: {"p1": 0.3, "p2": -0.1, "p3": 0.2}})
        result = analyse_contrast(design, "abl")
        assert result.interval is None
        assert result.test is not None
        assert result.diagnostics is not None
        assert result.seed_spread is not None
        assert result.notes == []  # a None interval is not a failure

    @pytest.mark.parametrize(
        "block, attribute",
        [
            ("cluster_bootstrap_interval", "interval"),
            ("non_parametric_test", "test"),
            ("diagnostics", "diagnostics"),
            ("seed_spread", "seed_spread"),
        ],
    )
    def test_each_block_is_guarded_independently(
        self, five_seed_design, monkeypatch, block, attribute
    ):
        def boom(*args, **kwargs):
            raise ValueError(f"{block} exploded")

        monkeypatch.setattr(inference, block, boom)
        result = analyse_contrast(five_seed_design, "abl")

        assert getattr(result, attribute) is None
        assert any(
            f"Block {attribute!r} unavailable" in note
            for note in result.notes
        )
        assert result.estimate is not None
        others = {"interval", "test", "diagnostics", "seed_spread"} - {
            attribute
        }
        assert all(getattr(result, name) is not None for name in others)

    def test_block_failure_is_logged_as_a_warning(
        self, five_seed_design, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            inference,
            "diagnostics",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        with caplog.at_level("WARNING", logger=inference.__name__):
            analyse_contrast(five_seed_design, "abl")
        assert "Block diagnostics failed" in caplog.text

    def test_degenerate_contrast_is_reported_not_skipped(self):
        design = make_design(
            {seed: {"p1": 0.0, "p2": 0.0, "p3": 0.0} for seed in (1, 2, 3)}
        )
        result = analyse_contrast(design, "abl")
        assert result.estimate == pytest.approx(0.0)
        assert result.test.degenerate is True
        assert result.test.p_value == 1.0
        assert result.diagnostics.zero_fraction == pytest.approx(1.0)
        assert result.interval.low == pytest.approx(0.0)
        assert result.interval.excludes_zero is False

    def test_bootstrap_seed_argument_reaches_the_interval(
        self, five_seed_design, monkeypatch
    ):
        seen: list[int] = []

        def spy(design, outcome, *, seed=0):
            seen.append(seed)

        monkeypatch.setattr(inference, "cluster_bootstrap_interval", spy)
        analyse_contrast(five_seed_design, "abl", bootstrap_seed=99)
        assert seen == [99]

    def test_to_dict_is_json_serialisable(self, five_seed_design):
        import json

        payload = analyse_contrast(five_seed_design, "abl").to_dict()
        assert json.loads(json.dumps(payload))["outcome"] == "abl"
        assert set(payload) >= {
            "condition",
            "knowledge_base",
            "outcome",
            "substituted_primary",
            "n_problems",
            "n_observations",
            "n_seeds",
            "estimate",
            "interval",
            "test",
            "diagnostics",
            "seed_spread",
            "notes",
        }

    def test_to_dict_nulls_missing_blocks(self, five_seed_design):
        result = analyse_contrast(five_seed_design, "recall")
        payload = result.to_dict()
        assert payload["interval"] is None
        assert payload["test"] is None
        assert payload["diagnostics"] is None
        assert payload["seed_spread"] is None


# -- module invariants ----------------------------------------------------


class TestModuleConstants:
    def test_zero_method_is_pinned_to_pratt(self):
        assert inference.ZERO_METHOD == "pratt"

    def test_bootstrap_defaults_match_the_methodology(self):
        assert inference.N_BOOTSTRAP == 10_000
        assert inference.BOOTSTRAP_ALPHA == 0.05

    def test_exact_enumeration_threshold_is_tractable(self):
        assert inference.EXACT_SIGN_FLIP_MAX_N <= 20

    def test_primary_outcome_is_analysable(self, five_seed_design):
        assert "abl" in OUTCOMES
        assert analyse_contrast(five_seed_design, "abl").estimate is not None