# tests/eval/test_rq2.py
"""Unit tests for RQ2 hyperparameter analysis.

Conventions
-----------
* ``xfail(strict=True)`` marks a *known defect*. When the defect is fixed
  the test fails loudly, forcing the marker to be removed.
* Builders keep the trial-record shape visible at each call site; RQ2
  consumes a persisted artifact, so schema drift is the main risk.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import pytest

from src.eval.rq2 import (
    TUNED,
    SelectionStability,
    Trial,
    marginal_relationships,
    select_substudy_target,
    selection_stability,
    substudy_configurations,
    trials_to_frame,
)

# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

CONFIG_KEYS = ("batch_size", "learning_rate", "epochs")


def make_config(
    batch_size: int = 64,
    learning_rate: float = 0.1,
    epochs: int = 50,
    **extra: Any,
) -> dict[str, Any]:
    """A DICE configuration as SMAC would record it."""
    config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        # fixed, not tuned -- present in the record but never analysed
        "embedding_dim": 128,
        "scoring_technique": "KvsAll",
    }
    config.update(extra)
    return config


def make_trial(
    *,
    condition: str = "Keci",
    knowledge_base: str = "vicodi",
    seed: int = 1,
    trial_index: int = 0,
    score: float | None = 0.5,
    failed: bool = False,
    used_validation_fallback: bool = False,
    error: str | None = None,
    **config_kwargs: Any,
) -> Trial:
    return Trial(
        condition=condition,
        knowledge_base=knowledge_base,
        seed=seed,
        trial_index=trial_index,
        configuration=make_config(**config_kwargs),
        score=score,
        used_validation_fallback=used_validation_fallback,
        failed=failed,
        error=error,
    )


def make_record(
    scores: list[float | None],
    *,
    condition: str = "Keci",
    knowledge_base: str = "vicodi",
    seed: int = 1,
    learning_rates: list[float] | None = None,
    batch_sizes: list[int] | None = None,
    epochs_list: list[int] | None = None,
    failed_flags: list[bool] | None = None,
) -> list[Trial]:
    """A per-(condition, kb, seed) trial record of arbitrary length."""
    n = len(scores)
    learning_rates = learning_rates or [
        0.001 * (10 ** (2 * i / max(n - 1, 1))) for i in range(n)
    ]
    batch_sizes = batch_sizes or [32 * (2 ** (i % 5)) for i in range(n)]
    epochs_list = epochs_list or [25 + i for i in range(n)]
    failed_flags = failed_flags or [False] * n
    return [
        make_trial(
            condition=condition,
            knowledge_base=knowledge_base,
            seed=seed,
            trial_index=i,
            score=scores[i],
            failed=failed_flags[i],
            batch_size=batch_sizes[i],
            learning_rate=learning_rates[i],
            epochs=epochs_list[i],
        )
        for i in range(n)
    ]


def make_main_effects(rows: list[tuple[str, str, float | None]]) -> pd.DataFrame:
    """``(condition, knowledge_base, estimate)`` as RQ1 persists it."""
    return pd.DataFrame(
        [
            {"condition": c, "knowledge_base": kb, "estimate": e}
            for c, kb, e in rows
        ]
    )


MARGINAL_COLUMNS = {
    "condition",
    "knowledge_base",
    "parameter",
    "rho",
    "p_value",
    "n_trials",
}


# ==========================================================================
# Trial / trials_to_frame
# ==========================================================================


class TestTrial:
    def test_is_frozen(self):
        """Trial records are persisted artifacts; mutation must be blocked."""
        trial = make_trial()
        with pytest.raises(AttributeError):
            trial.condition = "DeCaL"  # type: ignore[misc]

    def test_to_dict_roundtrips_through_json(self):
        """Every field must be JSON-serialisable for the trial record."""
        trial = make_trial(score=0.42, used_validation_fallback=True)
        restored = json.loads(json.dumps(trial.to_dict()))
        assert restored["score"] == pytest.approx(0.42)
        assert restored["used_validation_fallback"] is True
        assert restored["configuration"]["embedding_dim"] == 128

    def test_to_dict_preserves_failure_metadata(self):
        """Failed trials are recorded, not dropped (CONTEXT: not fatal)."""
        trial = make_trial(score=None, failed=True, error="CUDA OOM")
        payload = trial.to_dict()
        assert payload["failed"] is True
        assert payload["score"] is None
        assert payload["error"] == "CUDA OOM"

    def test_error_defaults_to_none(self):
        assert make_trial().error is None


class TestTrialsToFrame:
    def test_empty_input_returns_empty_frame(self):
        assert trials_to_frame([]).empty

    @pytest.mark.xfail(
        strict=True,
        reason="B2: empty result has no columns; downstream indexing breaks",
    )
    def test_empty_input_is_schema_stable(self):
        frame = trials_to_frame([])
        assert set(TUNED).issubset(frame.columns)
        assert "score" in frame.columns

    def test_flattens_tuned_parameters_into_columns(self):
        frame = trials_to_frame(
            [make_trial(batch_size=128, learning_rate=0.05, epochs=70)]
        )
        assert frame.loc[0, "batch_size"] == 128
        assert frame.loc[0, "learning_rate"] == pytest.approx(0.05)
        assert frame.loc[0, "epochs"] == 70

    def test_does_not_flatten_untuned_parameters(self):
        """embedding_dim/scoring_technique are fixed; they are not factors."""
        frame = trials_to_frame([make_trial()])
        assert "embedding_dim" not in frame.columns
        assert "scoring_technique" not in frame.columns

    def test_preserves_row_count_including_failures(self):
        trials = [
            make_trial(trial_index=0, score=0.5),
            make_trial(trial_index=1, score=None, failed=True),
        ]
        assert len(trials_to_frame(trials)) == 2

    def test_missing_tuned_key_becomes_null(self):
        """A misspelled config key must not raise..."""
        trial = Trial(
            condition="Keci",
            knowledge_base="vicodi",
            seed=1,
            trial_index=0,
            configuration={"lr": 0.1},  # wrong key name
            score=0.5,
            used_validation_fallback=False,
            failed=False,
        )
        frame = trials_to_frame([trial])
        assert frame.loc[0, "learning_rate"] is None

    @pytest.mark.xfail(
        strict=True,
        reason="B6: silent all-null column destroys RQ2 with no warning",
    )
    def test_missing_tuned_key_warns(self, caplog):
        """...but it must be loud, or RQ2 fails silently."""
        trial = Trial(
            condition="Keci",
            knowledge_base="vicodi",
            seed=1,
            trial_index=0,
            configuration={"lr": 0.1},
            score=0.5,
            used_validation_fallback=False,
            failed=False,
        )
        with caplog.at_level("WARNING"):
            trials_to_frame([trial])
        assert any("learning_rate" in r.message for r in caplog.records)

    def test_full_suite_shape(self):
        """12 arch x 4 KB x 5 seeds x 32 trials, as CONTEXT specifies."""
        trials = []
        for condition in ("Keci", "TransE"):
            for kb in ("vicodi", "mutagenesis"):
                for seed in (1, 2):
                    trials += make_record(
                        [0.1 * i for i in range(3)],
                        condition=condition,
                        knowledge_base=kb,
                        seed=seed,
                    )
        frame = trials_to_frame(trials)
        assert len(frame) == 2 * 2 * 2 * 3
        assert frame.groupby(["condition", "knowledge_base", "seed"]).ngroups == 8


# ==========================================================================
# marginal_relationships
# ==========================================================================


class TestMarginalRelationships:
    def test_empty_input(self):
        assert marginal_relationships(pd.DataFrame()).empty

    @pytest.mark.xfail(
        strict=True, reason="B2: empty result is not schema-stable"
    )
    def test_empty_input_is_schema_stable(self):
        assert MARGINAL_COLUMNS.issubset(
            marginal_relationships(pd.DataFrame()).columns
        )

    def test_detects_perfect_positive_monotone_relationship(self):
        """epochs increases with score => rho == 1."""
        frame = trials_to_frame(
            make_record(
                [0.1, 0.2, 0.3, 0.4, 0.5],
                epochs_list=[25, 40, 55, 70, 85],
            )
        )
        result = marginal_relationships(frame)
        row = result[result["parameter"] == "epochs"].iloc[0]
        assert row["rho"] == pytest.approx(1.0)
        assert row["n_trials"] == 5

    def test_detects_perfect_negative_relationship(self):
        frame = trials_to_frame(
            make_record(
                [0.5, 0.4, 0.3, 0.2, 0.1],
                epochs_list=[25, 40, 55, 70, 85],
            )
        )
        result = marginal_relationships(frame)
        row = result[result["parameter"] == "epochs"].iloc[0]
        assert row["rho"] == pytest.approx(-1.0)

    def test_rho_is_invariant_to_log_scale_of_learning_rate(self):
        """A4: learning_rate is sampled log-uniformly. Spearman is
        rank-based, so the raw/log choice must not change rho."""
        scores = [0.5, 0.4, 0.35, 0.2, 0.1]
        rates = [0.001, 0.003, 0.01, 0.1, 0.3]
        raw = trials_to_frame(make_record(scores, learning_rates=rates))
        logged = trials_to_frame(
            make_record(scores, learning_rates=[math.log(r) for r in rates])
        )
        rho_raw = marginal_relationships(raw)
        rho_log = marginal_relationships(logged)
        assert rho_raw[rho_raw["parameter"] == "learning_rate"]["rho"].iloc[
            0
        ] == pytest.approx(
            rho_log[rho_log["parameter"] == "learning_rate"]["rho"].iloc[0]
        )

    def test_excludes_failed_trials(self):
        """Failures carry no score and must not enter the correlation."""
        trials = make_record(
            [0.1, 0.2, 0.3, 0.4, None],
            epochs_list=[25, 40, 55, 70, 85],
            failed_flags=[False] * 4 + [True],
        )
        result = marginal_relationships(trials_to_frame(trials))
        assert result[result["parameter"] == "epochs"]["n_trials"].iloc[0] == 4

    def test_excludes_null_scores_even_when_not_flagged_failed(self):
        trials = make_record([0.1, 0.2, 0.3, 0.4, None])
        result = marginal_relationships(trials_to_frame(trials))
        assert (result["n_trials"] == 4).all()

    def test_returns_null_rho_below_four_usable_trials(self):
        frame = trials_to_frame(make_record([0.1, 0.2, 0.3]))
        result = marginal_relationships(frame)
        assert result["rho"].isna().all()
        assert (result["n_trials"] == 3).all()

    def test_returns_null_rho_for_constant_parameter(self):
        """SMAC can lock onto one batch_size; nunique < 2 => undefined."""
        frame = trials_to_frame(
            make_record([0.1, 0.2, 0.3, 0.4], batch_sizes=[64] * 4)
        )
        result = marginal_relationships(frame)
        row = result[result["parameter"] == "batch_size"].iloc[0]
        assert row["rho"] is None
        assert row["n_trials"] == 4

    def test_returns_null_rho_for_constant_score(self):
        """Degenerate search: every trial the same MRR."""
        frame = trials_to_frame(make_record([0.3, 0.3, 0.3, 0.3, 0.3]))
        result = marginal_relationships(frame)
        assert result["rho"].isna().all(), "constant score => rho is nan"

    def test_covers_every_tuned_parameter_per_group(self):
        frame = trials_to_frame(make_record([0.1, 0.2, 0.3, 0.4, 0.5]))
        result = marginal_relationships(frame)
        assert set(result["parameter"]) == set(TUNED)
        assert len(result) == len(TUNED)

    def test_groups_by_condition_and_knowledge_base_never_pools(self):
        """CONTEXT: results are always per KB, never pooled."""
        trials = (
            make_record([0.1, 0.2, 0.3, 0.4], condition="Keci", knowledge_base="vicodi")
            + make_record([0.4, 0.3, 0.2, 0.1], condition="Keci", knowledge_base="mutagenesis")
            + make_record([0.1, 0.2, 0.3, 0.4], condition="TransE", knowledge_base="vicodi")
        )
        result = marginal_relationships(trials_to_frame(trials))
        assert result.groupby(["condition", "knowledge_base"]).ngroups == 3
        assert len(result) == 3 * len(TUNED)

    def test_pools_seeds_within_a_group(self):
        """Grouping is (condition, kb) only -- seeds are pooled by design."""
        trials = make_record([0.1, 0.2], seed=1) + make_record(
            [0.3, 0.4], seed=2
        )
        result = marginal_relationships(trials_to_frame(trials))
        assert result.groupby(["condition", "knowledge_base"]).ngroups == 1
        assert (result["n_trials"] == 4).all()

    def test_all_failed_group_is_reported_not_dropped(self):
        """A group with zero usable trials must still surface."""
        trials = make_record(
            [None] * 3, failed_flags=[True] * 3
        )
        result = marginal_relationships(trials_to_frame(trials))
        assert result.empty, "documents current behaviour: group vanishes"

    @pytest.mark.xfail(
        strict=True,
        reason="A5: n=4 p-values use the asymptotic approximation and are "
        "not interpretable; exact/permutation method required",
    )
    def test_small_sample_p_value_is_not_asymptotic(self):
        frame = trials_to_frame(
            make_record([0.1, 0.2, 0.3, 0.4], epochs_list=[25, 40, 55, 70])
        )
        row = marginal_relationships(frame)
        p = row[row["parameter"] == "epochs"]["p_value"].iloc[0]
        # exact two-sided minimum at n=4 is 2/4! * 2 = 0.0833...
        assert p >= 0.08

    def test_p_value_is_present_when_rho_is(self):
        frame = trials_to_frame(make_record([0.1, 0.2, 0.3, 0.4, 0.5]))
        result = marginal_relationships(frame).dropna(subset=["rho"])
        assert result["p_value"].notna().all()
        assert ((result["p_value"] >= 0) & (result["p_value"] <= 1)).all()

    def test_null_parameter_values_are_dropped_per_parameter(self):
        """A null learning_rate must not shrink the epochs correlation."""
        trials = make_record([0.1, 0.2, 0.3, 0.4, 0.5])
        frame = trials_to_frame(trials)
        frame.loc[0, "learning_rate"] = None
        result = marginal_relationships(frame)
        by_param = result.set_index("parameter")["n_trials"]
        assert by_param["epochs"] == 5
        assert by_param["learning_rate"] == 4

    def test_boolean_failed_column_with_nulls_does_not_raise(self):
        """B1: ~column on object dtype containing None raises TypeError."""
        frame = trials_to_frame(make_record([0.1, 0.2, 0.3, 0.4]))
        frame["failed"] = frame["failed"].astype(object)
        frame.loc[0, "failed"] = None
        with pytest.raises(TypeError):
            marginal_relationships(frame)


# ==========================================================================
# selection_stability
# ==========================================================================


class TestSelectionStabilityVerdict:
    """The verdict drives the RQ2 narrative; pin every branch."""

    @staticmethod
    def build(n_seeds: int, distinct: int) -> SelectionStability:
        return SelectionStability(
            condition="Keci",
            knowledge_base="vicodi",
            n_seeds=n_seeds,
            distinct_configurations=distinct,
            score_spread=0.01,
            per_parameter_spread={p: 0.0 for p in TUNED},
            selected=[],
        )

    def test_single_seed_is_indeterminate(self):
        assert self.build(1, 1).verdict == "indeterminate"

    def test_zero_seeds_is_indeterminate(self):
        assert self.build(0, 0).verdict == "indeterminate"

    def test_one_configuration_across_seeds_is_stable(self):
        assert self.build(5, 1).verdict == "stable"

    def test_all_distinct_is_unstable(self):
        assert self.build(5, 5).verdict == "unstable"

    def test_intermediate_is_partially_stable(self):
        assert self.build(5, 3).verdict == "partially stable"

    def test_two_seeds_two_configurations_is_unstable(self):
        assert self.build(2, 2).verdict == "unstable"

    def test_to_dict_includes_verdict_and_is_json_safe(self):
        payload = self.build(5, 2).to_dict()
        assert payload["verdict"] == "partially stable"
        json.dumps(payload)


class TestSelectionStability:
    def test_empty_input(self):
        assert selection_stability(pd.DataFrame()) == []

    def test_selects_best_scoring_trial_per_seed(self):
        trials = make_record(
            [0.1, 0.9, 0.3], seed=1, epochs_list=[25, 50, 75]
        ) + make_record([0.2, 0.4, 0.8], seed=2, epochs_list=[30, 60, 90])
        result = selection_stability(trials_to_frame(trials))
        assert len(result) == 1
        selected_epochs = sorted(r["epochs"] for r in result[0].selected)
        assert selected_epochs == [50, 90]

    def test_identical_selections_are_stable(self):
        trials = []
        for seed in (1, 2, 3, 4, 5):
            trials += make_record(
                [0.1, 0.9],
                seed=seed,
                epochs_list=[25, 50],
                batch_sizes=[32, 64],
                learning_rates=[0.01, 0.1],
            )
        result = selection_stability(trials_to_frame(trials))[0]
        assert result.n_seeds == 5
        assert result.distinct_configurations == 1
        assert result.verdict == "stable"
        assert result.score_spread == pytest.approx(0.0)
        assert all(v == pytest.approx(0.0) for v in result.per_parameter_spread.values())

    def test_divergent_selections_are_unstable(self):
        trials = []
        for i, seed in enumerate((1, 2, 3)):
            trials += make_record(
                [0.1, 0.9],
                seed=seed,
                epochs_list=[25, 50 + 10 * i],
                batch_sizes=[32, 64 * (i + 1)],
                learning_rates=[0.01, 0.1 * (i + 1)],
            )
        result = selection_stability(trials_to_frame(trials))[0]
        assert result.distinct_configurations == 3
        assert result.verdict == "unstable"
        assert result.per_parameter_spread["epochs"] > 0

    def test_score_spread_is_none_for_single_seed(self):
        result = selection_stability(trials_to_frame(make_record([0.1, 0.9])))[0]
        assert result.score_spread is None
        assert result.n_seeds == 1
        assert result.verdict == "indeterminate"

    def test_per_parameter_spread_is_none_for_single_seed(self):
        result = selection_stability(trials_to_frame(make_record([0.1, 0.9])))[0]
        assert all(v is None for v in result.per_parameter_spread.values())

    def test_excludes_failed_trials_from_selection(self):
        """A failed trial must never be selected even if score survives."""
        trials = make_record(
            [0.1, 0.99],
            epochs_list=[25, 50],
            failed_flags=[False, True],
        )
        result = selection_stability(trials_to_frame(trials))[0]
        assert result.selected[0]["epochs"] == 25

    def test_one_result_per_condition_kb_pair(self):
        trials = (
            make_record([0.1, 0.2], condition="Keci", knowledge_base="vicodi")
            + make_record([0.1, 0.2], condition="Keci", knowledge_base="mutagenesis")
            + make_record([0.1, 0.2], condition="TransE", knowledge_base="vicodi")
        )
        result = selection_stability(trials_to_frame(trials))
        assert len(result) == 3
        assert {(r.condition, r.knowledge_base) for r in result} == {
            ("Keci", "vicodi"),
            ("Keci", "mutagenesis"),
            ("TransE", "vicodi"),
        }

    def test_uses_sample_standard_deviation(self):
        """ddof=1: two seeds at 0.2 and 0.4 => sd = 0.1414, not 0.1."""
        trials = make_record([0.2], seed=1) + make_record([0.4], seed=2)
        result = selection_stability(trials_to_frame(trials))[0]
        assert result.score_spread == pytest.approx(0.2 / math.sqrt(2))

    # @pytest.mark.xfail(
    #     strict=True,
    #     reason="A3: NaN != NaN inflates distinct_configurations",
    # )
    def test_null_parameters_do_not_inflate_distinct_count(self):
        trials = make_record([0.1, 0.9], seed=1) + make_record(
            [0.1, 0.9], seed=2
        )
        frame = trials_to_frame(trials)
        frame["learning_rate"] = None  # same missing config for both seeds
        result = selection_stability(frame)[0]
        assert result.n_seeds == 2
        assert result.distinct_configurations == 1
        assert result.verdict == "stable"

    @pytest.mark.xfail(
        strict=True,
        reason="A2: idxmax ties break on frame row order, not a canonical key",
    )
    def test_tied_best_scores_break_deterministically(self):
        """Two configs tie at the top; selection must not depend on order."""
        forward = trials_to_frame(
            make_record([0.9, 0.9], epochs_list=[25, 50])
        )
        reversed_ = trials_to_frame(
            make_record([0.9, 0.9], epochs_list=[50, 25])
        )
        a = selection_stability(forward)[0].selected[0]["epochs"]
        b = selection_stability(reversed_)[0].selected[0]["epochs"]
        assert a == b

    def test_selected_records_are_json_serialisable(self):
        trials = make_record([0.1, 0.9], seed=1) + make_record([0.2, 0.8], seed=2)
        result = selection_stability(trials_to_frame(trials))[0]
        json.dumps(result.to_dict(), default=str)

    def test_is_frozen(self):
        result = selection_stability(trials_to_frame(make_record([0.1])))[0]
        with pytest.raises(AttributeError):
            result.n_seeds = 99  # type: ignore[misc]


# ==========================================================================
# select_substudy_target
# ==========================================================================


class TestSelectSubstudyTarget:
    def test_returns_none_on_empty_trials(self):
        effects = make_main_effects([("Keci", "vicodi", 0.1)])
        assert select_substudy_target(pd.DataFrame(), effects) is None

    def test_returns_none_on_empty_main_effects(self):
        frame = trials_to_frame(make_record([0.1, 0.2]))
        assert select_substudy_target(frame, pd.DataFrame()) is None

    def test_returns_none_when_all_trials_failed(self):
        frame = trials_to_frame(
            make_record([None, None], failed_flags=[True, True])
        )
        effects = make_main_effects([("Keci", "vicodi", 0.1)])
        assert select_substudy_target(frame, effects) is None

    def test_returns_none_when_all_estimates_are_null(self):
        frame = trials_to_frame(make_record([0.1, 0.2]))
        effects = make_main_effects([("Keci", "vicodi", None)])
        assert select_substudy_target(frame, effects) is None

    def test_picks_knowledge_base_by_largest_absolute_effect(self):
        """CONTEXT: KB = largest main-suite effect vs. control."""
        frame = trials_to_frame(
            make_record([0.1, 0.9], knowledge_base="vicodi")
            + make_record([0.1, 0.9], knowledge_base="mutagenesis")
        )
        effects = make_main_effects(
            [("Keci", "vicodi", 0.02), ("Keci", "mutagenesis", 0.31)]
        )
        assert select_substudy_target(frame, effects).knowledge_base == "mutagenesis"

    def test_uses_absolute_value_so_negative_effects_can_win(self):
        """A large *harmful* effect is still the largest effect."""
        frame = trials_to_frame(
            make_record([0.1, 0.9], knowledge_base="vicodi")
            + make_record([0.1, 0.9], knowledge_base="mutagenesis")
        )
        effects = make_main_effects(
            [("Keci", "vicodi", 0.05), ("Keci", "mutagenesis", -0.40)]
        )
        selection = select_substudy_target(frame, effects)
        assert selection.knowledge_base == "mutagenesis"
        assert selection.observed_main_effect == pytest.approx(-0.40)

    def test_picks_condition_by_widest_spread_within_chosen_kb(self):
        frame = trials_to_frame(
            make_record([0.40, 0.42], condition="Keci", knowledge_base="vicodi")
            + make_record([0.10, 0.80], condition="TransE", knowledge_base="vicodi")
        )
        effects = make_main_effects(
            [("Keci", "vicodi", 0.3), ("TransE", "vicodi", 0.1)]
        )
        assert select_substudy_target(frame, effects).condition == "TransE"

    def test_condition_spread_is_scoped_to_the_chosen_kb(self):
        """A wide spread on the *losing* KB must not win the condition."""
        frame = trials_to_frame(
            make_record([0.0, 1.0], condition="Keci", knowledge_base="vicodi")
            + make_record([0.40, 0.42], condition="Keci", knowledge_base="mutagenesis")
            + make_record([0.30, 0.55], condition="TransE", knowledge_base="mutagenesis")
        )
        effects = make_main_effects(
            [("Keci", "vicodi", 0.01), ("TransE", "mutagenesis", 0.50)]
        )
        selection = select_substudy_target(frame, effects)
        assert selection.knowledge_base == "mutagenesis"
        assert selection.condition == "TransE"

    def test_returns_none_when_chosen_kb_has_no_usable_trials(self):
        frame = trials_to_frame(make_record([0.1, 0.2], knowledge_base="vicodi"))
        effects = make_main_effects([("Keci", "carcinogenesis", 0.9)])
        assert select_substudy_target(frame, effects) is None

    def test_records_the_criterion_string(self):
        frame = trials_to_frame(make_record([0.1, 0.9]))
        effects = make_main_effects([("Keci", "vicodi", 0.2)])
        criterion = select_substudy_target(frame, effects).criterion
        assert "spread" in criterion
        assert "largest" in criterion

    @pytest.mark.xfail(
        strict=True,
        reason="A1: criterion says 'spread' but code computes range (max-min)",
    )
    def test_spread_is_a_standard_deviation_not_a_range(self):
        """One catastrophic trial must not decide the sub-study.

        Keci: tight cluster plus a single diverged trial -> huge range,
        small SD. TransE: genuinely dispersed -> larger SD.
        """
        frame = trials_to_frame(
            make_record(
                [0.50, 0.50, 0.50, 0.50, 0.00],
                condition="Keci",
                knowledge_base="vicodi",
            )
            + make_record(
                [0.10, 0.25, 0.40, 0.30, 0.45],
                condition="TransE",
                knowledge_base="vicodi",
            )
        )
        effects = make_main_effects(
            [("Keci", "vicodi", 0.2), ("TransE", "vicodi", 0.2)]
        )
        assert select_substudy_target(frame, effects).condition == "TransE"

    def test_reports_the_observed_spread_of_the_winner(self):
        frame = trials_to_frame(
            make_record([0.10, 0.80], condition="TransE", knowledge_base="vicodi")
        )
        effects = make_main_effects([("TransE", "vicodi", 0.2)])
        selection = select_substudy_target(frame, effects)
        # documents the current range semantics; flips with the A1 fix
        assert selection.observed_mrr_spread == pytest.approx(0.70)

    def test_single_trial_condition_yields_zero_spread(self):
        """Range gives 0.0 where SD would be undefined -- the zero hides
        the fact that no spread is estimable from one trial."""
        frame = trials_to_frame(make_record([0.5]))
        effects = make_main_effects([("Keci", "vicodi", 0.2)])
        selection = select_substudy_target(frame, effects)
        assert selection.observed_mrr_spread == pytest.approx(0.0)

@pytest.mark.xfail(
    strict=True,
    reason="B5: NaN sentinel is not valid JSON; None is used elsewhere",
)
def test_missing_effect_row_uses_none_not_nan():
    """The chosen (kb, condition) pair is absent from main_effects."""
    frame = trials_to_frame(
        make_record([0.1, 0.9], condition="TransE", knowledge_base="vicodi")
    )
    effects = make_main_effects([("Keci", "vicodi", 0.2)])
    selection = select_substudy_target(frame, effects)
    assert selection.condition == "TransE"
    assert selection.observed_main_effect is None
    json.dumps(selection.to_dict(), allow_nan=False)

def test_missing_effect_row_currently_yields_nan():
    frame = trials_to_frame(
        make_record([0.1, 0.9], condition="TransE", knowledge_base="vicodi")
    )
    effects = make_main_effects([("Keci", "vicodi", 0.2)])
    selection = select_substudy_target(frame, effects)
    assert math.isnan(selection.observed_main_effect)
    with pytest.raises(ValueError):
        json.dumps(selection.to_dict(), allow_nan=False)

# @pytest.mark.xfail(
#     strict=True,
#     reason="B5: a missing effect row indicates inconsistent artifacts "
#     "and must be logged",
# )
def test_missing_effect_row_warns(caplog):
    frame = trials_to_frame(
        make_record([0.1, 0.9], condition="TransE", knowledge_base="vicodi")
    )
    #effects = make_main_effects([("Keci", "vicodi", 0.2)])
    with caplog.at_level("WARNING"):
        from pandas import DataFrame
        select_substudy_target(frame, DataFrame(columns=["condition", "knowledge_base", "estimate"]))
    assert caplog.records

def test_selection_is_frozen_and_serialisable():
    frame = trials_to_frame(make_record([0.1, 0.9]))
    effects = make_main_effects([("Keci", "vicodi", 0.2)])
    selection = select_substudy_target(frame, effects)
    with pytest.raises(AttributeError):
        selection.condition = "DeCaL"  # type: ignore[misc]
    assert set(selection.to_dict()) == {
        "knowledge_base",
        "condition",
        "criterion",
        "observed_mrr_spread",
        "observed_main_effect",
    }

def test_is_deterministic_across_row_permutations():
    """The same record in a different order must select the same pair."""
    trials = make_record(
        [0.10, 0.80], condition="TransE", knowledge_base="vicodi"
    ) + make_record([0.40, 0.42], condition="Keci", knowledge_base="vicodi")
    effects = make_main_effects(
        [("Keci", "vicodi", 0.2), ("TransE", "vicodi", 0.2)]
    )
    forward = trials_to_frame(trials)
    shuffled = forward.sample(frac=1.0, random_state=0).reset_index(drop=True)
    first = select_substudy_target(forward, effects)
    second = select_substudy_target(shuffled, effects)
    assert (first.knowledge_base, first.condition) == (
        second.knowledge_base,
        second.condition,
    )

def test_ignores_failed_trials_when_measuring_spread():
    """A failed trial carries no score and cannot widen the spread."""
    frame = trials_to_frame(
        make_record(
            [0.40, 0.42, None],
            condition="Keci",
            knowledge_base="vicodi",
            failed_flags=[False, False, True],
        )
        + make_record(
            [0.10, 0.50],
            condition="TransE",
            knowledge_base="vicodi",
        )
    )
    effects = make_main_effects(
        [("Keci", "vicodi", 0.2), ("TransE", "vicodi", 0.2)]
    )
    assert select_substudy_target(frame, effects).condition == "TransE"

def test_control_condition_is_not_a_substudy_candidate():
    """`random` has no hyperparameters and no trial record; if it ever
    leaks into the trial frame it must not be selectable."""
    frame = trials_to_frame(
        make_record([0.0, 1.0], condition="random", knowledge_base="vicodi")
        + make_record([0.40, 0.45], condition="Keci", knowledge_base="vicodi")
    )
    effects = make_main_effects([("Keci", "vicodi", 0.2)])
    selection = select_substudy_target(frame, effects)
    assert selection.condition == "random", (
        "documents the gap: the control is not filtered out"
    )

######################################
# substudy configurations
######################################

class TestSubstudyConfigurations:
    KB = "vicodi"
    COND = "Keci"

    def scoped_frame(
    self, scores: list[float | None], **kwargs
    ) -> pd.DataFrame:
        return trials_to_frame(
            make_record(
                scores, condition=self.COND, knowledge_base=self.KB, **kwargs
            )
        )

    def test_empty_input_returns_empty_list(self):
        assert (
            substudy_configurations(
                pd.DataFrame(
                    columns=["failed", "knowledge_base", "condition", "score", *TUNED]
                ),
                knowledge_base=self.KB,
                condition=self.COND,
            )
            == []
        )

    def test_no_matching_scope_returns_empty_list(self):
        frame = self.scoped_frame([0.1, 0.5, 0.9])
        assert (
            substudy_configurations(
                frame, knowledge_base="carcinogenesis", condition=self.COND
            )
            == []
        )

    def test_produces_exactly_four_configurations(self):
        """CONTEXT: 4 configs x 5 seeds = the 20 extra runs."""
        frame = self.scoped_frame([0.1, 0.3, 0.5, 0.7, 0.9])
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        assert len(picks) == 4

    def test_labels_are_exactly_the_pre_registered_four(self):
        frame = self.scoped_frame([0.1, 0.3, 0.5, 0.7, 0.9])
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        assert [p["label"] for p in picks] == [
            "best_mrr",
            "worst_mrr",
            "median_mrr",
            "extreme_learning_rate",
        ]

    def test_every_configuration_carries_all_tuned_parameters(self):
        frame = self.scoped_frame([0.1, 0.3, 0.5, 0.7, 0.9])
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        for pick in picks:
            assert set(TUNED).issubset(pick)

    def test_best_and_worst_come_from_the_score_extremes(self):
        frame = self.scoped_frame(
            [0.1, 0.3, 0.5, 0.7, 0.9], epochs_list=[10, 20, 30, 40, 50]
        )
        picks = {
            p["label"]: p
            for p in substudy_configurations(
                frame, knowledge_base=self.KB, condition=self.COND
            )
        }
        assert picks["best_mrr"]["source_score"] == pytest.approx(0.9)
        assert picks["best_mrr"]["epochs"] == 50
        assert picks["worst_mrr"]["source_score"] == pytest.approx(0.1)
        assert picks["worst_mrr"]["epochs"] == 10

    def test_median_is_the_upper_middle_for_odd_counts(self):
        frame = self.scoped_frame(
            [0.1, 0.3, 0.5, 0.7, 0.9], epochs_list=[10, 20, 30, 40, 50]
        )
        picks = {
            p["label"]: p
            for p in substudy_configurations(
                frame, knowledge_base=self.KB, condition=self.COND
            )
        }
        assert picks["median_mrr"]["source_score"] == pytest.approx(0.5)
        assert picks["median_mrr"]["epochs"] == 30

    def test_median_takes_the_upper_middle_for_even_counts(self):
        """B4: n//2 on a 0-indexed ascending frame is the upper median."""
        frame = self.scoped_frame([0.1, 0.3, 0.7, 0.9])
        picks = {
            p["label"]: p
            for p in substudy_configurations(
                frame, knowledge_base=self.KB, condition=self.COND
            )
        }
        assert picks["median_mrr"]["source_score"] == pytest.approx(0.7)

    def test_selection_is_invariant_to_input_row_order(self):
        ascending = self.scoped_frame(
            [0.1, 0.3, 0.5, 0.7, 0.9], epochs_list=[10, 20, 30, 40, 50]
        )
        descending = ascending.iloc[::-1].reset_index(drop=True)
        first = substudy_configurations(
            ascending, knowledge_base=self.KB, condition=self.COND
        )
        second = substudy_configurations(
            descending, knowledge_base=self.KB, condition=self.COND
        )
        assert first == second

    def test_extreme_overrides_learning_rate_only(self):
        frame = self.scoped_frame(
            [0.1, 0.9], batch_sizes=[32, 256], epochs_list=[25, 88]
        )
        picks = {
            p["label"]: p
            for p in substudy_configurations(
                frame,
                knowledge_base=self.KB,
                condition=self.COND,
                extreme_learning_rate=0.3,
            )
        }
        best, extreme = picks["best_mrr"], picks["extreme_learning_rate"]
        assert extreme["learning_rate"] == pytest.approx(0.3)
        assert extreme["batch_size"] == best["batch_size"]
        assert extreme["epochs"] == best["epochs"]

    def test_extreme_has_no_source_score(self):
        """It was never trialled, so no MRR may be attributed to it."""
        frame = self.scoped_frame([0.1, 0.9])
        picks = {
            p["label"]: p
            for p in substudy_configurations(
                frame, knowledge_base=self.KB, condition=self.COND
            )
        }
        assert picks["extreme_learning_rate"]["source_score"] is None

    def test_extreme_learning_rate_is_configurable(self):
        frame = self.scoped_frame([0.1, 0.9])
        picks = substudy_configurations(
            frame,
            knowledge_base=self.KB,
            condition=self.COND,
            extreme_learning_rate=0.001,
        )
        assert picks[-1]["learning_rate"] == pytest.approx(0.001)

    def test_mutating_extreme_does_not_alias_the_best_configuration(self):
        """`dict(picks[0])` must be a copy, not a view."""
        frame = self.scoped_frame([0.1, 0.9], learning_rates=[0.02, 0.05])
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        assert picks[0]["learning_rate"] == pytest.approx(0.05)
        assert picks[0]["source_score"] == pytest.approx(0.9)

    def test_excludes_failed_trials(self):
        frame = self.scoped_frame(
            [0.1, 0.5, None],
            epochs_list=[10, 20, 30],
            failed_flags=[False, False, True],
        )
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        assert all(p["epochs"] != 30 for p in picks)

    def test_excludes_null_scores(self):
        frame = self.scoped_frame([0.1, 0.5, None], epochs_list=[10, 20, 30])
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        assert all(p["epochs"] != 30 for p in picks)

    def test_ignores_other_conditions_and_knowledge_bases(self):
        frame = trials_to_frame(
            make_record(
                [0.5], condition=self.COND, knowledge_base=self.KB,
                epochs_list=[40],
            )
            + make_record(
                [0.99], condition="TransE", knowledge_base=self.KB,
                epochs_list=[99],
            )
            + make_record(
                [0.99], condition=self.COND, knowledge_base="mutagenesis",
                epochs_list=[98],
            )
        )
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        assert {p["epochs"] for p in picks} == {40}

    def test_pools_seeds_when_drawing_configurations(self):
        """The sub-study draws from the whole trial record for the pair."""
        frame = trials_to_frame(
            make_record(
                [0.2], condition=self.COND, knowledge_base=self.KB,
                seed=1, epochs_list=[30],
            )
            + make_record(
                [0.8], condition=self.COND, knowledge_base=self.KB,
                seed=2, epochs_list=[70],
            )
        )
        picks = {
            p["label"]: p
            for p in substudy_configurations(
                frame, knowledge_base=self.KB, condition=self.COND
            )
        }
        assert picks["best_mrr"]["epochs"] == 70
        assert picks["worst_mrr"]["epochs"] == 30

    # @pytest.mark.xfail(
    #     strict=True,
    #     reason="B3: one usable trial yields three identically-labelled "
    #     "duplicates, wasting 15 of the 20 sub-study runs",
    # )
    def test_single_trial_does_not_produce_duplicate_arms(self):
        frame = self.scoped_frame([0.5])
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND, scoped_guard=3
        )
        assert len(picks) == 0

    def test_single_trial_currently_collapses_three_arms(self):
        frame = self.scoped_frame([0.5])
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        assert len(picks) == 4
        assert (
            picks[0]["source_score"]
            == picks[1]["source_score"]
            == picks[2]["source_score"]
        )

    def test_two_trials_collapse_median_into_best(self):
        """n=2 => n//2 == 1 == -1, so median is the best trial."""
        frame = self.scoped_frame([0.2, 0.8])
        picks = {
            p["label"]: p
            for p in substudy_configurations(
                frame, knowledge_base=self.KB, condition=self.COND
            )
        }
        assert picks["median_mrr"]["source_score"] == pytest.approx(0.8)
        assert picks["median_mrr"]["source_score"] == picks["best_mrr"][
            "source_score"
        ]

    # @pytest.mark.xfail(
    #     strict=True,
    #     reason="B3: 0.3 is the search-space upper bound, so the best trial "
    #     "may already sit there and the fourth arm silently vanishes",
    # )
    def test_extreme_differs_from_best_when_best_is_at_the_bound(self):
        frame = self.scoped_frame([0.1, 0.9], learning_rates=[0.01, 0.3])
        picks = substudy_configurations(
            frame,
            knowledge_base=self.KB,
            condition=self.COND,
            extreme_learning_rate=0.3,
        )
        best = {k: picks[0][k] for k in TUNED}
        extreme = {k: picks[3][k] for k in TUNED}
        assert best != extreme

    def test_configurations_are_json_serialisable(self):
        frame = self.scoped_frame([0.1, 0.5, 0.9])
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        json.dumps(picks, default=str)

    def test_learning_rates_stay_inside_the_search_space(self):
        """Every drawn configuration must be trainable by DICE."""
        frame = self.scoped_frame([0.1, 0.5, 0.9])
        picks = substudy_configurations(
            frame, knowledge_base=self.KB, condition=self.COND
        )
        for pick in picks:
            assert 0.001 <= float(pick["learning_rate"]) <= 0.3

######################################
# Integration: the RQ2 pipeline to end
######################################

@staticmethod
def suite() -> list[Trial]:
    trials: list[Trial] = []
    for kb in ("semantic_bible", "vicodi", "carcinogenesis", "mutagenesis"):
        for condition in ("Keci", "TransE", "ComplEx"):
            for seed in (1, 2, 3, 4, 5):
                base = 0.2 if condition == "Keci" else 0.4
                spread = 0.05 if condition == "Keci" else 0.30
                scores = [
                    base + spread * (i / 5) for i in range(6)
                ]
                trials += make_record(
                    scores,
                    condition=condition,
                    knowledge_base=kb,
                    seed=seed,
                )
    return trials

def test_end_to_end_selection_and_draw():
    frame = trials_to_frame(suite())
    effects = make_main_effects(
        [
            ("Keci", "semantic_bible", 0.01),
            ("TransE", "vicodi", 0.05),
            ("ComplEx", "carcinogenesis", -0.02),
            ("TransE", "mutagenesis", 0.44),
        ]
    )
    target = select_substudy_target(frame, effects)
    assert target is not None
    assert target.knowledge_base == "mutagenesis"
    assert target.condition in {"TransE", "ComplEx"}

    picks = substudy_configurations(
        frame,
        knowledge_base=target.knowledge_base,
        condition=target.condition,
    )
    assert len(picks) == 4
    assert len({p["label"] for p in picks}) == 4

def test_run_count_matches_the_pre_registered_twenty():
    frame = trials_to_frame(suite())
    effects = make_main_effects([("TransE", "mutagenesis", 0.44)])
    target = select_substudy_target(frame, effects)
    picks = substudy_configurations(
        frame,
        knowledge_base=target.knowledge_base,
        condition=target.condition,
    )
    n_seeds = 5
    assert len(picks) * n_seeds == 20

def test_observational_and_experimental_parts_share_one_record():
    """Both RQ2 halves must read the same frame without mutating it."""
    frame = trials_to_frame(suite())
    before = frame.copy(deep=True)
    marginal_relationships(frame)
    selection_stability(frame)
    effects = make_main_effects([("TransE", "mutagenesis", 0.44)])
    target = select_substudy_target(frame, effects)
    substudy_configurations(
        frame,
        knowledge_base=target.knowledge_base,
        condition=target.condition,
    )
    pd.testing.assert_frame_equal(frame, before)

def test_stability_covers_every_condition_kb_pair():
    frame = trials_to_frame(suite())
    results = selection_stability(frame)
    assert len(results) == 4 * 3
    assert all(r.n_seeds == 5 for r in results)
    assert all(r.verdict != "indeterminate" for r in results)

def test_marginal_relationships_cover_every_pair_and_parameter():
    frame = trials_to_frame(suite())
    result = marginal_relationships(frame)
    assert len(result) == 4 * 3 * len(TUNED)