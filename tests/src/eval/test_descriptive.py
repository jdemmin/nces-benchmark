# tests/eval/test_descriptive.py
"""Tests for the descriptive layer.

These assert *no p-values anywhere* and that every summary degrades to
None-filled records rather than raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.descriptive import (
    EXTENSION_RATIO_BANDS,
    MECHANISM_OUTCOMES,
    SMALL_CELL,
    ExtensionSizeSummary,
    TrendSummary,
    _ols_slope,
    _safe_mean,
    breakdown,
    complexity_trend,
    dissociation_check,
    extension_size_summary,
    mechanism,
)


class FakeDesign:
    """Minimal stand-in for PairedDesign.

    ``collapse`` mirrors the real contract as inferred from usage: the
    per-problem mean of the paired difference, indexed by problem_id.
    """

    def __init__(self, frame: pd.DataFrame, seeds=(0, 1)):
        self.frame = frame
        self.seeds = list(seeds)

    @property
    def n_problems(self) -> int:
        return int(self.frame["problem_id"].nunique())

    def collapse(self, outcome: str) -> pd.Series:
        column = f"d_{outcome}"
        if column not in self.frame:
            treated = self.frame[f"{outcome}_treated"]
            control = self.frame[f"{outcome}_control"]
            differences = treated - control
            return differences.groupby(self.frame["problem_id"]).mean()
        return (
            self.frame.groupby("problem_id")[column].mean().dropna()
        )


def make_frame(
    n_problems: int = 6,
    seeds: tuple[int, ...] = (0, 1),
    *,
    treated_precision: float = 0.5,
    treated_recall: float = 0.8,
    control_precision: float = 0.5,
    control_recall: float = 0.4,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(11)
    for problem in range(n_problems):
        for seed in seeds:
            rows.append(
                {
                    "problem_id": f"p{problem:02d}",
                    "seed": seed,
                    "depth": 1 + problem % 4,
                    "extension_ratio": problem / max(n_problems - 1, 1),
                    "target_extension_size": 10 + problem,
                    "precision_treated": treated_precision,
                    "precision_control": control_precision,
                    "recall_treated": treated_recall,
                    "recall_control": control_recall,
                    "hypothesis_extension_size_treated": 20.0 + problem,
                    "hypothesis_extension_size_control": 10.0 + problem,
                    "d_abl": 0.1 * (problem - 2) + 0.01 * seed,
                    "d_precision": treated_precision - control_precision,
                    "d_recall": treated_recall - control_recall,
                    "d_hypothesis_extension_size": 10.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame["noise"] = rng.normal(size=len(frame))
    return frame


@pytest.fixture()
def design() -> FakeDesign:
    return FakeDesign(make_frame())


# --------------------------------------------------------------------------
# mechanism
# --------------------------------------------------------------------------


def test_mechanism_covers_declared_outcomes(design):
    rows = mechanism(design)
    assert [row.outcome for row in rows] == list(MECHANISM_OUTCOMES)


def test_mechanism_skips_outcomes_missing_either_arm(design):
    design.frame = design.frame.drop(columns=["precision_control"])
    outcomes = [row.outcome for row in mechanism(design)]
    assert "precision" not in outcomes
    assert "recall" in outcomes


def test_mechanism_detects_breadth_pattern(design):
    """Recall-heavy, precision-flat is the documented primary reading."""
    by_outcome = {row.outcome: row for row in mechanism(design)}
    assert by_outcome["precision"].mean_difference == pytest.approx(0.0)
    assert by_outcome["recall"].mean_difference > 0
    assert by_outcome["recall"].wins == design.n_problems
    assert by_outcome["recall"].losses == 0


def test_mechanism_win_loss_tie_partitions_problems(design):
    for row in mechanism(design):
        assert row.wins + row.losses + row.ties == design.n_problems


def test_mechanism_ties_counted_not_dropped():
    frame = make_frame(treated_recall=0.4, control_recall=0.4)
    rows = {row.outcome: row for row in mechanism(FakeDesign(frame))}
    recall = rows["recall"]
    assert recall.ties == 6
    assert (recall.wins, recall.losses) == (0, 0)


def test_mechanism_row_to_dict_shape(design):
    payload = mechanism(design)[0].to_dict()
    assert set(payload) == {
        "outcome",
        "mean_treated",
        "mean_control",
        "mean_difference",
        "hodges_lehmann",
        "win_loss_tie",
    }
    assert len(payload["win_loss_tie"]) == 3
    assert not any("p" == key or "pval" in key for key in payload)


def test_mechanism_handles_empty_collapse():
    frame = make_frame(n_problems=0)
    rows = mechanism(FakeDesign(frame))
    assert all(row.mean_difference is None for row in rows)
    assert all(row.hodges_lehmann is None for row in rows)


# --------------------------------------------------------------------------
# extension_size_summary
# --------------------------------------------------------------------------


def test_extension_summary_missing_columns_degrades(design):
    design.frame = design.frame.drop(columns=["target_extension_size"])
    summary = extension_size_summary(design)
    assert summary == ExtensionSizeSummary(None, None, None, None, 0, 6)


def test_extension_summary_reports_both_ratio_families(design):
    summary = extension_size_summary(design)
    assert summary.n_with_target_size == 6
    assert summary.n_problems == 6
    for value in (
        summary.ratio_of_means_treated,
        summary.ratio_of_means_control,
        summary.mean_of_ratios_treated,
        summary.mean_of_ratios_control,
    ):
        assert value is not None and np.isfinite(value)


def test_extension_summary_families_diverge_under_skew():
    """The two families exist because they disagree; assert they do."""
    frame = make_frame(n_problems=3)
    frame.loc[frame["problem_id"] == "p00", "target_extension_size"] = 1
    frame.loc[
        frame["problem_id"] == "p00",
        "hypothesis_extension_size_treated",
    ] = 500.0
    summary = extension_size_summary(FakeDesign(frame))
    assert summary.mean_of_ratios_treated > summary.ratio_of_means_treated


def test_extension_summary_excludes_zero_target_problems():
    frame = make_frame(n_problems=4)
    frame.loc[frame["problem_id"] == "p00", "target_extension_size"] = 0
    summary = extension_size_summary(FakeDesign(frame))
    assert summary.n_with_target_size == 3
    assert summary.n_problems == 4


def test_extension_summary_all_targets_zero():
    frame = make_frame(n_problems=3)
    frame["target_extension_size"] = 0
    summary = extension_size_summary(FakeDesign(frame))
    assert summary.n_with_target_size == 0
    assert summary.ratio_of_means_treated is None
    assert summary.mean_of_ratios_treated is None


def test_extension_summary_negative_target_treated_as_invalid():
    frame = make_frame(n_problems=3)
    frame.loc[frame["problem_id"] == "p01", "target_extension_size"] = -5
    summary = extension_size_summary(FakeDesign(frame))
    assert summary.n_with_target_size == 2


def test_extension_summary_families_cover_same_problems_under_nan():
    frame = make_frame(n_problems=4)
    frame.loc[
        frame["problem_id"] == "p02",
        "hypothesis_extension_size_treated",
    ] = np.nan
    summary = extension_size_summary(FakeDesign(frame))
    assert summary.mean_of_ratios_treated is not None
    assert np.isfinite(summary.mean_of_ratios_treated)


def test_extension_summary_to_dict_roundtrip(design):
    payload = extension_size_summary(design).to_dict()
    assert set(payload) == {
        "ratio_of_means_treated",
        "ratio_of_means_control",
        "mean_of_ratios_treated",
        "mean_of_ratios_control",
        "n_with_target_size",
        "n_problems",
    }


# --------------------------------------------------------------------------
# breakdown
# --------------------------------------------------------------------------


def test_breakdown_missing_column_returns_empty(design):
    assert breakdown(design, "nonexistent", by="depth").empty
    assert breakdown(design, "abl", by="nonexistent").empty


def test_breakdown_all_nan_returns_empty(design):
    design.frame["d_abl"] = np.nan
    assert breakdown(design, "abl", by="depth").empty


def test_breakdown_one_row_per_cell(design):
    table = breakdown(design, "abl", by="depth")
    assert len(table) == design.frame["depth"].nunique()
    assert table["n_problems"].sum() == design.n_problems


def test_breakdown_no_p_values(design):
    table = breakdown(design, "abl", by="depth")
    assert set(table.columns) == {
        "cell",
        "n_problems",
        "mean_difference",
        "median_difference",
        "wins",
        "losses",
        "ties",
        "small_cell",
    }


def test_breakdown_small_cells_reported_not_dropped():
    """Documented contract: no cell is silently omitted."""
    frame = make_frame(n_problems=7)
    frame.loc[frame["problem_id"] == "p06", "depth"] = 99
    table = breakdown(FakeDesign(frame), "abl", by="depth")
    flagged = table[table["small_cell"]]
    assert "99" in set(table["cell"])
    assert not flagged.empty
    assert (flagged["n_problems"] < SMALL_CELL).all()


def test_breakdown_small_cell_threshold_is_exclusive():
    frame = make_frame(n_problems=SMALL_CELL)
    frame["depth"] = 1
    table = breakdown(FakeDesign(frame), "abl", by="depth")
    assert table.loc[0, "n_problems"] == SMALL_CELL
    assert not bool(table.loc[0, "small_cell"])


def test_breakdown_extension_ratio_is_banded(design):
    table = breakdown(design, "abl", by="extension_ratio")
    assert len(table) <= len(EXTENSION_RATIO_BANDS) - 1
    assert all("," in cell for cell in table["cell"])


def test_breakdown_depth_is_not_banded(design):
    """Banding depth is explicitly rejected in the docstring."""
    table = breakdown(design, "abl", by="depth")
    assert all("," not in cell for cell in table["cell"])


def test_breakdown_wins_losses_ties_partition_cell(design):
    table = breakdown(design, "abl", by="depth")
    total = table["wins"] + table["losses"] + table["ties"]
    assert (total == table["n_problems"]).all()


def test_breakdown_median_differs_from_mean_under_outlier():
    frame = make_frame(n_problems=6)
    frame["depth"] = 1
    frame.loc[frame["problem_id"] == "p00", "d_abl"] = 100.0
    table = breakdown(FakeDesign(frame), "abl", by="depth")
    assert table.loc[0, "mean_difference"] > table.loc[0, "median_difference"]


def test_breakdown_drops_rows_with_missing_key_only(design):
    design.frame.loc[design.frame["problem_id"] == "p00", "depth"] = np.nan
    table = breakdown(design, "abl", by="depth")
    assert table["n_problems"].sum() == 5


def test_breakdown_sorted_by_cell(design):
    table = breakdown(design, "abl", by="depth")
    assert list(table["cell"]) == sorted(table["cell"])


# --------------------------------------------------------------------------
# _ols_slope
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0])),  # too few points
        (np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])),  # no spread
        (np.array([]), np.array([])),
    ],
)
def test_ols_slope_unidentified(x, y):
    assert _ols_slope(x, y) is None


def test_ols_slope_recovers_known_line():
    x = np.arange(6, dtype=float)
    slope, intercept = _ols_slope(x, 3.0 * x - 1.0)
    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# complexity_trend
# --------------------------------------------------------------------------


def test_complexity_trend_missing_predictor_notes_reason(design):
    summary = complexity_trend(design, "abl", predictor="absent")
    assert summary.slope is None
    assert summary.note is not None and "unavailable" in summary.note
    assert summary.confirmatory is False


def test_complexity_trend_single_level_unidentified():
    frame = make_frame(n_problems=6)
    frame["depth"] = 2
    summary = complexity_trend(
        FakeDesign(frame), "abl", n_resamples=10, seed=0
    )
    assert summary.slope is None
    assert summary.n_levels == 1
    assert "unidentified" in summary.note


def test_complexity_trend_never_confirmatory(design):
    """The whole module is descriptive; this must not drift."""
    for predictor in ("depth", "extension_ratio", "absent"):
        summary = complexity_trend(
            design, "abl", predictor=predictor, n_resamples=25
        )
        assert summary.confirmatory is False


def test_complexity_trend_recovers_planted_slope():
    frame = make_frame(n_problems=12)
    frame["depth"] = frame["problem_id"].str[1:].astype(int)
    frame["d_abl"] = 0.25 * frame["depth"]
    summary = complexity_trend(
        FakeDesign(frame), "abl", n_resamples=200, seed=3
    )
    assert summary.slope == pytest.approx(0.25, abs=1e-9)
    assert summary.n_problems == 12
    assert summary.ci_low is not None and summary.ci_high is not None
    assert summary.ci_low <= summary.slope <= summary.ci_high


def test_complexity_trend_is_seed_deterministic():
    frame = make_frame(n_problems=10)
    kwargs = {"n_resamples": 100, "seed": 7}
    first = complexity_trend(FakeDesign(frame), "abl", **kwargs)
    second = complexity_trend(FakeDesign(frame), "abl", **kwargs)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_complexity_trend_note_warns_about_confound(design):
    summary = complexity_trend(design, "abl", n_resamples=25)
    assert "extension_ratio" in summary.note
    assert "Descriptive only" in summary.note


def test_complexity_trend_to_dict_has_no_p_value(design):
    payload = complexity_trend(design, "abl", n_resamples=25).to_dict()
    assert "p_value" not in payload
    assert payload["confirmatory"] is False


def test_complexity_trend_survives_all_resamples_failing(monkeypatch):
    """CI stays None rather than raising when no resample identifies."""
    import src.eval.descriptive as module

    monkeypatch.setattr(module, "_ols_slope_calls", None, raising=False)
    frame = make_frame(n_problems=6)

    real = module._ols_slope
    state = {"first": True}

    def flaky(x, y):
        if state["first"]:
            state["first"] = False
            return real(x, y)
        return None

    monkeypatch.setattr(module, "_ols_slope", flaky)
    summary = complexity_trend(FakeDesign(frame), "abl", n_resamples=5)
    assert summary.slope is not None
    assert summary.ci_low is None and summary.ci_high is None


# --------------------------------------------------------------------------
# dissociation_check
# --------------------------------------------------------------------------


def trend(slope, low, high) -> TrendSummary:
    return TrendSummary(
        predictor="depth",
        slope=slope,
        intercept=0.0,
        ci_low=low,
        ci_high=high,
        n_levels=4,
        n_problems=20,
        confirmatory=False,
    )


@pytest.mark.parametrize(
    ("depth", "ratio", "expected"),
    [
        (trend(None, None, None), trend(0.1, 0.05, 0.2), "indeterminate"),
        (trend(0.1, 0.05, 0.2), trend(None, None, None), "indeterminate"),
        (trend(0.1, 0.05, 0.2), trend(0.1, -0.1, 0.3), "dissociated"),
        (trend(-0.1, -0.2, -0.05), trend(0.1, -0.1, 0.3), "dissociated"),
        (trend(0.1, 0.05, 0.2), trend(0.1, 0.05, 0.2), "confounded"),
        (trend(0.1, -0.1, 0.3), trend(0.1, -0.1, 0.3), "no trend"),
        (trend(0.1, -0.1, 0.3), trend(0.1, 0.05, 0.2), "no trend"),
    ],
)
def test_dissociation_check_verdicts(depth, ratio, expected):
    assert expected in dissociation_check(depth, ratio)


def test_dissociation_ci_touching_zero_is_not_a_signal():
    """CI whose bound is exactly zero must not count as a trend."""
    verdict = dissociation_check(
        trend(0.1, 0.0, 0.2), trend(0.1, -0.1, 0.3)
    )
    assert "no trend" in verdict


def test_dissociation_only_dissociated_licenses_complexity_claim():
    confounded = dissociation_check(
        trend(0.1, 0.05, 0.2), trend(0.1, 0.05, 0.2)
    )
    assert "not described as complexity-related" in confounded


# --------------------------------------------------------------------------
# _safe_mean
# --------------------------------------------------------------------------


def test_safe_mean_all_nan_is_none():
    assert _safe_mean(pd.Series([np.nan, np.nan])) is None


def test_safe_mean_empty_is_none():
    assert _safe_mean(pd.Series([], dtype=float)) is None


def test_safe_mean_ignores_nan():
    assert _safe_mean(pd.Series([1.0, np.nan, 3.0])) == pytest.approx(2.0)