# src/eval/descriptive.py
"""Mechanism layer, descriptive breakdowns, and the complexity trend.

Nothing in this module carries a p-value. These layers characterise *where*
and *how* an effect established elsewhere is concentrated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.eval.inference import hodges_lehmann
from src.eval.pairing import PairedDesign, resample_seed_clusters

logger = logging.getLogger(__name__)

MECHANISM_OUTCOMES = ("precision", "recall", "hypothesis_extension_size")
EXTENSION_RATIO_BANDS = (0.0, 0.05, 0.25, 0.75, 0.95, 1.0)
SMALL_CELL = 5


@dataclass(frozen=True)
class MechanismRow:
    outcome: str
    mean_treated: float | None
    mean_control: float | None
    mean_difference: float | None
    hodges_lehmann: float | None
    wins: int
    losses: int
    ties: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "mean_treated": self.mean_treated,
            "mean_control": self.mean_control,
            "mean_difference": self.mean_difference,
            "hodges_lehmann": self.hodges_lehmann,
            "win_loss_tie": [self.wins, self.losses, self.ties],
        }


@dataclass(frozen=True)
class ExtensionSizeSummary:
    """Two ratio families, because extension sizes are strongly skewed."""

    ratio_of_means_treated: float | None
    ratio_of_means_control: float | None
    mean_of_ratios_treated: float | None
    mean_of_ratios_control: float | None
    n_with_target_size: int
    n_problems: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio_of_means_treated": self.ratio_of_means_treated,
            "ratio_of_means_control": self.ratio_of_means_control,
            "mean_of_ratios_treated": self.mean_of_ratios_treated,
            "mean_of_ratios_control": self.mean_of_ratios_control,
            "n_with_target_size": self.n_with_target_size,
            "n_problems": self.n_problems,
        }


def mechanism(design: PairedDesign) -> list[MechanismRow]:
    """Characterise *how* embeddings alter hypotheses.

    The primary reading is the breadth hypothesis: a recall-heavy,
    precision-flat pattern indicates that one condition synthesises broader
    hypotheses rather than better ones.
    """
    rows: list[MechanismRow] = []
    for outcome in MECHANISM_OUTCOMES:
        treated = f"{outcome}_treated"
        control = f"{outcome}_control"
        if treated not in design.frame or control not in design.frame:
            continue
        collapsed = design.collapse(outcome)
        values = collapsed.to_numpy(dtype=float)
        rows.append(
            MechanismRow(
                outcome=outcome,
                mean_treated=_safe_mean(design.frame[treated]),
                mean_control=_safe_mean(design.frame[control]),
                mean_difference=(
                    float(values.mean()) if values.size else None
                ),
                hodges_lehmann=(
                    hodges_lehmann(values) if values.size else None
                ),
                wins=int(np.sum(values > 0)),
                losses=int(np.sum(values < 0)),
                ties=int(np.sum(values == 0)),
            )
        )
    return rows


def extension_size_summary(design: PairedDesign) -> ExtensionSizeSummary:
    """Compare |P| against |T| under both ratio families."""
    frame = design.frame
    needed = {
        "hypothesis_extension_size_treated",
        "hypothesis_extension_size_control",
        "target_extension_size_treated",
    }
    if not needed <= set(frame.columns):
        return ExtensionSizeSummary(
            None, None, None, None, 0, design.n_problems
        )

    per_problem = frame.groupby("problem_id").mean(numeric_only=True)
    target = per_problem["target_extension_size_treated"]
    valid = target > 0

    def ratio_of_means(column: str) -> float | None:
        subset = per_problem.loc[valid, column]
        denominator = target[valid].mean()
        if not len(subset) or not denominator:
            return None
        return float(subset.mean() / denominator)

    def mean_of_ratios(column: str) -> float | None:
        subset = per_problem.loc[valid, column] / target[valid]
        return float(subset.mean()) if len(subset) else None

    return ExtensionSizeSummary(
        ratio_of_means_treated=ratio_of_means(
            "hypothesis_extension_size_treated"
        ),
        ratio_of_means_control=ratio_of_means(
            "hypothesis_extension_size_control"
        ),
        mean_of_ratios_treated=mean_of_ratios(
            "hypothesis_extension_size_treated"
        ),
        mean_of_ratios_control=mean_of_ratios(
            "hypothesis_extension_size_control"
        ),
        n_with_target_size=int(valid.sum()),
        n_problems=design.n_problems,
    )


def breakdown(
    design: PairedDesign,
    outcome: str,
    *,
    by: str,
) -> pd.DataFrame:
    """Break the primary outcome down by a problem-level covariate.

    Reported as means and win/loss/tie triples with the per-cell problem
    count, and carrying no p-values: these cells exist to characterise where
    an effect is concentrated, not to test anything. No cell is silently
    omitted, so cells below ``SMALL_CELL`` appear with their count and a flag
    so the reader can discount them.
    """
    column = f"d_{outcome}"
    if column not in design.frame or by not in design.frame:
        return pd.DataFrame()

    usable = design.frame.dropna(subset=[column, by])
    if usable.empty:
        return pd.DataFrame()

    per_problem = (
        usable.groupby("problem_id")
        .agg(difference=(column, "mean"), key=(by, "first"))
        .dropna(subset=["key"])
    )
    if per_problem.empty:
        return pd.DataFrame()

    if by == "extension_ratio":
        # Banding is legitimate here in a way that banding `depth` is not:
        # the variable functions as a class-balance degeneracy indicator
        # rather than a contiguous predictor under test, and the edges are
        # fixed in advance.
        per_problem["key"] = pd.cut(
            per_problem["key"].astype(float),
            bins=list(EXTENSION_RATIO_BANDS),
            include_lowest=True,
            right=False,
        ).astype(str)

    rows: list[dict[str, Any]] = []
    for key, group in per_problem.groupby("key", observed=True):
        values = group["difference"].to_numpy(dtype=float)
        rows.append(
            {
                "cell": str(key),
                "n_problems": int(values.size),
                "mean_difference": float(values.mean()),
                "median_difference": float(np.median(values)),
                "wins": int(np.sum(values > 0)),
                "losses": int(np.sum(values < 0)),
                "ties": int(np.sum(values == 0)),
                "small_cell": bool(values.size < SMALL_CELL),
            }
        )
    return pd.DataFrame(rows).sort_values("cell").reset_index(drop=True)


@dataclass(frozen=True)
class TrendSummary:
    """OLS slope reported as a *description* of the trend plot."""

    predictor: str
    slope: float | None
    intercept: float | None
    ci_low: float | None
    ci_high: float | None
    n_levels: int
    n_problems: int
    confirmatory: bool
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictor": self.predictor,
            "slope": self.slope,
            "intercept": self.intercept,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_levels": self.n_levels,
            "n_problems": self.n_problems,
            "confirmatory": self.confirmatory,
            "note": self.note,
        }


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if x.size < 3 or np.unique(x).size < 2:
        return None
    design_matrix = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design_matrix, y, rcond=None)
    return float(coefficients[1]), float(coefficients[0])


def complexity_trend(
    design: PairedDesign,
    outcome: str,
    *,
    predictor: str = "depth",
    n_resamples: int = 2_000,
    seed: int = 0,
) -> TrendSummary:
    """Slope of the paired difference on a structural predictor.

    The predictor enters contiguously and is never binned: binning would
    discard information and make the bin boundaries an arbitrary analytical
    fork. The slope is a summary of the plot rather than a tested claim,
    because ``depth`` in an ALC population takes few distinct values.
    """
    column = f"d_{outcome}"
    if column not in design.frame or predictor not in design.frame:
        return TrendSummary(
            predictor=predictor,
            slope=None,
            intercept=None,
            ci_low=None,
            ci_high=None,
            n_levels=0,
            n_problems=0,
            confirmatory=False,
            note=f"predictor {predictor!r} unavailable",
        )

    usable = design.frame.dropna(subset=[column, predictor])
    per_problem = usable.groupby("problem_id").agg(
        difference=(column, "mean"), predictor=(predictor, "first")
    )
    x = per_problem["predictor"].to_numpy(dtype=float)
    y = per_problem["difference"].to_numpy(dtype=float)

    fit = _ols_slope(x, y)
    if fit is None:
        return TrendSummary(
            predictor=predictor,
            slope=None,
            intercept=None,
            ci_low=None,
            ci_high=None,
            n_levels=int(np.unique(x).size),
            n_problems=int(x.size),
            confirmatory=False,
            note="slope unidentified: too few distinct predictor levels",
        )

    slope, intercept = fit
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(n_resamples):
        resampled = resample_seed_clusters(usable, design.seeds, rng)
        grouped = resampled.groupby("problem_id").agg(
            difference=(column, "mean"), predictor=(predictor, "first")
        )
        candidate = _ols_slope(
            grouped["predictor"].to_numpy(dtype=float),
            grouped["difference"].to_numpy(dtype=float),
        )
        if candidate is not None:
            slopes.append(candidate[0])

    low = high = None
    if slopes:
        low, high = (float(v) for v in np.percentile(slopes, [2.5, 97.5]))

    return TrendSummary(
        predictor=predictor,
        slope=slope,
        intercept=intercept,
        ci_low=low,
        ci_high=high,
        n_levels=int(np.unique(x).size),
        n_problems=int(x.size),
        confirmatory=False,
        note=(
            "Descriptive only. Because deeper targets tend to have smaller "
            "extensions, and small extensions destabilise F1 and hence ABL, "
            "compare against the extension_ratio trend before describing any "
            "pattern as complexity-related."
        ),
    )


def dissociation_check(
    depth_trend: TrendSummary, ratio_trend: TrendSummary
) -> str:
    """Whether a depth trend survives the extension-size confound.

    A trend is described as complexity-related only when the two plots
    dissociate.
    """
    if depth_trend.slope is None or ratio_trend.slope is None:
        return "indeterminate: one or both slopes unidentified"
    depth_signal = (
        depth_trend.ci_low is not None
        and depth_trend.ci_high is not None
        and (depth_trend.ci_low > 0 or depth_trend.ci_high < 0)
    )
    ratio_signal = (
        ratio_trend.ci_low is not None
        and ratio_trend.ci_high is not None
        and (ratio_trend.ci_low > 0 or ratio_trend.ci_high < 0)
    )
    if depth_signal and not ratio_signal:
        return "dissociated: trend may be described as complexity-related"
    if depth_signal and ratio_signal:
        return (
            "confounded: both trends present, potential extension-size "
            "artifact; not described as complexity-related"
        )
    return "no trend at attainable precision"


def _safe_mean(series: pd.Series) -> float | None:
    values = series.dropna()
    return float(values.mean()) if len(values) else None