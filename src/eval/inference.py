# src/eval/inference.py
"""Estimation-focused inference: one estimator, one interval, one test.

With thirteen conditions, four knowledge bases and five seeds, any
per-contrast apparatus is multiplied by fifty-two, and the design's precision
is governed by the seed count rather than the problem count. The layer is
therefore deliberately lean.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.pairing import PairedDesign, resample_seed_clusters

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 10_000
BOOTSTRAP_ALPHA = 0.05
#: Zero handling is fixed at Pratt's rule and is not configurable.
ZERO_METHOD = "pratt"
#: Below this n the sign-flip null is enumerated exactly.
EXACT_SIGN_FLIP_MAX_N = 20
SIGN_FLIP_RESAMPLES = 10_000


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    excludes_zero: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "excludes_zero": self.excludes_zero,
        }


@dataclass(frozen=True)
class NonParametricResult:
    """Distribution-light agreement check on the sign and location."""

    p_value: float
    hodges_lehmann: float
    wins: int
    losses: int
    ties: int
    sign_test_p: float | None
    n_nonzero: int
    exact_null: bool
    degenerate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_value": self.p_value,
            "hodges_lehmann": self.hodges_lehmann,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "sign_test_p": self.sign_test_p,
            "n_nonzero": self.n_nonzero,
            "exact_null": self.exact_null,
            "degenerate": self.degenerate,
        }


@dataclass(frozen=True)
class SeedSpread:
    """The design's report on run-to-run variability."""

    per_seed_means: dict[int, float]
    seed_sd: float | None
    problem_sd: float | None

    @property
    def dominant_source(self) -> str:
        if self.seed_sd is None or self.problem_sd is None:
            return "unknown"
        if self.problem_sd > self.seed_sd:
            return "concepts"
        if self.seed_sd > self.problem_sd:
            return "runs"
        return "comparable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_seed_means": self.per_seed_means,
            "seed_sd": self.seed_sd,
            "problem_sd": self.problem_sd,
            "dominant_source": self.dominant_source,
        }


@dataclass(frozen=True)
class Diagnostics:
    """Reported with the same prominence as the interval."""

    zero_fraction: float
    identical_hypothesis_fraction: float | None
    wins: int
    losses: int
    ties: int
    empty_rate_treated: float | None
    empty_rate_control: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "zero_fraction": self.zero_fraction,
            "identical_hypothesis_fraction": self.identical_hypothesis_fraction,
            "win_loss_tie": [self.wins, self.losses, self.ties],
            "empty_rate_treated": self.empty_rate_treated,
            "empty_rate_control": self.empty_rate_control,
        }


@dataclass
class ContrastResult:
    condition: str
    knowledge_base: str
    outcome: str
    substituted_primary: bool
    n_problems: int
    n_observations: int
    n_seeds: int
    estimate: float | None
    interval: Interval | None
    test: NonParametricResult | None
    diagnostics: Diagnostics | None
    seed_spread: SeedSpread | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "knowledge_base": self.knowledge_base,
            "outcome": self.outcome,
            "substituted_primary": self.substituted_primary,
            "n_problems": self.n_problems,
            "n_observations": self.n_observations,
            "n_seeds": self.n_seeds,
            "estimate": self.estimate,
            "interval": self.interval.to_dict() if self.interval else None,
            "test": self.test.to_dict() if self.test else None,
            "diagnostics": (
                self.diagnostics.to_dict() if self.diagnostics else None
            ),
            "seed_spread": (
                self.seed_spread.to_dict() if self.seed_spread else None
            ),
            "notes": self.notes,
        }


# -- point estimate and interval ------------------------------------------


def point_estimate(design: PairedDesign, outcome: str) -> float | None:
    collapsed = design.collapse(outcome)
    return float(collapsed.mean()) if len(collapsed) else None


def cluster_bootstrap_interval(
    design: PairedDesign,
    outcome: str,
    *,
    n_resamples: int = N_BOOTSTRAP,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> Interval | None:
    """Percentile cluster bootstrap with the seed as the resampling unit.

    A percentile bootstrap over five clusters has limited resolution in its
    tails; the interval is an indication of magnitude and sign, not a precise
    bound.
    """
    column = f"d_{outcome}"
    try:
        usable = design.frame.dropna(subset=[column])
    except KeyError:
        logger.warning("Column %s not found in design frame", column)
        return None
    if usable.empty or len(design.seeds) < 2:
        return None

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        resampled = resample_seed_clusters(usable, design.seeds, rng)
        collapsed = resampled.groupby("problem_id")[column].mean()
        if len(collapsed):
            estimates.append(float(collapsed.mean()))

    if not estimates:
        return None
    low, high = np.percentile(
        estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    return Interval(
        low=float(low),
        high=float(high),
        excludes_zero=bool(low > 0 or high < 0),
    )


# -- non-parametric layer -------------------------------------------------


def hodges_lehmann(values: np.ndarray) -> float:
    """Median of pairwise Walsh averages; the Wilcoxon-matched estimator.

    Explicitly *not* the median of the differences.
    """
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return float(values[0])
    walsh = (values[:, None] + values[None, :]) / 2.0
    upper = walsh[np.triu_indices_from(walsh, k=0)]
    return float(np.median(upper))


def _pratt_signed_rank(values: np.ndarray) -> float:
    """Signed-rank statistic under Pratt's rule.

    Zeros are ranked with the rest and only then dropped from the signed sum.
    Wilcoxon's original rule discards zeros *before* ranking, which inflates
    the ranks of the survivors and overstates the non-zero evidence.
    """
    ranks = stats.rankdata(np.abs(values))
    return float(np.sum(ranks[values > 0]) - np.sum(ranks[values < 0]))


def _sign_flip_p_value(values: np.ndarray, *, seed: int = 0) -> tuple[float, bool]:
    """Two-sided p-value from a sign-flip null.

    The asymptotic normal approximation is unreliable with a large zero mass
    and few non-zero observations, so the null is enumerated exactly for
    ``n <= 20`` and sampled by Monte Carlo above.
    """
    nonzero = values[values != 0]
    n = nonzero.size
    if n == 0:
        return 1.0, True

    observed = abs(_pratt_signed_rank(values))
    zeros = values[values == 0]

    if n <= EXACT_SIGN_FLIP_MAX_N:
        at_least = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            candidate = np.concatenate([nonzero * np.asarray(signs), zeros])
            total += 1
            if abs(_pratt_signed_rank(candidate)) >= observed - 1e-12:
                at_least += 1
        return at_least / total, True

    rng = np.random.default_rng(seed)
    at_least = 1  # observed value counts toward the null
    for _ in range(SIGN_FLIP_RESAMPLES):
        signs = rng.choice((-1.0, 1.0), size=n)
        candidate = np.concatenate([nonzero * signs, zeros])
        if abs(_pratt_signed_rank(candidate)) >= observed - 1e-12:
            at_least += 1
    return at_least / (SIGN_FLIP_RESAMPLES + 1), False


def non_parametric_test(collapsed: pd.Series) -> NonParametricResult:
    """Wilcoxon signed-rank with Pratt zero handling and a sign-flip null."""
    values = collapsed.to_numpy(dtype=float)
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    ties = int(np.sum(values == 0))

    if values.size == 0 or ties == values.size:
        # Degenerate contrast: no test is meaningful.
        return NonParametricResult(
            p_value=1.0,
            hodges_lehmann=0.0,
            wins=0,
            losses=0,
            ties=int(values.size),
            sign_test_p=None,
            n_nonzero=0,
            exact_null=True,
            degenerate=True,
        )

    p_value, exact = _sign_flip_p_value(values)
    sign_p = float(
        stats.binomtest(wins, wins + losses, 0.5).pvalue
    ) if wins + losses else None

    return NonParametricResult(
        p_value=float(p_value),
        hodges_lehmann=hodges_lehmann(values),
        wins=wins,
        losses=losses,
        ties=ties,
        sign_test_p=sign_p,
        n_nonzero=wins + losses,
        exact_null=exact,
        degenerate=False,
    )


# -- diagnostics ----------------------------------------------------------


def diagnostics(design: PairedDesign, outcome: str) -> Diagnostics:
    collapsed = design.collapse(outcome)
    values = collapsed.to_numpy(dtype=float)
    n = max(values.size, 1)

    identical = None
    frame = design.frame
    if {"hypothesis_treated", "hypothesis_control"} <= set(frame.columns):
        matches = frame["hypothesis_treated"] == frame["hypothesis_control"]
        identical = float(matches.mean()) if len(frame) else None

    def empty_rate(column: str) -> float | None:
        if column not in frame:
            return None
        series = frame[column].dropna()
        return float((series == 0).mean()) if len(series) else None

    return Diagnostics(
        zero_fraction=float(np.sum(values == 0) / n),
        identical_hypothesis_fraction=identical,
        wins=int(np.sum(values > 0)),
        losses=int(np.sum(values < 0)),
        ties=int(np.sum(values == 0)),
        empty_rate_treated=empty_rate("hypothesis_extension_size_treated"),
        empty_rate_control=empty_rate("hypothesis_extension_size_control"),
    )


def seed_spread(design: PairedDesign, outcome: str) -> SeedSpread:
    """Report seed-level and per-problem spread side by side.

    When per-problem spread substantially exceeds seed-level spread, the
    embedding effect varies more across concepts than across training runs,
    and a single run is comparatively trustworthy. The reverse pattern implies
    conclusions from few seeds are fragile.
    """
    per_seed = design.seed_means(outcome)
    collapsed = design.collapse(outcome)
    return SeedSpread(
        per_seed_means={int(k): float(v) for k, v in per_seed.items()},
        seed_sd=(
            float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None
        ),
        problem_sd=(
            float(collapsed.std(ddof=1)) if len(collapsed) > 1 else None
        ),
    )


def analyse_contrast(
    design: PairedDesign,
    outcome: str,
    *,
    bootstrap_seed: int = 0,
) -> ContrastResult:
    """Run the full inferential layer for one contrast, guarding each block."""
    result = ContrastResult(
        condition=design.condition,
        knowledge_base=design.knowledge_base,
        outcome=outcome,
        substituted_primary=design.substituted_primary,
        n_problems=design.n_problems,
        n_observations=design.n_observations,
        n_seeds=len(design.seeds),
        estimate=None,
        interval=None,
        test=None,
        diagnostics=None,
        seed_spread=None,
        notes=list(design.notes),
    )

    collapsed = design.collapse(outcome)
    if not len(collapsed):
        result.notes.append(f"Outcome {outcome!r} unavailable; analysis skipped.")
        return result

    result.estimate = float(collapsed.mean())

    for label, block in (
        ("interval", lambda: cluster_bootstrap_interval(
            design, outcome, seed=bootstrap_seed
        )),
        ("test", lambda: non_parametric_test(collapsed)),
        ("diagnostics", lambda: diagnostics(design, outcome)),
        ("seed_spread", lambda: seed_spread(design, outcome)),
    ):
        try:
            setattr(result, label, block())
        except Exception as exc:  # each layer is guarded independently
            logger.warning(
                "Block %s failed for %s/%s: %s",
                label,
                design.condition,
                design.knowledge_base,
                exc,
            )
            result.notes.append(f"Block {label!r} unavailable: {exc}")
    return result