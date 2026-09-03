#src/benchmarking/inference.py
"""Inferential analysis of the embedding-condition contrast.

Implements the pre-specified outcome hierarchy: a confirmatory primary
estimate on ``lift`` differences, a confirmatory complexity trend, a
nonparametric robustness layer, and a Benjamini-Hochberg-screened
exploratory grid.

The atomic observation is one ``(learning problem, seed)`` pair and the
response is always a paired difference

    d_ij = m_dice_ij - m_random_ij

which cancels the target concept, examples, split, knowledge base, NCES
architecture and seed shared by the two embedding conditions.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.lp import stable_id
from src.data.results import LearningProblemResult, SingleRunResult
from src.random_utils import seed_everything

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pre-specified outcome hierarchy
# --------------------------------------------------------------------------

#: The single confirmatory primary outcome.
PRIMARY_OUTCOME: str = "lift"

#: Agreement check against the primary. Not a second primary.
ROBUSTNESS_OUTCOME: str = "f1_score"

#: Descriptive mechanism outcomes. Reported, never tested confirmatorily.
MECHANISM_OUTCOMES: tuple[str, ...] = (
    "precision",
    "recall",
    "hypothesis_extension_size",
)

#: Screening only. Enters the Benjamini-Hochberg family.
EXPLORATORY_OUTCOMES: tuple[str, ...] = (
    "mcc",
    "accuracy",
    "jaccard",
    "semantic_equivalence",
)

#: Complexity fields used to bucket the exploratory grid.
EXPLORATORY_BUCKETINGS: tuple[str, ...] = (
    "depth",
    "expressivity",
    "extension_ratio",
)

#: Predictor for the confirmatory secondary. Contiguous; never binned.
TREND_PREDICTOR: str = "dl_length"

#: Covariate for the trend confounding check.
TREND_COVARIATE: str = "extension_ratio"

#: False discovery rate for the exploratory family.
EXPLORATORY_Q: float = 0.10

#: Bootstrap resamples, clustered on seed.
BOOTSTRAP_RESAMPLES: int = 10_000


class InferenceError(RuntimeError):
    """Raised when the paired design cannot be assembled at all."""


# --------------------------------------------------------------------------
# Paired observations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedObservation:
    """One ``(learning problem, seed)`` pair for a single outcome.

    ``difference`` is ``dice - random``. ``problem_id`` is the join key:
    learning-problem generation and splitting are seed-independent, so a
    problem's ``id`` identifies the same target concept in every seed.
    """

    problem_id: str
    seed: int
    outcome: str
    dice_value: float
    random_value: float
    dl_length: int | None = None
    depth: int | None = None
    expressivity: str | None = None
    extension_ratio: float | None = None

    @property
    def difference(self) -> float:
        return self.dice_value - self.random_value

    @property
    def is_zero(self) -> bool:
        """Both conditions produced the same score for this problem."""
        return self.dice_value == self.random_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "seed": self.seed,
            "outcome": self.outcome,
            "dice_value": self.dice_value,
            "random_value": self.random_value,
            "difference": self.difference,
            "dl_length": self.dl_length,
            "depth": self.depth,
            "expressivity": self.expressivity,
            "extension_ratio": self.extension_ratio,
        }


@dataclass(frozen=True)
class PairedHypotheses:
    """Both conditions' hypotheses for one problem, side by side.

    Yields the zero-difference count directly and enables per-concept
    inspection: rank target concepts by mean lift difference, read the
    top and the bottom.
    """

    problem_id: str
    seed: int
    target_concept: str
    dice_hypothesis: str
    random_hypothesis: str

    @property
    def identical(self) -> bool:
        return self.dice_hypothesis == self.random_hypothesis

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "seed": self.seed,
            "target_concept": self.target_concept,
            "dice_hypothesis": self.dice_hypothesis,
            "random_hypothesis": self.random_hypothesis,
            "identical": self.identical,
        }


# --------------------------------------------------------------------------
# Extraction from run results
# --------------------------------------------------------------------------


def _complexity_of(result: LearningProblemResult) -> Mapping[str, Any]:
    """Best-effort access to a truncated problem's complexity object."""
    problem = result.learning_problem
    complexity = getattr(problem, "complexity", None)
    if complexity is None:
        return {}
    if isinstance(complexity, Mapping):
        return complexity
    to_dict = getattr(complexity, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {
        name: getattr(complexity, name)
        for name in (
            "dl_length",
            "depth",
            "expressivity",
            "extension_ratio",
            "atomic_baseline_f1",
            "redundant",
        )
        if hasattr(complexity, name)
    }


def _problem_id(result: LearningProblemResult) -> str:
    """The stable join key, with a deterministic fallback.

    ``LearningProblem.id`` is the documented identifier. If a report
    predates it, fall back to a hash of the target concept so pairing is
    still possible and still deterministic.
    """
    problem = result.learning_problem
    identifier = getattr(problem, "id", None)
    if identifier:
        return str(identifier)
    logger.warning(
        "Learning problem has no id; falling back to hash of target" \
        "concept and examples."
    )
    id = stable_id(problem)
    return id


def _outcome_value(
    result: LearningProblemResult, outcome: str
) -> float | None:
    """Read one outcome off a per-problem result, or ``None``."""
    if result.error is not None:
        return None
    
    if outcome == "hypothesis_extension_size":
        extension = result.hypothesis_extension
        if extension is None:
            return None
        return float(extension.positive)

    if outcome == "empty_hypothesis":
        extension = result.hypothesis_extension
        if extension is None:
            return None
        return 1.0 if extension.positive == 0 else 0.0

    if outcome == "runtime_seconds":
        return None if result.runtime is None else float(result.runtime)

    if outcome == "mcc":
        matrix = _confusion_matrix(result)
        if matrix is None:
            return None
        return matthews_correlation_coefficient(matrix)

    metrics = result.metrics
    if metrics is None:
        return None
    if not hasattr(metrics, outcome):
        return None
    value = getattr(metrics, outcome)
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _index_by_problem(
    results: Sequence[LearningProblemResult],
) -> dict[str, LearningProblemResult]:
    indexed: dict[str, LearningProblemResult] = {}
    for result in results:
        identifier = _problem_id(result)
        if identifier in indexed:
            logger.warning(
                "Duplicate learning-problem id %s in one condition; "
                "keeping the first occurrence.",
                identifier,
            )
            continue
        indexed[identifier] = result
    return indexed


@dataclass(frozen=True)
class PairedDesign:
    """Every paired observation for one knowledge base, all outcomes."""

    knowledge_base: str
    observations: dict[str, list[PairedObservation]]
    hypotheses: list[PairedHypotheses]
    seeds: tuple[int, ...]
    problem_ids: tuple[str, ...]
    unpaired_problem_ids: tuple[str, ...]
    redundant_problem_ids: tuple[str, ...]
    error_counts: dict[str, int]

    target_extension_sizes: dict[str, int] = field(default_factory=dict)
    # |U|, the individual count of the knowledge base, when recoverable.
    universe_size: int | None = None

    @property
    def n_observations(self) -> int:
        primary = self.observations.get(PRIMARY_OUTCOME, [])
        return len(primary)

    def for_outcome(self, outcome: str) -> list[PairedObservation]:
        return self.observations.get(outcome, [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_base": self.knowledge_base,
            "n_seeds": len(self.seeds),
            "seeds": list(self.seeds),
            "n_problems": len(self.problem_ids),
            "n_observations": self.n_observations,
            "unpaired_problem_ids": list(self.unpaired_problem_ids),
            "redundant_problem_ids": list(self.redundant_problem_ids),
            "error_counts": dict(self.error_counts),
        }


def build_paired_design(
    runs: Mapping[int, SingleRunResult],
    *,
    outcomes: Iterable[str] | None = None,
) -> PairedDesign:
    """Assemble the paired design from ``{seed: SingleRunResult}``.

    A problem contributes an observation for a given outcome only when
    both conditions produced a usable value for it. Problems that NCES
    failed on are excluded from aggregates, as the failure-handling
    contract requires, but counted in ``error_counts``.
    """
    if not runs:
        raise InferenceError("No benchmark runs supplied.")

    requested = (
        tuple(outcomes)
        if outcomes is not None
        else (
            PRIMARY_OUTCOME,
            ROBUSTNESS_OUTCOME,
            *MECHANISM_OUTCOMES,
            *EXPLORATORY_OUTCOMES,
            "empty_hypothesis",
            "runtime_seconds",
        )
    )

    knowledge_bases = {run.knowledge_base for run in runs.values()}
    if len(knowledge_bases) > 1:
        raise InferenceError(
            "build_paired_design expects one knowledge base per call, got "
            f"{sorted(knowledge_bases)}."
        )
    knowledge_base = next(iter(knowledge_bases))

    observations: dict[str, list[PairedObservation]] = {
        outcome: [] for outcome in requested
    }
    hypotheses: list[PairedHypotheses] = []
    error_counts: dict[str, int] = {"dice": 0, "random": 0}
    target_extension_sizes: dict[str, int] = {}
    paired_ids: set[str] = set()
    unpaired_ids: set[str] = set()
    used_seeds: list[int] = []

    for seed in sorted(runs):
        run = runs[seed]
        dice = run.dice_embedding_result
        random_ = run.random_embedding_result
        if dice is None or random_ is None:
            logger.warning(
                "Seed %s of %s lacks one embedding condition; the paired "
                "contrast is undefined and the seed is skipped.",
                seed,
                knowledge_base,
            )
            continue

        used_seeds.append(seed)
        dice_by_id = _index_by_problem(dice.learning_problem_results)
        random_by_id = _index_by_problem(random_.learning_problem_results)

        error_counts["dice"] += sum(
            1 for r in dice.learning_problem_results if r.error is not None
        )
        error_counts["random"] += sum(
            1 for r in random_.learning_problem_results if r.error is not None
        )

        unpaired_ids |= set(dice_by_id) ^ set(random_by_id)
        redundant_ids = {identifier for identifier in set(dice_by_id) & set(random_by_id) if dice_by_id[identifier].learning_problem.complexity.hardness.redundant}
        unpaired_ids |= redundant_ids

        for identifier in sorted(set(dice_by_id) & set(random_by_id)):
            dice_result = dice_by_id[identifier]
            random_result = random_by_id[identifier]
            complexity = _complexity_of(dice_result)
            target_size = _target_extension_size(dice_result, random_result)
            if target_size is not None:
                previous = target_extension_sizes.get(identifier)
                if previous is not None and previous != target_size:
                    logger.warning(
                        "Target extension size for %s differs across seeds "
                        "(%s vs %s). The target concept should be "
                        "seed-invariant; keeping the first value.",
                        identifier,
                        previous,
                        target_size,
                    )
                else:
                    target_extension_sizes[identifier] = target_size
            dl_length = complexity.get("dl_length")
            depth = complexity.get("depth")
            expressivity = complexity.get("expressivity")
            hardness = complexity.get("hardness")
            extension_ratio = hardness.get("extension_ratio") if hardness is not None else None

            contributed = False
            for outcome in requested:
                dice_value = _outcome_value(dice_result, outcome)
                random_value = _outcome_value(random_result, outcome)
                if dice_value is None or random_value is None:
                    continue
                observations[outcome].append(
                    PairedObservation(
                        problem_id=identifier,
                        seed=seed,
                        outcome=outcome,
                        dice_value=dice_value,
                        random_value=random_value,
                        dl_length=(
                            None if dl_length is None else int(dl_length)
                        ),
                        depth=None if depth is None else int(depth),
                        expressivity=(
                            None if expressivity is None else str(expressivity)
                        ),
                        extension_ratio=(
                            None
                            if extension_ratio is None
                            else float(extension_ratio)
                        ),
                    )
                )
                contributed = True
            if contributed:
                paired_ids.add(identifier)
            hypotheses.append(
                PairedHypotheses(
                    problem_id=identifier,
                    seed=seed,
                    target_concept=str(
                        getattr(
                            dice_result.learning_problem, "target_concept", ""
                        )
                    ),
                    dice_hypothesis=dice_result.hypothesis,
                    random_hypothesis=random_result.hypothesis,
                )
            )
    if not used_seeds:
        raise InferenceError(
            f"No seed of {knowledge_base} carries both embedding conditions."
        )
    return PairedDesign(
        knowledge_base=knowledge_base,
        observations=observations,
        hypotheses=hypotheses,
        seeds=tuple(used_seeds),
        problem_ids=tuple(sorted(paired_ids)),
        unpaired_problem_ids=tuple(sorted(unpaired_ids)),
        redundant_problem_ids=tuple(sorted(redundant_ids)),
        error_counts=error_counts,
        target_extension_sizes=target_extension_sizes,
    )


def _target_extension_size(
    dice_result: LearningProblemResult,
    random_result: LearningProblemResult,
) -> int | None:
    """|T| for one problem, preferring the direct field.

    ``TargetExtensionStructure.total`` is ``positive + negative``. Whether
    that equals |T| or |U| depends on what ``negative`` counts, so the
    ``positive`` field is used directly -- it is unambiguously the number
    of individuals in the target extension.
    """
    for result in (dice_result, random_result):
        extension = result.target_extension
        if extension is None:
            continue
        positive = getattr(extension, "positive", None)
        if positive is not None:
            return int(positive)
    return None


# --------------------------------------------------------------------------
# Crossed mixed model: d ~ 1 + (1|seed) + (1|problem)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MixedModelResult:
    """Estimate of the mean embedding effect and its variance components."""

    outcome: str
    model: str
    n_observations: int
    n_seeds: int
    n_problems: int
    beta_0: float
    beta_0_se: float
    ci95: tuple[float, float]
    p_value: float
    df: float
    df_method: str
    var_seed: float
    var_problem: float
    var_residual: float
    bootstrap_ci95: tuple[float, float] | None
    bootstrap_resamples: int
    agreement: str
    converged: bool
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "model": self.model,
            "n_observations": self.n_observations,
            "n_seeds": self.n_seeds,
            "n_problems": self.n_problems,
            "beta_0": self.beta_0,
            "beta_0_se": self.beta_0_se,
            "ci95": list(self.ci95),
            "p": self.p_value,
            "df": self.df,
            "df_method": self.df_method,
            "var_seed": self.var_seed,
            "var_problem": self.var_problem,
            "var_residual": self.var_residual,
            "bootstrap_ci95": (
                None if self.bootstrap_ci95 is None else list(self.bootstrap_ci95)
            ),
            "bootstrap_resamples": self.bootstrap_resamples,
            "agreement": self.agreement,
            "converged": self.converged,
            "note": self.note,
        }


def _profile_reml(
    y,  # numpy array of differences
    seed_index,
    problem_index,
    n_seeds: int,
    n_problems: int,
):
    """REML fit of an intercept-only crossed two-way random-effects model.

    Optimizes the profiled REML criterion over the two variance ratios
    ``(var_seed / var_residual, var_problem / var_residual)``. The
    marginal covariance is

        V = var_seed * Z_s Z_s' + var_problem * Z_p Z_p' + var_res * I

    which is dense but tiny at benchmark scale (a few hundred rows), so
    it is formed and factorized directly rather than exploiting sparsity.
    """
    import numpy as np
    from scipy.optimize import minimize

    n = y.shape[0]
    z_seed = np.zeros((n, n_seeds))
    z_seed[np.arange(n), seed_index] = 1.0
    z_problem = np.zeros((n, n_problems))
    z_problem[np.arange(n), problem_index] = 1.0

    g_seed = z_seed @ z_seed.T
    g_problem = z_problem @ z_problem.T
    identity = np.eye(n)
    design = np.ones((n, 1))

    def neg_reml(log_ratios: np.ndarray) -> float:
        ratio_seed, ratio_problem = np.exp(log_ratios)
        covariance = ratio_seed * g_seed + ratio_problem * g_problem + identity
        # Attempt Cholesky decomposition to check positive
        # definiteness of the covariance matrix.
        try:
            cho = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return 1e12
        log_det = 2.0 * np.sum(np.log(np.diag(cho)))
        # The generalized least-squares estimate is
        # beta = (X' V^-1 X)^-1 X' V^-1 y
        solved_y = np.linalg.solve(covariance, y)
        solved_x = np.linalg.solve(covariance, design)
        xtx = float(design.T @ solved_x)
        if xtx <= 0.0:
            return 1e12
        beta = float(design.T @ solved_y) / xtx
        residual = y - design.flatten() * beta
        quadratic = float(residual @ np.linalg.solve(covariance, residual))
        sigma2 = quadratic / (n - 1)
        if sigma2 <= 0.0:
            return 1e12
        # negative log-likelihood of the REML criterion.
        return 0.5 * (
            log_det
            + math.log(xtx)
            + (n - 1) * (math.log(sigma2) + 1.0 + math.log(2.0 * math.pi))
        )

    best = None
    for start in ((-2.0, -2.0), (0.0, 0.0), (-4.0, 0.0), (0.0, -4.0)):
        try:
            # Start the optimization from the current initial guess.
            # Use the Nelder-Mead simplex algorithm to minimize the
            # negative REML log-likelihood. Derivative-free
            # optimization is used. The initial guess is given by
            # the current start tuple.
            candidate = minimize(
                neg_reml,
                x0=np.array(start),
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
            )
        except Exception:  # pragma: no cover - optimizer robustness
            continue
        if best is None or candidate.fun < best.fun:
            best = candidate

    if best is None:
        raise InferenceError("REML optimization failed to start.")

    ratio_seed, ratio_problem = np.exp(best.x)
    covariance = ratio_seed * g_seed + ratio_problem * g_problem + identity
    solved_y = np.linalg.solve(covariance, y)
    solved_x = np.linalg.solve(covariance, design)
    xtx = float(design.T @ solved_x)
    beta = float(design.T @ solved_y) / xtx
    residual = y - design.flatten() * beta
    quadratic = float(residual @ np.linalg.solve(covariance, residual))
    var_residual = quadratic / (n - 1)
    # The standard error of the fixed effect estimate is computed using the
    # delta method, taking into account the variance of the residual and
    # the design matrix. It accounts for the uncertainty in the residual
    # variance when estimating the standard error of the fixed effect.
    standard_error = math.sqrt(var_residual / xtx)

    return {
        "beta_0": beta,
        "se": standard_error,
        "var_seed": ratio_seed * var_residual,
        "var_problem": ratio_problem * var_residual,
        "var_residual": var_residual,
        "converged": bool(best.success),
    }


def _satterthwaite_df(
    n_observations: int, n_seeds: int, n_problems: int
) -> float:
    """Conservative denominator degrees of freedom.

    With five seeds the seed variance is poorly estimated, so the naive
    ``n - 1`` is far too optimistic. Between-seed contrasts carry at most
    ``n_seeds - 1`` degrees of freedom, and that is the binding
    constraint on an intercept shared by every observation in a run.
    """
    # thanks to past me for poor choices
    return float(max(1, min(n_seeds - 1, n_problems - 1)))


def _cluster_bootstrap_ci(
    observations: Sequence[PairedObservation],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float] | None:
    """Percentile bootstrap over seeds, the resampling unit.

    Lift differences are bounded and spike at zero, so the model-based
    interval is not to be trusted on its own. Observations from the
    same seed might be correlated. Clustering on seed preserves the
    within-run dependence that the seed random intercept is there to
    absorb. 
    """
    import numpy as np

    by_seed: dict[int, list[float]] = {}
    for observation in observations:
        by_seed.setdefault(observation.seed, []).append(observation.difference)
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        return None

    rng = np.random.default_rng(seed)
    seed_everything(seed)
    clusters = [np.asarray(by_seed[s], dtype=float) for s in seeds]
    means = np.empty(resamples, dtype=float)
    count = len(clusters)
    for index in range(resamples):
        picks = rng.integers(0, count, size=count)
        pooled = np.concatenate([clusters[p] for p in picks])
        means[index] = pooled.mean()
    # Useful because the distribution of the mean under clustered
    # resampling may not be symmetric or normal.
    # Remember that lift differences are bounded and spike at zero, so the
    # percentile bootstrap is more reliable than a normal approximation.
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def fit_mixed_model(
    observations: Sequence[PairedObservation],
    *,
    outcome: str | None = None,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = 0,
) -> MixedModelResult:
    """Fit ``d ~ 1 + (1|seed) + (1|problem)`` by REML.

    ``beta_0`` is the headline estimate: the mean embedding effect.
    ``var_seed`` versus ``var_problem`` is itself a result -- it answers
    whether the advantage varies more across concepts than across
    training runs, and therefore whether a single run is trustworthy.
    """
    import numpy as np
    from scipy import stats

    if not observations:
        raise InferenceError("Cannot fit a model on zero observations.")

    label = outcome or observations[0].outcome
    differences = np.asarray(
        [o.difference for o in observations], dtype=float
    )
    seed_labels = sorted({o.seed for o in observations})
    problem_labels = sorted({o.problem_id for o in observations})
    seed_lookup = {s: i for i, s in enumerate(seed_labels)}
    problem_lookup = {p: i for i, p in enumerate(problem_labels)}
    seed_index = np.asarray(
        [seed_lookup[o.seed] for o in observations], dtype=int
    )
    problem_index = np.asarray(
        [problem_lookup[o.problem_id] for o in observations], dtype=int
    )

    n_observations = differences.shape[0]
    n_seeds = len(seed_labels)
    n_problems = len(problem_labels)
    note: str | None = None

    if n_seeds < 2 or n_problems < 2 or n_observations < 3:
        # Not enough structure for two variance components. Fall back to
        # a plain paired t-interval and say so.
        mean = float(differences.mean())
        standard_error = (
            float(differences.std(ddof=1) / math.sqrt(n_observations))
            if n_observations > 1
            else float("nan")
        )
        fit = {
            "beta_0": mean,
            "se": standard_error,
            "var_seed": 0.0,
            "var_problem": 0.0,
            "var_residual": (
                float(differences.var(ddof=1)) if n_observations > 1 else 0.0
            ),
            "converged": False,
        }
        degrees = float(max(1, n_observations - 1))
        df_method = "naive-t"
        note = (
            "Insufficient crossing to identify two variance components; "
            "reduced to a paired t-interval."
        )
    else:
        fit = _profile_reml(
            differences, seed_index, problem_index, n_seeds, n_problems
        )
        degrees = _satterthwaite_df(n_observations, n_seeds, n_problems)
        df_method = "satterthwaite-conservative"

    beta_0 = fit["beta_0"]
    standard_error = fit["se"]
    if not math.isfinite(standard_error) or standard_error <= 0.0:
        p_value = float("nan")
        ci = (float("nan"), float("nan"))
    else:
        t_statistic = beta_0 / standard_error
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=degrees))
        critical = float(stats.t.ppf(0.975, df=degrees))
        ci = (
            beta_0 - critical * standard_error,
            beta_0 + critical * standard_error,
        )

    bootstrap_ci = _cluster_bootstrap_ci(
        observations, resamples=bootstrap_resamples, seed=bootstrap_seed
    )

    # Determine agreement between model-based and bootstrap confidence intervals.
    # Bootstrap conclusion is more robust when available.
    agreement = "bootstrap-unavailable"
    if bootstrap_ci is not None and math.isfinite(ci[0]):
        model_excludes_zero = ci[0] > 0.0 or ci[1] < 0.0
        bootstrap_excludes_zero = (
            bootstrap_ci[0] > 0.0 or bootstrap_ci[1] < 0.0
        )
        agreement = (
            "agree"
            if model_excludes_zero == bootstrap_excludes_zero
            else "disagree-trust-bootstrap"
        )

    return MixedModelResult(
        outcome=label,
        model="d ~ 1 + (1|seed) + (1|problem)",
        n_observations=n_observations,
        n_seeds=n_seeds,
        n_problems=n_problems,
        beta_0=beta_0,
        beta_0_se=standard_error,
        ci95=ci,
        p_value=p_value,
        df=degrees,
        df_method=df_method,
        var_seed=fit["var_seed"],
        var_problem=fit["var_problem"],
        var_residual=fit["var_residual"],
        bootstrap_ci95=bootstrap_ci,
        bootstrap_resamples=(0 if bootstrap_ci is None else bootstrap_resamples),
        agreement=agreement,
        converged=bool(fit["converged"]),
        note=note,
    )


# --------------------------------------------------------------------------
# Confirmatory secondary: complexity trend
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendResult:
    """Change in the embedding effect per additional DL token."""

    outcome: str
    predictor: str
    n_observations: int
    beta_0: float
    beta_1: float
    beta_1_se: float
    ci95: tuple[float, float]
    p_value: float
    df: float
    df_method: str
    predictor_mean: float
    covariate: str | None
    covariate_adjusted_beta_1: float | None
    covariate_adjusted_ci95: tuple[float, float] | None
    covariate_adjusted_p: float | None
    survives_covariate_adjustment: bool | None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "predictor": f"{self.predictor}_centered",
            "n_observations": self.n_observations,
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_1_se": self.beta_1_se,
            "ci95": list(self.ci95),
            "p": self.p_value,
            "df": self.df,
            "df_method": self.df_method,
            "predictor_mean": self.predictor_mean,
            "covariate": self.covariate,
            "covariate_adjusted_beta_1": self.covariate_adjusted_beta_1,
            "covariate_adjusted_ci95": (
                None
                if self.covariate_adjusted_ci95 is None
                else list(self.covariate_adjusted_ci95)
            ),
            "covariate_adjusted_p": self.covariate_adjusted_p,
            "survives_covariate_adjustment": (
                self.survives_covariate_adjustment
            ),
            "note": self.note,
        }


def _cluster_robust_ols(design, y, cluster_index):
    """OLS with a cluster-robust (CR1) covariance, clustered on seed.

    A pragmatic stand-in for the full crossed model with a slope: the
    point estimate is identical to the mixed-model fixed effect under
    balance, and clustering on seed keeps the standard error honest about
    within-run dependence.
    """
    import numpy as np

    n, k = design.shape
    xtx_inverse = np.linalg.pinv(design.T @ design)
    beta = xtx_inverse @ design.T @ y
    residual = y - design @ beta

    meat = np.zeros((k, k))
    clusters = np.unique(cluster_index)
    for cluster in clusters:
        mask = cluster_index == cluster
        contribution = design[mask].T @ residual[mask]
        meat += np.outer(contribution, contribution)

    n_clusters = clusters.shape[0]
    if n_clusters > 1 and n > k:
        adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    else:
        adjustment = 1.0
    covariance = adjustment * (xtx_inverse @ meat @ xtx_inverse)
    return beta, covariance, n_clusters


def fit_complexity_trend(
    observations: Sequence[PairedObservation],
    *,
    predictor: str = TREND_PREDICTOR,
    covariate: str | None = TREND_COVARIATE,
) -> TrendResult:
    """Regress paired differences on centered ``dl_length``.

    ``dl_length`` is contiguous and is never binned: binning discards
    information, costs power, and makes the bin boundaries an arbitrary
    forking path. Centering is required rather than cosmetic -- without
    it ``beta_0`` is the effect at length zero, which does not exist.

    The covariate check addresses a real confound: longer concepts tend
    to have smaller extensions, and small extensions destabilize F1.
    """
    import numpy as np
    from scipy import stats

    usable = [
        o
        for o in observations
        if getattr(o, predictor, None) is not None
    ]
    if len(usable) < 3:
        raise InferenceError(
            f"Fewer than three observations carry {predictor}; the trend "
            "is not estimable."
        )

    label = usable[0].outcome
    differences = np.asarray([o.difference for o in usable], dtype=float)
    predictor_values = np.asarray(
        [float(getattr(o, predictor)) for o in usable], dtype=float
    )
    predictor_mean = float(predictor_values.mean())
    centered = predictor_values - predictor_mean

    seed_labels = sorted({o.seed for o in usable})
    seed_lookup = {s: i for i, s in enumerate(seed_labels)}
    cluster_index = np.asarray(
        [seed_lookup[o.seed] for o in usable], dtype=int
    )

    note: str | None = None
    if np.allclose(centered, 0.0):
        note = (
            f"{predictor} is constant across paired problems; the slope "
            "is not identified."
        )
        return TrendResult(
            outcome=label,
            predictor=predictor,
            n_observations=len(usable),
            beta_0=float(differences.mean()),
            beta_1=float("nan"),
            beta_1_se=float("nan"),
            ci95=(float("nan"), float("nan")),
            p_value=float("nan"),
            df=float("nan"),
            df_method="cluster-robust",
            predictor_mean=predictor_mean,
            covariate=covariate,
            covariate_adjusted_beta_1=None,
            covariate_adjusted_ci95=None,
            covariate_adjusted_p=None,
            survives_covariate_adjustment=None,
            note=note,
        )

    design = np.column_stack([np.ones_like(centered), centered])
    beta, covariance, n_clusters = _cluster_robust_ols(
        design, differences, cluster_index
    )
    standard_error = math.sqrt(max(covariance[1, 1], 0.0))
    degrees = float(max(1, n_clusters - 1))
    if standard_error > 0.0:
        t_statistic = beta[1] / standard_error
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=degrees))
        critical = float(stats.t.ppf(0.975, df=degrees))
        ci = (
            beta[1] - critical * standard_error,
            beta[1] + critical * standard_error,
        )
    else:
        p_value = float("nan")
        ci = (float("nan"), float("nan"))

    adjusted_beta_1: float | None = None
    adjusted_ci: tuple[float, float] | None = None
    adjusted_p: float | None = None
    survives: bool | None = None

    if covariate is not None:
        with_covariate = [
            o for o in usable if getattr(o, covariate, None) is not None
        ]
        if len(with_covariate) >= 4:
            adjusted_differences = np.asarray(
                [o.difference for o in with_covariate], dtype=float
            )
            adjusted_predictor = np.asarray(
                [float(getattr(o, predictor)) for o in with_covariate],
                dtype=float,
            )
            covariate_values = np.asarray(
                [float(getattr(o, covariate)) for o in with_covariate],
                dtype=float,
            )
            adjusted_design = np.column_stack(
                [
                    np.ones_like(adjusted_predictor),
                    adjusted_predictor - adjusted_predictor.mean(),
                    covariate_values - covariate_values.mean(),
                ]
            )
            adjusted_clusters = np.asarray(
                [seed_lookup[o.seed] for o in with_covariate], dtype=int
            )
            (
                adjusted_beta,
                adjusted_covariance,
                adjusted_n_clusters,
            ) = _cluster_robust_ols(
                adjusted_design, adjusted_differences, adjusted_clusters
            )
            adjusted_se = math.sqrt(max(adjusted_covariance[1, 1], 0.0))
            adjusted_beta_1 = float(adjusted_beta[1])
            adjusted_df = float(max(1, adjusted_n_clusters - 1))
            if adjusted_se > 0.0:
                adjusted_p = float(
                    2.0
                    * stats.t.sf(
                        abs(adjusted_beta_1 / adjusted_se), df=adjusted_df
                    )
                )
                adjusted_critical = float(stats.t.ppf(0.975, df=adjusted_df))
                adjusted_ci = (
                    adjusted_beta_1 - adjusted_critical * adjusted_se,
                    adjusted_beta_1 + adjusted_critical * adjusted_se,
                )
                survives = bool(
                    (adjusted_ci[0] > 0.0 or adjusted_ci[1] < 0.0)
                    and math.copysign(1.0, adjusted_beta_1)
                    == math.copysign(1.0, beta[1])
                )
        else:
            note = (
                f"Too few observations carry {covariate} for the "
                "confounding check."
            )

    return TrendResult(
        outcome=label,
        predictor=predictor,
        n_observations=len(usable),
        beta_0=float(beta[0]),
        beta_1=float(beta[1]),
        beta_1_se=standard_error,
        ci95=ci,
        p_value=p_value,
        df=degrees,
        df_method="cluster-robust",
        predictor_mean=predictor_mean,
        covariate=covariate,
        covariate_adjusted_beta_1=adjusted_beta_1,
        covariate_adjusted_ci95=adjusted_ci,
        covariate_adjusted_p=adjusted_p,
        survives_covariate_adjustment=survives,
        note=note,
    )


# --------------------------------------------------------------------------
# Robustness: nonparametric layer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessResult:
    """Wilcoxon signed-rank with Pratt zeros, plus a sign test.

    ``n_zero / n_total`` is reported as a result, not as diagnostics: it
    is the fraction of problems where the embedding condition changed
    nothing about the hypothesis. A large fraction makes "embeddings
    rarely alter NCES's output, but when they do it is usually an
    improvement" the sharper finding.
    """

    outcome: str
    test: str
    zero_method: str
    null: str
    n_total: int
    n_zero: int
    zero_fraction: float
    wins: int
    losses: int
    ties: int
    statistic: float
    p_value: float
    hodges_lehmann: float
    sign_test_p: float
    identical_hypothesis_fraction: float | None
    mean_difference: float
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "test": self.test,
            "zero_method": self.zero_method,
            "null": self.null,
            "n_total": self.n_total,
            "n_zero": self.n_zero,
            "zero_fraction": self.zero_fraction,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "statistic": self.statistic,
            "p": self.p_value,
            "hodges_lehmann": self.hodges_lehmann,
            "sign_test_p": self.sign_test_p,
            "identical_hypothesis_fraction": (
                self.identical_hypothesis_fraction
            ),
            "mean_difference": self.mean_difference,
            "note": self.note,
        }


def collapse_over_seeds(
    observations: Sequence[PairedObservation],
) -> dict[str, float]:
    """Average each problem's difference over seeds.

    Reduces per-problem noise by roughly ``sqrt(n_seeds)`` and makes the
    across-problem test's independence assumption defensible.
    """
    grouped: dict[str, list[float]] = {}
    for observation in observations:
        grouped.setdefault(observation.problem_id, []).append(
            observation.difference
        )
    return {
        problem_id: sum(values) / len(values)
        for problem_id, values in grouped.items()
    }


def _hodges_lehmann(values) -> float:
    """Median of pairwise Walsh averages -- the estimate matching Wilcoxon.

    Reported alongside the mean, because a significant Wilcoxon with a
    near-zero mean is possible when the zero mass is large and the
    nonzero tails are asymmetric.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    walsh = (array[:, None] + array[None, :]) / 2.0
    upper = walsh[np.triu_indices(array.size)]
    return float(np.median(upper))


def _permutation_signflip_p(values, statistic_fn, *, resamples, seed):
    """Exact-in-spirit null by sign-flipping the observed differences."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    observed = statistic_fn(array)
    n = array.size
    rng = np.random.default_rng(seed)

    if n <= 20:
        # Enumerate all 2^n sign assignments.
        count = 0
        total = 1 << n
        for mask in range(total):
            signs = np.array(
                [1.0 if (mask >> bit) & 1 == 0 else -1.0 for bit in range(n)]
            )
            if statistic_fn(array * signs) >= observed - 1e-12:
                count += 1
        return float(count / total), "exact-signflip"

    count = 0
    for _ in range(resamples):
        signs = rng.choice((-1.0, 1.0), size=n)
        if statistic_fn(array * signs) >= observed - 1e-12:
            count += 1
    return float((count + 1) / (resamples + 1)), "monte-carlo-signflip"

def run_robustness(
    observations: Sequence[PairedObservation],
    *,
    hypotheses: Sequence[PairedHypotheses] | None = None,
    permutation_resamples: int = 20_000,
    permutation_seed: int = 0,
) -> RobustnessResult:
    """Collapse over seeds, then test across problems.

    Pratt's zero handling is used deliberately. Ties at exactly zero are
    common here because NCES frequently synthesizes the identical
    hypothesis under both conditions; Wilcoxon's original rule discards
    zeros before ranking and thereby inflates the remaining ranks. Pratt
    ranks the zeros, then drops them from the sum.

    The null is a sign-flip permutation rather than the asymptotic
    normal approximation, which is unreliable with a large zero mass and
    few nonzero observations.
    """
    import numpy as np
    from scipy import stats

    if not observations:
        raise InferenceError("Cannot run robustness on zero observations.")

    label = observations[0].outcome
    collapsed = collapse_over_seeds(observations)
    problem_ids = sorted(collapsed)
    differences = np.asarray(
        [collapsed[p] for p in problem_ids], dtype=float
    )

    n_total = differences.size
    zero_mask = np.isclose(differences, 0.0, atol=1e-12)
    n_zero = int(zero_mask.sum())
    wins = int((differences > 1e-12).sum())
    losses = int((differences < -1e-12).sum())

    note: str | None = None
    statistic = float("nan")
    p_value = float("nan")
    null_label = "not-run"

    nonzero = differences[~zero_mask]
    if nonzero.size == 0:
        note = (
            "Every paired difference is exactly zero: the embedding "
            "condition never changed the score. No test is meaningful."
        )
        p_value = 1.0
        null_label = "degenerate"
    else:
        try:
            wilcoxon = stats.wilcoxon(
                differences,
                zero_method="pratt",
                alternative="two-sided",
                #method="auto",
            )
            statistic = float(wilcoxon.statistic)
        except (ValueError, TypeError):
            # Older/newer SciPy signatures differ on `mode`/`method`.
            try:
                wilcoxon = stats.wilcoxon(
                    differences,
                    zero_method="pratt",
                    alternative="two-sided",
                )
                statistic = float(wilcoxon.statistic)
            except ValueError as error:  # pragma: no cover
                note = f"Wilcoxon unavailable: {error}"

        def signed_rank_statistic(values) -> float:
            """Pratt-style signed-rank sum, absolute value.

            Ranks include the zeros (Pratt), which are then excluded from
            the sum. Taking the absolute value makes the statistic
            two-sided-monotone, which is what the sign-flip null needs.
            """
            ranks = stats.rankdata(np.abs(values))
            contributions = np.sign(values) * ranks
            return float(abs(contributions.sum()))

        p_value, null_label = _permutation_signflip_p(
            differences,
            signed_rank_statistic,
            resamples=permutation_resamples,
            seed=permutation_seed,
        )

    # Sign test on the nonzero differences: assumption-light check.
    n_nonzero = wins + losses
    if n_nonzero > 0:
        sign_test_p = float(
            stats.binomtest(wins, n=n_nonzero, p=0.5).pvalue
        )
    else:
        sign_test_p = 1.0

    identical_fraction: float | None = None
    if hypotheses:
        identical_fraction = float(
            sum(1 for h in hypotheses if h.identical) / len(hypotheses)
        )

    return RobustnessResult(
        outcome=label,
        test="wilcoxon_signed_rank",
        zero_method="pratt",
        null=null_label,
        n_total=n_total,
        n_zero=n_zero,
        zero_fraction=(0.0 if n_total == 0 else n_zero / n_total),
        wins=wins,
        losses=losses,
        ties=n_zero,
        statistic=statistic,
        p_value=p_value,
        hodges_lehmann=_hodges_lehmann(differences),
        sign_test_p=sign_test_p,
        identical_hypothesis_fraction=identical_fraction,
        mean_difference=float(differences.mean()) if n_total else float("nan"),
        note=note,
    )

# --------------------------------------------------------------------------
# Mechanism layer (descriptive only)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanismSummary:
    """Descriptive characterization of how embeddings alter hypotheses.

    Deliberately carries no p-value. Precision, recall and |P| exist to
    interpret the primary result -- for instance to test the "broader
    hypotheses" reading of a recall-heavy, precision-flat pattern --
    not to generate additional claims.
    """

    outcome: str
    n_problems: int
    mean_dice: float
    mean_random: float
    mean_difference: float
    hodges_lehmann: float
    wins: int
    losses: int
    ties: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "n_problems": self.n_problems,
            "mean_dice": self.mean_dice,
            "mean_random": self.mean_random,
            "mean_difference": self.mean_difference,
            "hodges_lehmann": self.hodges_lehmann,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
        }


def summarize_mechanism(
    observations: Sequence[PairedObservation],
) -> MechanismSummary | None:
    if not observations:
        return None

    import numpy as np

    label = observations[0].outcome
    collapsed = collapse_over_seeds(observations)
    differences = np.asarray(list(collapsed.values()), dtype=float)
    dice_values = np.asarray([o.dice_value for o in observations], dtype=float)
    random_values = np.asarray(
        [o.random_value for o in observations], dtype=float
    )

    return MechanismSummary(
        outcome=label,
        n_problems=len(collapsed),
        mean_dice=float(dice_values.mean()),
        mean_random=float(random_values.mean()),
        mean_difference=float(differences.mean()),
        hodges_lehmann=_hodges_lehmann(differences),
        wins=int((differences > 1e-12).sum()),
        losses=int((differences < -1e-12).sum()),
        ties=int(np.isclose(differences, 0.0, atol=1e-12).sum()),
    )


@dataclass(frozen=True)
class ExtensionSizeSummary:
    """|P| against |T| -- the direct test of the breadth reading."""

    mean_hypothesis_size_dice: float
    mean_hypothesis_size_random: float
    mean_target_size: float
    n_problems: int
    n_problems_with_target_size: int
    dice_over_target_ratio: float
    random_over_target_ratio: float
    mean_per_problem_dice_ratio: float
    mean_per_problem_random_ratio: float
    empty_hypothesis_rate_dice: float
    empty_hypothesis_rate_random: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_hypothesis_size_dice": self.mean_hypothesis_size_dice,
            "mean_hypothesis_size_random": self.mean_hypothesis_size_random,
            "mean_target_size": self.mean_target_size,
            "n_problems": self.n_problems,
            "n_problems_with_target_size": self.n_problems_with_target_size,
            "dice_over_target_ratio": self.dice_over_target_ratio,
            "random_over_target_ratio": self.random_over_target_ratio,
            "mean_per_problem_dice_ratio": self.mean_per_problem_dice_ratio,
            "mean_per_problem_random_ratio": (
                self.mean_per_problem_random_ratio
            ),
            "empty_hypothesis_rate_dice": self.empty_hypothesis_rate_dice,
            "empty_hypothesis_rate_random": self.empty_hypothesis_rate_random,
        }

# --------------------------------------------------------------------------
# Exploratory grid with Benjamini-Hochberg screening
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExploratoryFinding:
    """One cell of the exploratory grid. Screening only."""

    outcome: str
    bucketing: str | None
    bucket: str | None
    n_problems: int
    mean_difference: float
    hodges_lehmann: float
    p_value: float
    p_adjusted: float | None = None
    discovery: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "bucketing": self.bucketing,
            "bucket": self.bucket,
            "n_problems": self.n_problems,
            "mean_difference": self.mean_difference,
            "hodges_lehmann": self.hodges_lehmann,
            "p": self.p_value,
            "p_adjusted": self.p_adjusted,
            "discovery": self.discovery,
            "role": "exploratory",
        }


def benjamini_hochberg(
    p_values: Sequence[float], *, q: float = EXPLORATORY_Q
) -> tuple[list[float], list[bool]]:
    """Step-up BH. Returns adjusted p-values and discovery flags.

    FDR rather than FWER: family-wise control over a grid this large is
    self-defeating when the goal is screening.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    indexed = sorted(
        (p, i) for i, p in enumerate(p_values) if not math.isnan(p)
    )
    if not indexed:
        return [float("nan")] * n, [False] * n

    m = len(indexed)
    adjusted = [float("nan")] * n
    running = 1.0
    for rank in range(m, 0, -1):
        p, original_index = indexed[rank - 1]
        candidate = min(1.0, p * m / rank)
        running = min(running, candidate)
        adjusted[original_index] = running

    discoveries = [
        (not math.isnan(adjusted[i])) and adjusted[i] <= q for i in range(n)
    ]
    return adjusted, discoveries


def _bucket_key(observation: PairedObservation, bucketing: str) -> str | None:
    value = getattr(observation, bucketing, None)
    if value is None:
        return None
    if bucketing == "extension_ratio":
        # The only place binning is legitimate: a degeneracy indicator,
        # not a contiguous predictor being tested for trend.
        ratio = float(value)
        if ratio < 0.05:
            return "[0.00,0.05)"
        if ratio < 0.25:
            return "[0.05,0.25)"
        if ratio < 0.75:
            return "[0.25,0.75)"
        if ratio < 0.95:
            return "[0.75,0.95)"
        return "[0.95,1.00]"
    return str(value)


def run_exploratory_grid(
    design: PairedDesign,
    *,
    outcomes: Sequence[str] = EXPLORATORY_OUTCOMES,
    bucketings: Sequence[str] = EXPLORATORY_BUCKETINGS,
    q: float = EXPLORATORY_Q,
    min_problems: int = 5,
    permutation_resamples: int = 5_000,
    permutation_seed: int = 0,
) -> list[ExploratoryFinding]:
    """Screen other metrics and other bucketings. Labeled exploratory.

    Semantic equivalence is handled as a proportion via McNemar rather
    than Wilcoxon, since it is a rate and not a continuous score.
    """
    import numpy as np
    from scipy import stats

    findings: list[ExploratoryFinding] = []

    for outcome in outcomes:
        observations = design.for_outcome(outcome)
        if not observations:
            continue

        cells: list[tuple[str | None, str | None, list[PairedObservation]]] = [
            (None, None, list(observations))
        ]
        for bucketing in bucketings:
            grouped: dict[str, list[PairedObservation]] = {}
            for observation in observations:
                key = _bucket_key(observation, bucketing)
                if key is None:
                    continue
                grouped.setdefault(key, []).append(observation)
            for key in sorted(grouped):
                cells.append((bucketing, key, grouped[key]))

        for bucketing, bucket, cell in cells:
            collapsed = collapse_over_seeds(cell)
            if len(collapsed) < min_problems:
                continue
            differences = np.asarray(list(collapsed.values()), dtype=float)

            if outcome == "semantic_equivalence":
                # A proportion: paired McNemar on the discordant pairs.
                dice_only = sum(
                    1
                    for o in cell
                    if o.dice_value > 0.5 and o.random_value <= 0.5
                )
                random_only = sum(
                    1
                    for o in cell
                    if o.random_value > 0.5 and o.dice_value <= 0.5
                )
                discordant = dice_only + random_only
                p_value = (
                    1.0
                    if discordant == 0
                    else float(
                        stats.binomtest(
                            dice_only, n=discordant, p=0.5
                        ).pvalue
                    )
                )
            elif np.allclose(differences, 0.0, atol=1e-12):
                p_value = 1.0
            else:

                def signed_rank_statistic(values) -> float:
                    ranks = stats.rankdata(np.abs(values))
                    return float(abs((np.sign(values) * ranks).sum()))

                p_value, _ = _permutation_signflip_p(
                    differences,
                    signed_rank_statistic,
                    resamples=permutation_resamples,
                    seed=permutation_seed,
                )

            findings.append(
                ExploratoryFinding(
                    outcome=outcome,
                    bucketing=bucketing,
                    bucket=bucket,
                    n_problems=len(collapsed),
                    mean_difference=float(differences.mean()),
                    hodges_lehmann=_hodges_lehmann(differences),
                    p_value=p_value,
                )
            )

    adjusted, discoveries = benjamini_hochberg(
        [f.p_value for f in findings], q=q
    )
    return [
        ExploratoryFinding(
            outcome=finding.outcome,
            bucketing=finding.bucketing,
            bucket=finding.bucket,
            n_problems=finding.n_problems,
            mean_difference=finding.mean_difference,
            hodges_lehmann=finding.hodges_lehmann,
            p_value=finding.p_value,
            p_adjusted=adjusted[index],
            discovery=discoveries[index],
        )
        for index, finding in enumerate(findings)
    ]


# --------------------------------------------------------------------------
# Matthews correlation coefficient
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionMatrix:
    """Extension-based confusion matrix for one hypothesis.

    Reconstructed from the fields a ``LearningProblemResult`` carries:

        TP = |P ∩ T|                    (metrics.intersection)
        FP = |P| - TP                   (hypothesis_extension.positive)
        FN = |T| - TP                   (target_extension.positive)
        TN = |U| - TP - FP - FN

    |U| comes from ``target_extension.total`` -- ``positive + negative``
    where ``negative`` counts the individuals outside the target
    extension, making the total the full individual count of the
    knowledge base. This is the same universe the ``accuracy`` metric is
    already scored over.
    """

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def universe_size(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    @property
    def is_consistent(self) -> bool:
        """No cell may be negative; a negative cell means bad inputs."""
        return all(
            cell >= 0
            for cell in (
                self.true_positive,
                self.false_positive,
                self.false_negative,
                self.true_negative,
            )
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
        }


def matthews_correlation_coefficient(matrix: ConfusionMatrix) -> float:
    r"""MCC in :math:`[-1, 1]`, computed in log space for stability.

    .. math::

        \mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}
        {\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}

    The denominator is a product of four marginals, each of which can
    reach the size of the knowledge base. On ``vicodi`` that product
    overflows float64 well before the individual counts become large, so
    the square root is taken as a sum of logarithms instead of forming
    the product directly.

    Returns ``0.0`` when any marginal is zero -- the conventional
    definition for the degenerate case, matching ``scikit-learn``. This
    happens whenever the hypothesis selects everything, selects nothing,
    or the target extension is empty or universal.
    """
    tp = float(matrix.true_positive)
    fp = float(matrix.false_positive)
    fn = float(matrix.false_negative)
    tn = float(matrix.true_negative)

    marginals = (tp + fp, tp + fn, tn + fp, tn + fn)
    if any(marginal <= 0.0 for marginal in marginals):
        # Undefined denominator. Zero is the standard convention: the
        # hypothesis carries no information about the target.
        return 0.0

    numerator = tp * tn - fp * fn
    if numerator == 0.0:
        return 0.0

    log_denominator = sum(math.log(marginal) for marginal in marginals)
    # Equivalent to numerator / sqrt(prod(marginals)), without forming
    # the product.
    value = math.copysign(
        math.exp(math.log(abs(numerator)) - 0.5 * log_denominator),
        numerator,
    )
    # Guard against accumulated floating-point drift past the bounds.
    return max(-1.0, min(1.0, value))


def _confusion_matrix(
    result: LearningProblemResult,
) -> ConfusionMatrix | None:
    """Reconstruct the confusion matrix, or ``None`` if not recoverable."""
    if result.error is not None:
        return None

    metrics = result.metrics
    target = result.target_extension
    hypothesis = getattr(result, "hypothesis_extension", None)
    if metrics is None or target is None or hypothesis is None:
        return None

    target_size = getattr(target, "positive", None)
    universe_size = getattr(target, "total", None)
    hypothesis_size = getattr(hypothesis, "positive", None)
    if (
        target_size is None
        or universe_size is None
        or hypothesis_size is None
    ):
        return None

    true_positive = int(metrics.intersection)
    false_positive = int(hypothesis_size) - true_positive
    false_negative = int(target_size) - true_positive
    true_negative = (
        int(universe_size) - true_positive - false_positive - false_negative
    )

    matrix = ConfusionMatrix(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )
    if not matrix.is_consistent:
        logger.warning(
            "Inconsistent confusion matrix for %s: %s. |P|=%s, |T|=%s, "
            "|U|=%s, |P∩T|=%s. Skipping MCC for this problem.",
            _problem_id(result),
            matrix.to_dict(),
            hypothesis_size,
            target_size,
            universe_size,
            true_positive,
        )
        return None
    return matrix


@dataclass(frozen=True)
class ClassificationSummary:
    """Pooled confusion matrix and MCC per embedding condition.

    Two MCC figures are reported and they answer different questions.
    ``mean_mcc`` weights every learning problem equally and is the
    quantity the paired analysis operates on. ``pooled_mcc`` is computed
    from the summed confusion matrix and is dominated by problems with
    large extensions. Divergence between them indicates that the
    embedding effect is concentrated in problems of a particular size.
    """

    condition: str
    n_observations: int
    mean_mcc: float
    pooled_mcc: float
    pooled_matrix: ConfusionMatrix
    mean_accuracy: float
    degenerate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "n_observations": self.n_observations,
            "mean_mcc": self.mean_mcc,
            "pooled_mcc": self.pooled_mcc,
            "pooled_matrix": self.pooled_matrix.to_dict(),
            "mean_accuracy": self.mean_accuracy,
            "degenerate_count": self.degenerate_count,
        }


def summarize_classification(
    runs: Mapping[int, SingleRunResult], condition: str
) -> ClassificationSummary | None:
    """Pool confusion matrices across every problem and seed."""
    import numpy as np

    per_problem_mcc: list[float] = []
    per_problem_accuracy: list[float] = []
    totals = [0, 0, 0, 0]
    degenerate = 0

    for seed in sorted(runs):
        embedding_result = runs[seed].get_embedding_result(condition)
        if embedding_result is None:
            continue
        for result in embedding_result.learning_problem_results:
            matrix = _confusion_matrix(result)
            if matrix is None:
                continue
            value = matthews_correlation_coefficient(matrix)
            if value == 0.0 and matrix.true_positive * matrix.true_negative == (
                matrix.false_positive * matrix.false_negative
            ):
                degenerate += 1
            per_problem_mcc.append(value)
            if result.metrics is not None:
                per_problem_accuracy.append(float(result.metrics.accuracy))
            totals[0] += matrix.true_positive
            totals[1] += matrix.false_positive
            totals[2] += matrix.false_negative
            totals[3] += matrix.true_negative

    if not per_problem_mcc:
        return None

    pooled = ConfusionMatrix(
        true_positive=totals[0],
        false_positive=totals[1],
        false_negative=totals[2],
        true_negative=totals[3],
    )
    return ClassificationSummary(
        condition=condition,
        n_observations=len(per_problem_mcc),
        mean_mcc=float(np.mean(per_problem_mcc)),
        pooled_mcc=matthews_correlation_coefficient(pooled),
        pooled_matrix=pooled,
        mean_accuracy=(
            float(np.mean(per_problem_accuracy))
            if per_problem_accuracy
            else float("nan")
        ),
        degenerate_count=degenerate,
    )


# --------------------------------------------------------------------------
# Assembled evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationResult:
    """The inferential result for one knowledge base.

    Serialized next to the descriptive summaries so the analysis is
    auditable and the family of tests is explicit.
    """

    knowledge_base: str
    n_problems: int
    n_seeds: int
    n_observations: int
    primary: MixedModelResult | None
    trend: TrendResult | None
    robustness: RobustnessResult | None
    mechanism: list[MechanismSummary] = field(default_factory=list)
    classification: list[ClassificationSummary] = field(default_factory=list)
    extension_sizes: ExtensionSizeSummary | None = None
    exploratory: list[ExploratoryFinding] = field(default_factory=list)
    design: PairedDesign | None = None
    outcome_unavailable: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation": {
                "knowledge_base": self.knowledge_base,
                "n_problems": self.n_problems,
                "n_seeds": self.n_seeds,
                "n_observations": self.n_observations,
                "primary": (
                    None if self.primary is None else self.primary.to_dict()
                ),
                "trend": None if self.trend is None else self.trend.to_dict(),
                "robustness": (
                    None
                    if self.robustness is None
                    else self.robustness.to_dict()
                ),
                "mechanism": [m.to_dict() for m in self.mechanism],
                "extension_sizes": (
                    None
                    if self.extension_sizes is None
                    else self.extension_sizes.to_dict()
                ),
                "exploratory": {
                    "adjustment": "benjamini-hochberg",
                    "q": EXPLORATORY_Q,
                    "findings": [f.to_dict() for f in self.exploratory],
                },
                "design": (
                    None if self.design is None else self.design.to_dict()
                ),
                "outcome_unavailable": list(self.outcome_unavailable),
                "notes": list(self.notes),
                "classification": [c.to_dict() for c in self.classification],
            }
        }


def _extension_sizes(design: PairedDesign) -> ExtensionSizeSummary | None:
    """|P| against |T| -- the direct test of the breadth reading.

    A recall-heavy, precision-flat pattern under ``dice`` predicts
    |P_dice| > |P_random|; the ratios against |T| say whether either
    condition systematically over- or under-generalizes.
    """
    import numpy as np

    size_observations = design.for_outcome("hypothesis_extension_size")
    if not size_observations:
        return None

    # Collapse to one value per problem before averaging, so that a
    # problem surviving in more seeds does not gain weight.
    dice_by_problem: dict[str, list[float]] = {}
    random_by_problem: dict[str, list[float]] = {}
    for observation in size_observations:
        dice_by_problem.setdefault(observation.problem_id, []).append(
            observation.dice_value
        )
        random_by_problem.setdefault(observation.problem_id, []).append(
            observation.random_value
        )

    problem_ids = sorted(dice_by_problem)
    dice_sizes = np.asarray(
        [float(np.mean(dice_by_problem[p])) for p in problem_ids],
        dtype=float,
    )
    random_sizes = np.asarray(
        [float(np.mean(random_by_problem[p])) for p in problem_ids],
        dtype=float,
    )

    # |T| is condition- and seed-invariant, so restrict it to exactly the
    # problems that contributed a hypothesis size. Averaging over a
    # different problem set would make the ratios incomparable.
    target_values = [
        float(design.target_extension_sizes[p])
        for p in problem_ids
        if p in design.target_extension_sizes
    ]
    target_mean = (
        float(np.mean(target_values)) if target_values else float("nan")
    )

    empty = design.for_outcome("empty_hypothesis")
    empty_dice = (
        float(np.mean([o.dice_value for o in empty])) if empty else float("nan")
    )
    empty_random = (
        float(np.mean([o.random_value for o in empty]))
        if empty
        else float("nan")
    )

    def ratio(numerator: float, denominator: float) -> float:
        return (
            float("nan")
            if not math.isfinite(denominator) or denominator == 0.0
            else numerator / denominator
        )

    return ExtensionSizeSummary(
        mean_hypothesis_size_dice=float(dice_sizes.mean()),
        mean_hypothesis_size_random=float(random_sizes.mean()),
        mean_target_size=target_mean,
        n_problems=len(problem_ids),
        n_problems_with_target_size=len(target_values),
        dice_over_target_ratio=ratio(float(dice_sizes.mean()), target_mean),
        random_over_target_ratio=ratio(float(random_sizes.mean()), target_mean),
        mean_per_problem_dice_ratio=_mean_per_problem_ratio(
            problem_ids, dice_by_problem, design.target_extension_sizes
        ),
        mean_per_problem_random_ratio=_mean_per_problem_ratio(
            problem_ids, random_by_problem, design.target_extension_sizes
        ),
        empty_hypothesis_rate_dice=empty_dice,
        empty_hypothesis_rate_random=empty_random,
    )


def _mean_per_problem_ratio(
    problem_ids: Sequence[str],
    sizes_by_problem: Mapping[str, Sequence[float]],
    target_sizes: Mapping[str, int],
) -> float:
    """Mean of |P|/|T| per problem, not a ratio of means.

    The two differ substantially when extension sizes are skewed, which
    they are: a ratio of means is dominated by the largest target
    extensions, whereas this weights every problem equally. Both are
    reported because they answer different questions.
    """
    import numpy as np

    ratios = [
        float(np.mean(sizes_by_problem[p])) / float(target_sizes[p])
        for p in problem_ids
        if target_sizes.get(p, 0) > 0
    ]
    return float(np.mean(ratios)) if ratios else float("nan")


def evaluate_knowledge_base(
    runs: Mapping[int, SingleRunResult],
    *,
    q: float = EXPLORATORY_Q,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    random_seed: int = 0,
    include_design: bool = True,
) -> EvaluationResult:
    """Run the full pre-specified hierarchy for one knowledge base.

    Every layer is guarded: a layer that cannot be estimated is recorded
    in ``notes`` or ``outcome_unavailable`` and the remaining layers
    still run, consistent with the suite's design to finish and report
    rather than abort.
    """
    design = build_paired_design(runs)
    notes: list[str] = []
    unavailable: list[str] = []

    primary_observations = design.for_outcome(PRIMARY_OUTCOME)
    primary: MixedModelResult | None = None
    trend: TrendResult | None = None
    robustness: RobustnessResult | None = None

    if not primary_observations:
        unavailable.append(PRIMARY_OUTCOME)
        notes.append(
            "lift is unavailable -- hardness annotation supplies "
            "atomic_baseline_f1, without which lift is undefined. The "
            "primary confirmatory analysis did not run; the f1_score "
            "robustness layer is reported in its place."
        )
    else:
        try:
            primary = fit_mixed_model(
                primary_observations,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=random_seed,
            )
        except (InferenceError, Exception) as error:  # noqa: BLE001
            notes.append(f"Primary model failed: {error}")

        try:
            trend = fit_complexity_trend(primary_observations)
        except (InferenceError, Exception) as error:  # noqa: BLE001
            notes.append(f"Complexity trend failed: {error}")


    # Robustness check with f1_score.
    robustness_observations = (
        design.for_outcome(ROBUSTNESS_OUTCOME)
    ) 
    # robustness_observations = (
    #     primary_observations
    #     if primary_observations
    #     else design.for_outcome(ROBUSTNESS_OUTCOME)
    # )
    if robustness_observations:
        try:
            robustness = run_robustness(
                robustness_observations,
                hypotheses=design.hypotheses,
                permutation_seed=random_seed,
            )
        except (InferenceError, Exception) as error:  # noqa: BLE001
            notes.append(f"Robustness layer failed: {error}")

    mechanism = [
        summary
        for outcome in MECHANISM_OUTCOMES
        if (summary := summarize_mechanism(design.for_outcome(outcome)))
        is not None
    ]
    classification = [
        summary
        for condition in ("dice", "random")
        if (summary := summarize_classification(runs, condition)) is not None
    ]
    try:
        exploratory = run_exploratory_grid(
            design, q=q, permutation_seed=random_seed
        )
    except Exception as error:  # noqa: BLE001
        notes.append(f"Exploratory grid failed: {error}")
        exploratory = []

    return EvaluationResult(
        knowledge_base=design.knowledge_base,
        n_problems=len(design.problem_ids),
        n_seeds=len(design.seeds),
        n_observations=design.n_observations,
        primary=primary,
        trend=trend,
        robustness=robustness,
        mechanism=mechanism,
        classification=classification,
        extension_sizes=_extension_sizes(design),
        exploratory=exploratory,
        design=design if include_design else None,
        outcome_unavailable=unavailable,
        notes=notes,
    )


@dataclass(frozen=True)
class SuiteEvaluation:
    """Evaluations across knowledge bases, plus the conjunction verdict."""

    evaluations: list[EvaluationResult]
    conjunction_holds: bool | None
    conjunction_statement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_evaluation": {
                "multiplicity": {
                    "primary": (
                        "none -- the claim is a conjunction across "
                        "knowledge bases, not a disjunction. Requiring "
                        "simultaneous rejection on every knowledge base "
                        "is already conservative."
                    ),
                    "trend": "none -- one pre-specified coefficient per KB.",
                    "exploratory": (
                        f"benjamini-hochberg at q={EXPLORATORY_Q}"
                    ),
                },
                "conjunction_holds": self.conjunction_holds,
                "conjunction_statement": self.conjunction_statement,
                "knowledge_bases": [
                    e.to_dict()["evaluation"] for e in self.evaluations
                ],
            }
        }


def evaluate_suite(
    runs_by_knowledge_base: Mapping[int, Mapping[str, SingleRunResult]]
    | Mapping[str, Mapping[int, SingleRunResult]],
    **kwargs: Any,
) -> SuiteEvaluation:
    """Evaluate every knowledge base and report the conjunction.

    No correction is applied across knowledge bases: the confirmatory
    claim is "dice > random on every knowledge base", a conjunction.
    Correcting a conjunction makes it needlessly weaker. All results are
    reported regardless of outcome.
    """
    # Accept either nesting order.
    normalized: dict[str, dict[int, SingleRunResult]] = {}
    for outer_key, inner in runs_by_knowledge_base.items():
        for inner_key, run in inner.items():
            if isinstance(outer_key, int):
                seed, name = outer_key, str(inner_key)
            else:
                name, seed = str(outer_key), int(inner_key)
            normalized.setdefault(name, {})[seed] = run

    evaluations: list[EvaluationResult] = []
    for name in sorted(normalized):
        try:
            evaluations.append(
                evaluate_knowledge_base(normalized[name], **kwargs)
            )
        except InferenceError as error:
            logger.warning("Skipping %s: %s", name, error)

    verdicts: list[bool] = []
    for evaluation in evaluations:
        primary = evaluation.primary
        if primary is None:
            verdicts.append(False)
            continue
        interval = (
            primary.bootstrap_ci95
            if primary.agreement == "disagree-trust-bootstrap"
            and primary.bootstrap_ci95 is not None
            else primary.ci95
        )
        verdicts.append(bool(interval[0] > 0.0))

    if not evaluations:
        holds: bool | None = None
        statement = "No knowledge base yielded an estimable contrast."
    else:
        holds = all(verdicts)
        passing = sum(verdicts)
        statement = (
            f"dice > random on {passing}/{len(verdicts)} knowledge bases "
            f"(95% interval excluding zero). Conjunction "
            f"{'holds' if holds else 'does not hold'}."
        )

    return SuiteEvaluation(
        evaluations=evaluations,
        conjunction_holds=holds,
        conjunction_statement=statement,
    )

# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def write_evaluation(
    evaluation: EvaluationResult | SuiteEvaluation,
    output_path: Path,  
    *,
    paired_observations_path: Path | None = None,
) -> None:
    """Write the inferential artifact, and optionally the raw pairs.

    The paired observations are worth persisting separately: they are the
    analysis's input, and storing both conditions' hypotheses side by
    side per problem enables per-concept inspection without re-running
    anything.
    """
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation.to_dict(), handle, indent=2, ensure_ascii=False)

    if paired_observations_path is None:
        return

    designs: list[PairedDesign] = []
    if isinstance(evaluation, EvaluationResult):
        if evaluation.design is not None:
            designs.append(evaluation.design)
    else:
        designs.extend(
            e.design for e in evaluation.evaluations if e.design is not None
        )

    if not designs:
        return

    payload = {
        design.knowledge_base: {
            "observations": {
                outcome: [o.to_dict() for o in items]
                for outcome, items in design.observations.items()
                if items
            },
            "paired_hypotheses": [h.to_dict() for h in design.hypotheses],
        }
        for design in designs
    }
    paired_observations_path.parent.mkdir(parents=True, exist_ok=True)
    with paired_observations_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)