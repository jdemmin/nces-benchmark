# src/analysis/stats.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from scipy import stats

from src.data.complexity import COMPLEXITY_STRATA, Complexity
from src.data.results import EmbeddingResult, LearningProblemResult, SingleRunResult

METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "jaccard",
)


@dataclass(frozen=True)
class PairedSample:
    """Per-LP paired observation of one metric under two conditions."""

    lp_id: str
    target_concept: str
    complexity: Complexity
    dice_value: float
    random_value: float

    @property
    def delta(self) -> float:
        return self.dice_value - self.random_value


def extract_pairs(
    run: SingleRunResult,
    metric: str,
) -> list[PairedSample]:
    """Join dice and random per-LP metrics on the learning-problem id."""
    dice = run.dice_embedding_result
    rand = run.random_embedding_result
    if dice is None or rand is None:
        raise ValueError(
            "Both embedding conditions are required for a paired test; "
            f"got dice={dice is not None}, random={rand is not None}."
        )

    def index(result: EmbeddingResult) -> dict[str, LearningProblemResult]:
        return {
            lpr.learning_problem.id: lpr
            for lpr in result.learning_problem_results
            if lpr.metrics is not None and lpr.error is None
        }

    dice_by_id = index(dice)
    rand_by_id = index(rand)
    shared = sorted(set(dice_by_id) & set(rand_by_id))

    pairs: list[PairedSample] = []
    for lp_id in shared:
        d, r = dice_by_id[lp_id], rand_by_id[lp_id]
        pairs.append(
            PairedSample(
                lp_id=lp_id,
                target_concept=d.learning_problem.target_concept,
                complexity=d.learning_problem.complexity,
                dice_value=getattr(d.metrics, metric),
                random_value=getattr(r.metrics, metric),
            )
        )
    return pairs

def extract_pairs_for_all_metrics(
    run: SingleRunResult,
) -> dict[str, list[PairedSample]]:
    """Extract paired samples for all metrics."""
    return {metric: extract_pairs(run, metric) for metric in METRIC_KEYS}

def wilcoxon_for_all_metrics(
    run: SingleRunResult,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "pratt",
) -> dict[str, WilcoxonResult]:
    """Perform Wilcoxon tests for all metrics."""
    results: dict[str, WilcoxonResult] = {}
    pairs_by_metric = extract_pairs_for_all_metrics(run)
    for metric, pairs in pairs_by_metric.items():
        results[metric] = wilcoxon_compare(
            pairs,
            metric,
            alternative=alternative,
            zero_method=zero_method,
        )
    return results

@dataclass(frozen=True)
class WilcoxonResult:
    metric: str
    n_pairs: int
    n_nonzero: int
    statistic: float
    p_value: float
    rank_biserial: float
    median_delta: float
    alternative: str
    zero_method: str
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "n_pairs": self.n_pairs,
            "n_nonzero": self.n_nonzero,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "rank_biserial": self.rank_biserial,
            "median_delta": self.median_delta,
            "alternative": self.alternative,
            "zero_method": self.zero_method,
            "note": self.note,
        }


def _rank_biserial(deltas: list[float]) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1]."""
    nonzero = [d for d in deltas if d != 0.0]
    if not nonzero:
        return 0.0
    ranks = stats.rankdata([abs(d) for d in nonzero])
    w_pos = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    total = w_pos + w_neg
    return (w_pos - w_neg) / total if total else 0.0


def wilcoxon_compare(
    pairs: list[PairedSample],
    metric: str,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "pratt",
) -> WilcoxonResult:
    deltas = [p.delta for p in pairs]
    nonzero = [d for d in deltas if d != 0.0]
    median = float(stats.scoreatpercentile(deltas, 50)) if deltas else 0.0

    if len(nonzero) < 6:
        return WilcoxonResult(
            metric=metric,
            n_pairs=len(pairs),
            n_nonzero=len(nonzero),
            statistic=math.nan,
            p_value=math.nan,
            rank_biserial=_rank_biserial(deltas),
            median_delta=median,
            alternative=alternative,
            zero_method=zero_method,
            note=(
                f"Underpowered: {len(nonzero)} non-zero differences. "
                "Minimum attainable two-sided p at n=6 is 0.031; "
                "below that no result can reach alpha=0.05."
            ),
        )

    res = stats.wilcoxon(
        deltas,
        alternative=alternative,
        zero_method=zero_method,
        method="auto",
    )
    return WilcoxonResult(
        metric=metric,
        n_pairs=len(pairs),
        n_nonzero=len(nonzero),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        rank_biserial=_rank_biserial(deltas),
        median_delta=median,
        alternative=alternative,
        zero_method=zero_method,
    )

def holm_adjust(results: list[WilcoxonResult]) -> list[tuple[WilcoxonResult, float]]:
    """Holm-Bonferroni step-down adjustment over a family of tests."""
    valid = [r for r in results if not math.isnan(r.p_value)]
    order = sorted(valid, key=lambda r: r.p_value)
    m = len(order)
    adjusted: list[tuple[WilcoxonResult, float]] = []
    running_max = 0.0
    for i, r in enumerate(order):
        p_adj = min(1.0, (m - i) * r.p_value)
        running_max = max(running_max, p_adj)  # enforce monotonicity
        adjusted.append((r, running_max))
    return adjusted


def wilcoxon_by_stratum_by_metric(
    pairs: list[PairedSample],
    metric: str = "f1_score",
    key: str = "dl_length",
) -> dict[str, WilcoxonResult]:
    from collections import defaultdict
    """
    Perform Wilcoxon tests for a single metric, stratified by a complexity key.

    Returns a dictionary mapping stratum values to WilcoxonResult objects.
    """

    buckets: dict[str, list[PairedSample]] = defaultdict(list)
    for p in pairs:
        buckets[str(getattr(p.complexity, key))].append(p)
    return {
        bucket: wilcoxon_compare(group, metric)
        for bucket, group in sorted(buckets.items())
    }

def wilcoxon_by_stratum_for_all_metrics(
    single_run_result: SingleRunResult,
    key: str = "dl_length",
) -> dict[str, dict[str, WilcoxonResult]]:
    """
    Perform Wilcoxon tests for all metrics, stratified by a complexity key.
    Return type has the structure:
        {
            metric: {
                stratum_value: WilcoxonResult
            }
        }
    
    """

    results: dict[str, dict[str, WilcoxonResult]] = {}
    pairs = extract_pairs_for_all_metrics(single_run_result)
    for metric in METRIC_KEYS:
        results[metric] = wilcoxon_by_stratum_by_metric(pairs=pairs[metric], metric=metric, key=key)
    return results


def wilcoxon_for_all_strata_for_all_metrics(
    single_run_result: SingleRunResult,
) -> dict[str, dict[str, dict[str, WilcoxonResult]]]:
    """
    Perform Wilcoxon tests for all metrics, stratified by all complexity keys.
    Return type has the structure:
        {
            complexity_stratum: {
                metric: {
                    stratum_value: WilcoxonResult
                }
            }
        }
    """

    results: dict[str, dict[str, dict[str, WilcoxonResult]]] = {}
    for complexity_stratum in COMPLEXITY_STRATA:
        results[complexity_stratum] = (
            wilcoxon_by_stratum_for_all_metrics(
                single_run_result = single_run_result, 
                key=complexity_stratum
            )
        )
    return results


def wilcoxon_for_all_strata_by_metric(
    single_run_result: SingleRunResult,
    metric: str = "f1_score",
) -> dict[str, dict[str, WilcoxonResult]]:
    """
    Perform Wilcoxon tests for a single metric, stratified by all complexity keys.
    Return type has the structure:
        {
            complexity_stratum: {
                stratum_value: WilcoxonResult
            }
        }
    """

    results: dict[str, dict[str, WilcoxonResult]] = {}
    for complexity_stratum in COMPLEXITY_STRATA:
        results[complexity_stratum] = wilcoxon_by_stratum_by_metric(
            pairs=extract_pairs_for_all_metrics(single_run_result)[metric],
            metric=metric,
            key=complexity_stratum,
        )
    return results


# def holm_adjust_across_metrics(
#         single_run_result: SingleRunResult
#     ) -> dict[str, tuple[WilcoxonResult, float]]:
#     """
#     Apply Holm-Bonferroni step-down adjustment on Wilcoxon test results
#     across all metrics.

#     Return type has the structure:
#         {
#             metric: (WilcoxonResult, adjusted_p_value)
#         }
#     """
#     wilcoxon_results = wilcoxon_for_all_metrics(single_run_result)
#     holm_adjusted = holm_adjust([wr for wr in wilcoxon_results.values()])

#     result: dict[str, tuple[WilcoxonResult, float]] = {}
#     for metric, wilcoxon_result in wilcoxon_results.items():
#         holm_adjusted_entry = holm_adjusted.pop(0)
#         if wilcoxon_result == holm_adjusted_entry[0]:
#             result[metric] = holm_adjusted_entry
#     return result

def holm_adjust_across_kbs(
        run_by_kb: dict[str, SingleRunResult]
        ) -> dict[str, tuple[dict[str, Any], dict[str, float]]]:
    """
    Apply Holm-Bonferroni step-down adjustment on Wilcoxon test results
    across all knowledge bases (KBs).

    Return type has the structure:
        {
            kb: WilcoxonResult
            adjusted_p_value: float
        }
    """

    wilcoxon_results = {
        kb: wilcoxon_compare(extract_pairs(result, "f1_score"), "f1_score")
        for kb, result in run_by_kb.items()
    }
    collapsed_wilcox_dict: list[WilcoxonResult] = []
    for kb_result in wilcoxon_results.values():
        collapsed_wilcox_dict.extend([kb_result])
    holm_adjusted = holm_adjust(collapsed_wilcox_dict)

    result: dict[str, tuple[dict[str, Any], dict[str, float]]] = {}
    for adjusted_p_value in sorted(holm_adjusted, key=lambda x: x[0].p_value):
        for kb, kb_result in wilcoxon_results.items():
            if kb_result == adjusted_p_value[0]:
                result[kb] = (
                    adjusted_p_value[0].to_dict(),
                    {"adjusted_p_value": adjusted_p_value[1]}
                )
    return result


def holm_adjust_across_complexity_strata(
    single_run_result: SingleRunResult,
) -> dict[str, dict[str, tuple[dict[str, Any], dict[str, float]]]]:
    """
    Apply Holm-Bonferroni step-down adjustment on Wilcoxon test results
    for `f1_score` metric, stratified by all complexity keys.
    The `f1_score` metric is the only one considered for Holm-Bonferroni
    adjustment because it is the primary metric of interest.

    Return type has the structure:
        {
            complexity_stratum: {
                stratum_value: (WilcoxonResult, adjusted_p_value)
            }
        }
    """

    # Collapse all Wilcoxon results into a single list for Holm-Bonferroni adjustment.
    collapsed_wilcox_dict: list[WilcoxonResult] = []
    wilcoxon_results = wilcoxon_for_all_strata_by_metric(single_run_result)
    for stratum, wilcoxon_group in wilcoxon_results.items():
        for wilcoxon_result in wilcoxon_group.values():
            collapsed_wilcox_dict.extend([wilcoxon_result])
    holm_adjusted = holm_adjust(collapsed_wilcox_dict)

    # Rebuild the nested dictionary structure with adjusted p-values.
    results: dict[str, dict[str, tuple[dict[str, Any], dict[str, float]]]] = {}
    for adjusted_p_value in sorted(holm_adjusted, key=lambda x: x[0].p_value):
        for stratum, wilcoxon_group in wilcoxon_results.items():
            for stratum_value, wilcoxon_result in sorted(
                wilcoxon_group.items(), 
                key=lambda x: x[1].p_value
            ):
                if wilcoxon_result == adjusted_p_value[0]:
                    results.setdefault(stratum, {})
                    results[stratum][stratum_value] = (
                        adjusted_p_value[0].to_dict(), 
                        {"adjusted_p_value": adjusted_p_value[1]}
                    )
    return results


# def pairs_across_seeds(
#     runs_by_seed: dict[int, SingleRunResult],
#     metric: str,
# ) -> list[PairedSample]:
#     """Mean-per-LP across seeds, preserving pairing."""
#     from collections import defaultdict

#     dice_acc: dict[str, list[float]] = defaultdict(list)
#     rand_acc: dict[str, list[float]] = defaultdict(list)
#     meta: dict[str, PairedSample] = {}

#     for run in runs_by_seed.values():
#         for p in extract_pairs(run, metric):
#             dice_acc[p.lp_id].append(p.dice_value)
#             rand_acc[p.lp_id].append(p.random_value)
#             meta[p.lp_id] = p

#     return [
#         PairedSample(
#             lp_id=lp_id,
#             target_concept=meta[lp_id].target_concept,
#             complexity=meta[lp_id].complexity,
#             dice_value=sum(dice_acc[lp_id]) / len(dice_acc[lp_id]),
#             random_value=sum(rand_acc[lp_id]) / len(rand_acc[lp_id]),
#         )
#         for lp_id in sorted(meta)
#     ]

# def wilcoxon_for_all_seeds_by_metric(
#     runs_by_seed: dict[int, SingleRunResult],
#     metric: str,
# ) -> WilcoxonResult:
#     """
#     Perform Wilcoxon tests for a specific metric across all seeds.

#     Return type is a WilcoxonResult.
#     """
#     all_pairs: list[PairedSample] = pairs_across_seeds(runs_by_seed, metric=metric)
#     return wilcoxon_compare(pairs=all_pairs, metric=metric)


# def wilcoxon_for_all_seeds_for_all_metrics(
#     runs_by_seed: dict[int, SingleRunResult],
# ) -> dict[str, WilcoxonResult]:
#     """
#     Perform Wilcoxon tests for all metrics across all seeds.

#     Return type has the structure:
#         {
#             metric: WilcoxonResult
#         }
#     """

#     results: dict[str, WilcoxonResult] = {}
#     for metric in METRIC_KEYS:
#         results[metric] = wilcoxon_for_all_seeds_by_metric(
#             runs_by_seed=runs_by_seed,
#             metric=metric,
#         )
#     return results



# def wilcoxon_for_all_seeds_by_stratum_by_metric(
#     runs_by_seed: dict[int, SingleRunResult],
#     key: str,
#     metric: str,
# ) -> dict[str, WilcoxonResult]:
#     """
#     Perform Wilcoxon tests for all metrics, stratified by all complexity keys, across seeds.

#     Return type has the structure:
#         {
#             stratum_value: WilcoxonResult
#         }
#     """

#     all_pairs: list[PairedSample] = pairs_across_seeds(runs_by_seed, metric=metric)
#     return wilcoxon_by_stratum_by_metric(all_pairs, metric=metric, key=key)


# def wilcoxon_for_all_seeds_by_stratum_for_all_metrics(
#     runs_by_seed: dict[int, SingleRunResult],
#     key: str,
# ) -> dict[str, dict[str, WilcoxonResult]]:
#     """
#     Perform Wilcoxon tests for all metrics, stratified by a specific complexity key, across seeds.

#     Return type has the structure:
#         {
#             metric: {
#                 stratum_value: WilcoxonResult
#             }
#         }
#     """

#     results: dict[str, dict[str, WilcoxonResult]] = {}
#     for metric in METRIC_KEYS:
#         results[metric] = wilcoxon_for_all_seeds_by_stratum_by_metric(
#             runs_by_seed=runs_by_seed,
#             key=key,
#             metric=metric,
#         )
#     return results


# def wilcoxon_for_all_seeds_for_all_strata_for_all_metrics(
#     runs_by_seed: dict[int, SingleRunResult],
# ) -> dict[str, dict[str, dict[str, WilcoxonResult]]]:
#     """
#     Perform Wilcoxon tests for all metrics, stratified by all complexity keys, across seeds.

#     Return type has the structure:
#         {
#             complexity_stratum: {
#                 metric: {
#                     stratum_value: WilcoxonResult
#                 }
#             }
#         }
#     """

#     results: dict[str, dict[str, dict[str, WilcoxonResult]]] = {}
#     for key in COMPLEXITY_STRATA:
#         results[key] = wilcoxon_for_all_seeds_by_stratum_for_all_metrics(
#             runs_by_seed=runs_by_seed,
#             key=key,
#         )
#     return results