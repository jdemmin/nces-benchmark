# src/eval/rq1.py
"""Artifacts answering RQ1, in increasing order of interpretative weight.

Results are reported per knowledge base and never pooled: a pooled estimate
would average over domain, size, structure and problem-population differences
and would be dominated by whichever knowledge base contributed the most
problems, producing a number that describes no actual setting.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.inference import ContrastResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankingConcordance:
    """Rank agreement of the condition ordering between two knowledge bases."""

    kb_a: str
    kb_b: str
    kendall_tau: float | None
    p_value: float | None
    n_conditions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kb_a": self.kb_a,
            "kb_b": self.kb_b,
            "kendall_tau": self.kendall_tau,
            "p_value": self.p_value,
            "n_conditions": self.n_conditions,
        }


def conditional_table(
    results: Sequence[ContrastResult], knowledge_base: str
) -> pd.DataFrame:
    """The descriptive answer: which architectures help, by how much."""
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.knowledge_base != knowledge_base:
            continue
        interval = result.interval
        test = result.test
        rows.append(
            {
                "condition": result.condition,
                "outcome": result.outcome,
                "estimate": result.estimate,
                "ci_low": interval.low if interval else None,
                "ci_high": interval.high if interval else None,
                "excludes_zero": interval.excludes_zero if interval else None,
                "p_value": test.p_value if test else None,
                "hodges_lehmann": test.hodges_lehmann if test else None,
                "wins": test.wins if test else None,
                "losses": test.losses if test else None,
                "ties": test.ties if test else None,
                "n_problems": result.n_problems,
                "n_seeds": result.n_seeds,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        "estimate", ascending=False, na_position="last"
    ).reset_index(drop=True)


def condition_rankings(
    results: Sequence[ContrastResult],
) -> dict[str, list[str]]:
    """Rank conditions by the point estimate within each knowledge base."""
    by_kb: dict[str, list[tuple[str, float]]] = {}
    for result in results:
        if result.estimate is None:
            continue
        by_kb.setdefault(result.knowledge_base, []).append(
            (result.condition, result.estimate)
        )
    return {
        kb: [
            condition
            for condition, _ in sorted(
                entries, key=lambda kv: (-kv[1], kv[0])
            )
        ]
        for kb, entries in by_kb.items()
    }


def ranking_concordance(
    results: Sequence[ContrastResult],
) -> list[RankingConcordance]:
    """Kendall's tau between each pair of knowledge bases.

    High concordance indicates an architecture ordering that transfers across
    domains; low concordance indicates that the best embedding model is a
    property of the ontology rather than of the architecture, which itself
    answers RQ1 in the negative sense.
    """
    estimates: dict[str, dict[str, float]] = {}
    for result in results:
        if result.estimate is None:
            continue
        estimates.setdefault(result.knowledge_base, {})[
            result.condition
        ] = result.estimate

    concordances: list[RankingConcordance] = []
    for kb_a, kb_b in combinations(sorted(estimates), 2):
        shared = sorted(set(estimates[kb_a]) & set(estimates[kb_b]))
        if len(shared) < 3:
            concordances.append(
                RankingConcordance(kb_a, kb_b, None, None, len(shared))
            )
            continue
        a = [estimates[kb_a][c] for c in shared]
        b = [estimates[kb_b][c] for c in shared]
        tau, p_value = stats.kendalltau(a, b)
        concordances.append(
            RankingConcordance(
                kb_a=kb_a,
                kb_b=kb_b,
                kendall_tau=None if np.isnan(tau) else float(tau),
                p_value=None if np.isnan(p_value) else float(p_value),
                n_conditions=len(shared),
            )
        )
    return concordances


def sign_agreement(results: Sequence[ContrastResult]) -> pd.DataFrame:
    """Per condition, the count of knowledge bases with a positive estimate.

    Consistency across knowledge bases is assessed by agreement in sign and
    ordering: a condition that improves on the control in all four supports a
    stronger conclusion than one that improves in two.
    """
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(
        [
            {
                "condition": r.condition,
                "knowledge_base": r.knowledge_base,
                "estimate": r.estimate,
            }
            for r in results
            if r.estimate is not None
        ]
    )
    if frame.empty:
        return frame
    for condition, group in frame.groupby("condition"):
        values = group["estimate"].to_numpy(dtype=float)
        rows.append(
            {
                "condition": condition,
                "n_knowledge_bases": int(values.size),
                "n_positive": int(np.sum(values > 0)),
                "n_negative": int(np.sum(values < 0)),
                "unanimous": bool(
                    np.all(values > 0) or np.all(values < 0)
                ),
                "mean_estimate_unpooled": float(values.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_positive", "mean_estimate_unpooled"], ascending=False
    ).reset_index(drop=True)


@dataclass(frozen=True)
class LinkSummary:
    """Spearman's rho between MRR and downstream ABL, within one KB."""

    knowledge_base: str
    rho: float | None
    p_value: float | None
    n_points: int
    reading: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_base": self.knowledge_base,
            "spearman_rho": self.rho,
            "p_value": self.p_value,
            "n_points": self.n_points,
            "reading": self.reading,
        }


#: Interpretation stated in advance, in both directions, so that neither
#: outcome can be presented as the expected one.
_POSITIVE_READING = (
    "Link-prediction quality is a usable proxy for embedding utility in "
    "neural class expression synthesis; improving the former is a productive "
    "route to improving the latter."
)
_FLAT_READING = (
    "NCES is insensitive to the properties of an embedding that MRR "
    "measures. Combined with a small effect against the random control, this "
    "supports the stronger reading that the learner uses its embedding input "
    "largely as an identifier rather than as a carrier of graph structure."
)


def link_summary(
    quality: pd.DataFrame,
    *,
    mrr_column: str = "mrr",
    abl_column: str = "mean_abl",
) -> list[LinkSummary]:
    """Relate the two independent measurements the suite produces.

    ``quality`` carries one row per (condition, knowledge base, seed) with a
    link-prediction score and the mean downstream ABL over the test split.
    Rho is computed *within* each knowledge base so that no cross-ontology
    scale difference on either axis can generate a spurious association.
    """
    summaries: list[LinkSummary] = []
    if quality.empty:
        return summaries

    for kb, group in quality.groupby("knowledge_base"):
        usable = group.dropna(subset=[mrr_column, abl_column])
        if len(usable) < 4 or usable[mrr_column].nunique() < 2:
            summaries.append(
                LinkSummary(
                    knowledge_base=str(kb),
                    rho=None,
                    p_value=None,
                    n_points=len(usable),
                    reading="indeterminate: too few distinct points",
                )
            )
            continue
        rho, p_value = stats.spearmanr(
            usable[mrr_column], usable[abl_column]
        )
        significant = (not np.isnan(p_value)) and p_value < 0.05
        summaries.append(
            LinkSummary(
                knowledge_base=str(kb),
                rho=float(rho),
                p_value=p_value,
                n_points=len(usable),
                reading=(
                    _POSITIVE_READING
                    if significant and rho > 0
                    else _FLAT_READING
                    if not significant
                    else "negative monotone relationship; unexpected, inspect "
                    "the per-condition scatter before interpreting"
                ),
            )
        )
    return summaries


def quality_frame(
    ranking_metrics: Mapping[tuple[str, str, int], Mapping[str, float]],
    downstream_abl: Mapping[tuple[str, str, int], float],
) -> pd.DataFrame:
    """Join ranking metrics to downstream ABL on (condition, kb, seed)."""
    rows: list[dict[str, Any]] = []
    for key, metrics in ranking_metrics.items():
        condition, knowledge_base, seed = key
        rows.append(
            {
                "condition": condition,
                "knowledge_base": knowledge_base,
                "seed": seed,
                "mrr": metrics.get("mrr"),
                "hits_at_1": metrics.get("hits@1"),
                "hits_at_3": metrics.get("hits@3"),
                "hits_at_10": metrics.get("hits@10"),
                "mean_abl": downstream_abl.get(key),
            }
        )
    return pd.DataFrame(rows)