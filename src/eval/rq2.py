# src/eval/rq2.py
"""RQ2: hyperparameter effects, observationally and experimentally.

The observational part relates hyperparameters to *embedding* quality across
the whole suite. It cannot by itself relate them to downstream performance,
because only the selected configuration is ever passed to the concept
learner; that gap is what the configuration sub-study closes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.pairing import dataframe_records

logger = logging.getLogger(__name__)

TUNED = ("batch_size", "learning_rate", "epochs")


def _native_scalar(value: Any) -> Any:
    """Unbox a numpy scalar (e.g. from a pandas Series) to a native Python type."""
    return value.item() if hasattr(value, "item") else value


@dataclass(frozen=True)
class Trial:
    """One attempted hyperparameter trial, including failures."""

    condition: str
    knowledge_base: str
    seed: int
    trial_index: int
    configuration: dict[str, Any]
    score: float | None
    used_validation_fallback: bool
    failed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "knowledge_base": self.knowledge_base,
            "seed": self.seed,
            "trial_index": self.trial_index,
            "configuration": self.configuration,
            "score": self.score,
            "used_validation_fallback": self.used_validation_fallback,
            "failed": self.failed,
            "error": self.error,
        }


def trials_to_frame(trials: Sequence[Trial]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        row = {
            "condition": trial.condition,
            "knowledge_base": trial.knowledge_base,
            "seed": trial.seed,
            "trial_index": trial.trial_index,
            "score": trial.score,
            "failed": trial.failed,
            "used_validation_fallback": trial.used_validation_fallback,
        }
        row.update(
            {name: trial.configuration.get(name) for name in TUNED}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def marginal_relationships(trials: pd.DataFrame) -> pd.DataFrame:
    """Spearman's rho between each tuned hyperparameter and validation MRR."""
    if trials.empty:
        return pd.DataFrame()
    usable = trials[~trials["failed"]].dropna(subset=["score"])
    rows: list[dict[str, Any]] = []
    grouped = usable.groupby(["condition", "knowledge_base"])
    for (condition, kb), group in grouped:
        for parameter in TUNED:
            values = group.dropna(subset=[parameter])
            if len(values) < 4 or values[parameter].nunique() < 2:
                rows.append(
                    {
                        "condition": condition,
                        "knowledge_base": kb,
                        "parameter": parameter,
                        "rho": None,
                        "p_value": None,
                        "n_trials": len(values),
                    }
                )
                continue
            rho, p_value = stats.spearmanr(
                values[parameter].astype(float), values["score"]
            )
            rows.append(
                {
                    "condition": condition,
                    "knowledge_base": kb,
                    "parameter": parameter,
                    "rho": None if np.isnan(rho) else float(rho),
                    "p_value": None if np.isnan(p_value) else float(p_value),
                    "n_trials": len(values),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class SelectionStability:
    """Whether the search converges to similar configurations across seeds.

    If the selected configuration varies widely across seeds while downstream
    performance does not, the architecture is hyperparameter-insensitive in
    the region the search explores. If both vary together, hyperparameter
    choice is consequential and the 32-trial budget is a binding constraint.
    """

    condition: str
    knowledge_base: str
    n_seeds: int
    distinct_configurations: int
    score_spread: float | None
    per_parameter_spread: dict[str, float | None]
    selected: list[dict[str, Any]]

    @property
    def verdict(self) -> str:
        if self.n_seeds < 2:
            return "indeterminate"
        if self.distinct_configurations == 1:
            return "stable"
        if self.distinct_configurations == self.n_seeds:
            return "unstable"
        return "partially stable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "knowledge_base": self.knowledge_base,
            "n_seeds": self.n_seeds,
            "distinct_configurations": self.distinct_configurations,
            "score_spread": self.score_spread,
            "per_parameter_spread": self.per_parameter_spread,
            "verdict": self.verdict,
            "selected": self.selected,
        }


def selection_stability(trials: pd.DataFrame) -> list[SelectionStability]:
    if trials.empty:
        return []
    usable = trials[~trials["failed"]].dropna(subset=["score"])
    results: list[SelectionStability] = []
    for (condition, kb), group in usable.groupby(
        ["condition", "knowledge_base"]
    ):
        best = group.loc[group.groupby("seed")["score"].idxmax()]
        signatures = set(
            best[list(TUNED)]
            .round({"learning_rate": 12})
            .astype(object)
            .where(best[list(TUNED)].notna(), None)
            .itertuples(index=False, name=None)
        )
        spreads: dict[str, float | None] = {}
        for parameter in TUNED:
            values = best[parameter].dropna().astype(float)
            spreads[parameter] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
        results.append(
            SelectionStability(
                condition=str(condition),
                knowledge_base=str(kb),
                n_seeds=int(best["seed"].nunique()),
                distinct_configurations=len(signatures),
                score_spread=(
                    float(best["score"].std(ddof=1))
                    if len(best) > 1
                    else None
                ),
                per_parameter_spread=spreads,
                selected=dataframe_records(best),
            )
        )
    return results


@dataclass(frozen=True)
class SubStudySelection:
    """The pre-recorded choice of knowledge base and architecture.

    The architecture is the one with the widest observed spread of validation
    MRR across its trial record, on the knowledge base whose main-suite
    results show the largest effect against the control. Recording the
    criterion prevents the choice from becoming a post-hoc selection of the
    most favourable pair.
    """

    knowledge_base: str
    condition: str
    criterion: str
    observed_mrr_spread: float
    observed_main_effect: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_base": self.knowledge_base,
            "condition": self.condition,
            "criterion": self.criterion,
            "observed_mrr_spread": self.observed_mrr_spread,
            "observed_main_effect": self.observed_main_effect,
        }


def select_substudy_target(
    trials: pd.DataFrame,
    main_effects: pd.DataFrame,
) -> SubStudySelection | None:
    """Apply the recorded selection criterion.

    ``main_effects`` carries columns ``condition``, ``knowledge_base`` and
    ``estimate``.
    """
    if trials.empty or main_effects.empty:
        logger.warning("Empty trials or main effects; cannot select substudy target.")
        return None
    usable = trials[~trials["failed"]].dropna(subset=["score"])
    if usable.empty:
        logger.warning("No usable trials found; cannot select substudy target.")
        return None

    effects = main_effects.dropna(subset=["estimate"])
    if effects.empty:
        logger.warning("No main effects available; cannot select substudy target.")
        return None
    kb = (
        effects.assign(magnitude=effects["estimate"].abs())
        .groupby("knowledge_base")["magnitude"]
        .max()
        .idxmax()
    )

    scoped = usable[usable["knowledge_base"] == kb]
    if scoped.empty:
        logger.warning("No trials found for selected knowledge_base=%s; cannot select substudy target.", kb)
        return None
    spreads = scoped.groupby("condition")["score"].std(ddof=1)
    spreads = spreads.dropna()
    if spreads.empty:
        logger.warning("No score spreads found for selected knowledge_base=%s; cannot select substudy target.", kb)
        return None
    condition = spreads.idxmax(skipna=True)

    effect_row = effects[
        (effects["knowledge_base"] == kb)
        & (effects["condition"] == condition)
    ]
    if effect_row.empty:
        logger.warning(
            "No main effect found for knowledge_base=%s and condition=%s",
            kb,
            condition,
        )
    return SubStudySelection(
        knowledge_base=str(kb),
        condition=str(condition),
        criterion=(
            "widest validation-MRR spread across the trial record, on the "
            "knowledge base whose main-suite results show the largest "
            "absolute effect against the control"
        ),
        observed_mrr_spread=float(spreads.loc[condition]),
        observed_main_effect=(
            float(effect_row["estimate"].iloc[0])
            if len(effect_row)
            else None
        ),
    )


def substudy_configurations(
    trials: pd.DataFrame,
    *,
    knowledge_base: str,
    condition: str,
    extreme_learning_rate: float = 0.3,
    scoped_guard: int = 0
) -> list[dict[str, Any]]:
    """Draw the four sub-study configurations from the trial record.

    Best-MRR, worst-MRR, median-MRR, and one with a deliberately extreme
    learning rate.
    """
    scoped = (
        trials[
            (~trials["failed"])
            & (trials["knowledge_base"] == knowledge_base)
            & (trials["condition"] == condition)
        ]
        .dropna(subset=["score"])
        .sort_values("score")
        .reset_index(drop=True)
    )
    if scoped.empty:
        logger.warning(
            "No successful trials found for knowledge_base=%s and condition=%s",
            knowledge_base,
            condition,
        )
        return []
    if len(scoped) < scoped_guard:
        logger.warning(
            "Not enough successful trials to draw all sub-study configurations for knowledge_base=%s and condition=%s",
            knowledge_base,
            condition,
        )
        return []

    def configuration(row: pd.Series, label: str) -> dict[str, Any]:
        # A Series is numpy-backed; unbox each cell to a native Python
        # scalar so the payload can be handed straight to ``json.dump``.
        payload = {
            parameter: _native_scalar(row[parameter]) for parameter in TUNED
        }
        payload["label"] = label
        payload["source_score"] = float(row["score"])
        return payload

    picks = [
        configuration(scoped.iloc[-1], "best_mrr"),
        configuration(scoped.iloc[0], "worst_mrr"),
        configuration(scoped.iloc[len(scoped) // 2], "median_mrr"),
    ]
    extreme_learning_rate_is_best_learning_rate = extreme_learning_rate == picks[0].get("learning_rate")
    extreme = dict(picks[0])
    extreme["label"] = "extreme_learning_rate"
    extreme["source_score"] = None
    if extreme_learning_rate_is_best_learning_rate:
        logger.info(
            "Extreme learning rate configuration for knowledge_base=%s and condition=%s "
                "will be used",
            knowledge_base,
            condition,
        )    
        extreme["learning_rate"] = None
    else:
        extreme["learning_rate"] = extreme_learning_rate
    picks.append(extreme)
    return picks