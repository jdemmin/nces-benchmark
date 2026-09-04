# src/eval/suite.py
"""Suite-level analysis: consumes persisted artifacts, never re-executes.

Re-analysing therefore carries no risk of perturbing the results being
analysed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval import descriptive, rq1, rq2
from src.eval.failures import ShortfallLedger
from src.eval.inference import ContrastResult, analyse_contrast
from src.eval.pairing import (
    CONTROL,
    PairedDesign,
    PairedDesignImpossible,
    assemble,
    dataframe_records,
    primary_outcome,
)

logger = logging.getLogger(__name__)


def _json_fallback(obj: Any) -> Any:
    """Last-resort numpy-scalar unboxing for ``json.dump``.

    ``dataframe_records`` already converts every DataFrame-derived section;
    this only guards against a stray numpy scalar reaching ``json.dump``
    some other way.
    """
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


@dataclass
class SuiteAnalysis:
    contrasts: list[ContrastResult] = field(default_factory=list)
    mechanism: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    extension_sizes: dict[str, dict[str, Any]] = field(default_factory=dict)
    breakdowns: dict[str, dict[str, list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    trends: dict[str, dict[str, Any]] = field(default_factory=dict)
    concordance: list[dict[str, Any]] = field(default_factory=list)
    sign_agreement: list[dict[str, Any]] = field(default_factory=list)
    link: list[dict[str, Any]] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    ledger: ShortfallLedger = field(default_factory=ShortfallLedger)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrasts": [c.to_dict() for c in self.contrasts],
            "mechanism": self.mechanism,
            "extension_sizes": self.extension_sizes,
            "breakdowns": self.breakdowns,
            "trends": self.trends,
            "ranking_concordance": self.concordance,
            "sign_agreement": self.sign_agreement,
            "mrr_against_abl": self.link,
            "hyperparameters": self.hyperparameters,
            "shortfalls": self.ledger.to_dict(),
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=2,
                ensure_ascii=False,
                default=_json_fallback,
            )
        logger.info("Wrote suite analysis to %s", path)


def _key(design: PairedDesign) -> str:
    return f"{design.knowledge_base}/{design.condition}"


def analyse_suite(
    observations: pd.DataFrame,
    *,
    quality: pd.DataFrame | None = None,
    trials: pd.DataFrame | None = None,
    bootstrap_seed: int = 0,
) -> SuiteAnalysis:
    """Run every layer for every (condition, knowledge base)."""
    analysis = SuiteAnalysis()
    ledger = analysis.ledger

    conditions = sorted(
        c for c in observations["condition"].unique() if c != CONTROL
    )
    knowledge_bases = sorted(observations["knowledge_base"].unique())

    for knowledge_base in knowledge_bases:
        for condition in conditions:
            try:
                design = assemble(
                    observations,
                    condition=condition,
                    knowledge_base=knowledge_base,
                )
            except PairedDesignImpossible as exc:
                # The only situation that would raise for the suite as a
                # whole; here it is scoped to one contrast so the rest runs.
                ledger.note(f"{knowledge_base}/{condition}", str(exc))
                continue

            outcome = primary_outcome(design)
            key = _key(design)

            analysis.contrasts.append(
                analyse_contrast(
                    design, outcome, bootstrap_seed=bootstrap_seed
                )
            )

            rows = ledger.guard(
                f"{key}:mechanism", lambda d=design: descriptive.mechanism(d)
            )
            if rows is not None:
                analysis.mechanism[key] = [r.to_dict() for r in rows]

            sizes = ledger.guard(
                f"{key}:extension_size",
                lambda d=design: descriptive.extension_size_summary(d),
            )
            if sizes is not None:
                analysis.extension_sizes[key] = sizes.to_dict()

            cells: dict[str, list[dict[str, Any]]] = {}
            for by in ("expressivity", "extension_ratio"):
                frame = ledger.guard(
                    f"{key}:breakdown:{by}",
                    lambda d=design, b=by, o=outcome: descriptive.breakdown(
                        d, o, by=b
                    ),
                )
                if frame is not None and not frame.empty:
                    cells[by] = dataframe_records(frame)
            if cells:
                analysis.breakdowns[key] = cells

            depth = ledger.guard(
                f"{key}:trend:depth",
                lambda d=design, o=outcome: descriptive.complexity_trend(
                    d, o, predictor="depth", seed=bootstrap_seed
                ),
            )
            ratio = ledger.guard(
                f"{key}:trend:extension_ratio",
                lambda d=design, o=outcome: descriptive.complexity_trend(
                    d, o, predictor="extension_ratio", seed=bootstrap_seed
                ),
            )
            if depth is not None and ratio is not None:
                analysis.trends[key] = {
                    "depth": depth.to_dict(),
                    "extension_ratio": ratio.to_dict(),
                    "dissociation": descriptive.dissociation_check(
                        depth, ratio
                    ),
                }

    concordance = ledger.guard(
        "ranking_concordance",
        lambda: rq1.ranking_concordance(analysis.contrasts),
    )
    if concordance is not None:
        analysis.concordance = [c.to_dict() for c in concordance]

    agreement = ledger.guard(
        "sign_agreement", lambda: rq1.sign_agreement(analysis.contrasts)
    )
    if agreement is not None and not agreement.empty:
        analysis.sign_agreement = dataframe_records(agreement)

    if quality is not None and not quality.empty:
        link = ledger.guard("mrr_against_abl", lambda: rq1.link_summary(quality))
        if link is not None:
            analysis.link = [s.to_dict() for s in link]

    if trials is not None and not trials.empty:
        marginals = ledger.guard(
            "hpo:marginals", lambda: rq2.marginal_relationships(trials)
        )
        stability = ledger.guard(
            "hpo:stability", lambda: rq2.selection_stability(trials)
        )
        main_effects = pd.DataFrame(
            [
                {
                    "condition": c.condition,
                    "knowledge_base": c.knowledge_base,
                    "estimate": c.estimate,
                }
                for c in analysis.contrasts
            ]
        )
        selection = ledger.guard(
            "hpo:substudy_selection",
            lambda: rq2.select_substudy_target(trials, main_effects),
        )
        analysis.hyperparameters = {
            "marginal_relationships": (
                dataframe_records(marginals)
                if marginals is not None and not marginals.empty
                else []
            ),
            "selection_stability": (
                [s.to_dict() for s in stability] if stability else []
            ),
            "substudy_selection": (
                selection.to_dict() if selection else None
            ),
            "substudy_configurations": (
                rq2.substudy_configurations(
                    trials,
                    knowledge_base=selection.knowledge_base,
                    condition=selection.condition,
                )
                if selection
                else []
            ),
        }

    return analysis


def load_observations(paths: Sequence[Path]) -> pd.DataFrame:
    """Load persisted paired-observation artifacts into one tidy frame."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        frames.append(pd.DataFrame(payload["observations"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)