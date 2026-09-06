# src/eval/pairing.py
"""Assembly of the paired design from persisted run artifacts.

The atomic observation is one (condition, learning problem, seed) triple for
one outcome. Pairing matches a KGE result to the ``random`` control by problem
id within one seed; a problem contributes an observation only when both
conditions produced a usable value.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONTROL = "random"


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records with native Python types.

    ``DataFrame.to_dict(orient="records")`` leaks numpy scalar dtypes (e.g.
    ``numpy.int64``), which the standard library's ``json`` module cannot
    serialize. Round-tripping through pandas' own JSON encoder first
    guarantees native Python ``int``/``float``/``bool``/``None``.
    """
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))

#: Outcomes carried through the paired design. ``abl`` is primary.
OUTCOMES = (
    "abl",
    "abl_norm",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "semantic_equivalence",
    "hypothesis_extension_size",
)

PRIMARY_OUTCOME = "abl"
FALLBACK_OUTCOME = "f1"


class PairedDesignImpossible(RuntimeError):
    """Raised when no seed carries both a condition and the control.

    This is the only condition under which the analysis stage raises: every
    downstream shortfall degrades and reports instead.
    """


@dataclass(frozen=True)
class Observation:
    """One scored problem under one condition and seed."""

    condition: str
    knowledge_base: str
    seed: int
    problem_id: str
    hypothesis: str
    depth: int | None
    dl_length: int | None
    expressivity: str | None
    extension_ratio: float | None
    atomic_baseline_f1: float | None
    target_extension_size: int | None
    values: Mapping[str, float | None]


@dataclass
class PairedDesign:
    """The paired design for one (condition, knowledge base)."""

    condition: str
    knowledge_base: str
    frame: pd.DataFrame
    seeds: tuple[int, ...]
    unpaired_problem_ids: tuple[str, ...]
    failures: Mapping[str, int]
    substituted_primary: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def n_observations(self) -> int:
        return len(self.frame)

    @property
    def n_problems(self) -> int:
        return self.frame["problem_id"].nunique()

    def available_outcomes(self) -> tuple[str, ...]:
        return tuple(
            outcome
            for outcome in OUTCOMES
            if f"d_{outcome}" in self.frame
            and self.frame[f"d_{outcome}"].notna().any()
        )

    def collapse(self, outcome: str) -> pd.Series:
        """Average each problem's paired differences across its seeds.

        Collapsing reduces per-problem noise by roughly ``sqrt(n_seeds)`` and
        makes across-problem independence defensible, which is what the
        signed-rank test requires. Run-to-run variability is not discarded; it
        is reported separately as the seed-level spread.
        """
        column = f"d_{outcome}"
        if column not in self.frame:
            return pd.Series(dtype=float)
        return (
            self.frame.dropna(subset=[column])
            .groupby("problem_id")[column]
            .mean()
            .sort_index()
        )

    def seed_means(self, outcome: str) -> pd.Series:
        column = f"d_{outcome}"
        if column not in self.frame:
            return pd.Series(dtype=float)
        return (
            self.frame.dropna(subset=[column])
            .groupby("seed")[column]
            .mean()
            .sort_index()
        )


def observations_to_frame(observations: Iterable[Observation]) -> pd.DataFrame:
    """Flatten observations into a tidy frame."""
    rows: list[dict[str, Any]] = []
    for obs in observations:
        row: dict[str, Any] = {
            "condition": obs.condition,
            "knowledge_base": obs.knowledge_base,
            "seed": obs.seed,
            "problem_id": obs.problem_id,
            "hypothesis": obs.hypothesis,
            "depth": obs.depth,
            "dl_length": obs.dl_length,
            "expressivity": obs.expressivity,
            "extension_ratio": obs.extension_ratio,
            "atomic_baseline_f1": obs.atomic_baseline_f1,
            "target_extension_size": obs.target_extension_size,
        }
        for outcome in OUTCOMES:
            row[outcome] = obs.values.get(outcome)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["semantic_equivalence"] = frame["semantic_equivalence"].astype(
            "float64"
        )
    return frame


def assemble(
    frame: pd.DataFrame,
    *,
    condition: str,
    knowledge_base: str,
    failures: Mapping[str, int] | None = None,
) -> PairedDesign:
    """Build the paired design for one (condition, knowledge base)."""
    scoped = frame[frame["knowledge_base"] == knowledge_base]
    treated = scoped[scoped["condition"] == condition]
    control = scoped[scoped["condition"] == CONTROL]

    if treated.empty or control.empty:
        raise PairedDesignImpossible(
            f"No runs for condition {condition!r} and/or the {CONTROL!r} "
            f"control on {knowledge_base!r}."
        )

    shared_seeds = sorted(set(treated["seed"]) & set(control["seed"]))
    if not shared_seeds:
        raise PairedDesignImpossible(
            f"No seed carries both {condition!r} and {CONTROL!r} on "
            f"{knowledge_base!r}; the paired design cannot be assembled."
        )

    keys = ["seed", "problem_id"]
    meta = [
        "depth",
        "dl_length",
        "expressivity",
        "extension_ratio",
        "atomic_baseline_f1",
        "target_extension_size",
    ]
    left = treated[treated["seed"].isin(shared_seeds)].drop_duplicates()
    right = control[control["seed"].isin(shared_seeds)].drop_duplicates()

    merged = left.merge(
        right[[*keys, "hypothesis", *OUTCOMES]],
        on=keys,
        how="inner",
        suffixes=("_treated", "_control"),
    )

    unpaired = sorted(
        (set(left[["seed", "problem_id"]].itertuples(index=False, name=None)) | set(right[["seed", "problem_id"]].itertuples(index=False, name=None)))
        - set(merged[["seed", "problem_id"]].itertuples(index=False, name=None))
    )

    columns: dict[str, Any] = {
        "seed": merged["seed"],
        "problem_id": merged["problem_id"],
        "hypothesis_treated": merged["hypothesis_treated"],
        "hypothesis_control": merged["hypothesis_control"],
    }
    for column in meta:
        columns[column] = merged[column]
    for outcome in OUTCOMES:
        treated_values = merged[f"{outcome}_treated"]
        control_values = merged[f"{outcome}_control"]
        columns[f"{outcome}_treated"] = treated_values
        columns[f"{outcome}_control"] = control_values
        columns[f"d_{outcome}"] = treated_values - control_values

    design = PairedDesign(
        condition=condition,
        knowledge_base=knowledge_base,
        frame=pd.DataFrame(columns),
        seeds=tuple(shared_seeds),
        unpaired_problem_ids=tuple(problem_id for _, problem_id in unpaired),
        failures=dict(failures or {}),
    )

    if design.frame.empty:
        design.notes.append("No paired observations available; design is empty.")
    elif not design.frame[f"d_{PRIMARY_OUTCOME}"].notna().any():
        design.substituted_primary = True
        design.notes.append(
            f"{PRIMARY_OUTCOME!r} unavailable (missing hardness annotation); "
            f"the same estimator is reported on {FALLBACK_OUTCOME!r} instead."
        )
    if unpaired:
        design.notes.append(
            f"{len(unpaired)} unpaired problem(s) excluded from every estimate."
        )
    return design


def primary_outcome(design: PairedDesign) -> str:
    """The outcome the estimates are reported on, after any substitution."""
    if design.substituted_primary and FALLBACK_OUTCOME in design.available_outcomes():
        return FALLBACK_OUTCOME
    elif not design.substituted_primary:
        return PRIMARY_OUTCOME
    else:
        raise ValueError("No valid primary outcome available.")


def resample_seed_clusters(
    frame: pd.DataFrame,
    seeds: Sequence[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw whole seeds with replacement.

    Individual observations are never resampled: doing so would destroy the
    within-run dependence that is the dominant correlation structure of the
    design and would yield an interval that is far too narrow.
    """
    drawn = rng.choice(np.asarray(seeds), size=len(seeds), replace=True)
    return pd.concat(
        [frame[frame["seed"] == seed] for seed in drawn], ignore_index=True
    )