
import logging
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from random import Random
from typing import Any

from src.config import EmbeddingSettings

logger = logging.getLogger(__name__)


class MRRNotFound(Enum):
        ValidationUnavailable = "Validation MRR unavailable. used test MRR."
        TrainOnly = (
            "Only train MRR was available; refusing to select on it "
            "(train MRR rewards memorization). Check that valid.txt/"
            "test.txt are discovered by dicee and that eval_model="
            "'train_val_test' took effect."
        )
        NoMRR = "No MRR metric was reported by DICE."


def best_trial_run_dir(
    trials: Sequence[dict[str, Any]], best: EmbeddingSettings
) -> Path:
    """Locate the stored run directory belonging to the winning trial."""
    for record in trials:
        if (
            record.get("embedding_dim") == best.embedding_dim
            and record.get("batch_size") == best.batch_size
            and "run_dir" in record
        ):
            return Path(record["run_dir"])
    raise RuntimeError("Could not locate the run directory of the best trial.")

def selection_score(report: dict[str, Any]) -> tuple[float | None, str | None]:
    """Extract the selection metric: validation MRR, then test MRR.
    Returns ``(score, validation_error)``.
    """
    for key in ("Val", "Valid", "Validation"):
        section = report.get(key)
        if isinstance(section, dict) and "MRR" in section:
            return float(section["MRR"]), None

    test = report.get("Test")
    if isinstance(test, dict) and "MRR" in test:
        return float(test["MRR"]), MRRNotFound.ValidationUnavailable.value
    train = report.get("Train")
    if isinstance(train, dict) and "MRR" in train:
        return None, MRRNotFound.TrainOnly.value
    return None, MRRNotFound.NoMRR.value


def tiebreaker(
        best: tuple[float, EmbeddingSettings, dict[str, Any], str | None] | None,
        trials: list[dict[str, Any]],
        seed: int
    )-> tuple[float, EmbeddingSettings, dict[str, Any], str | None] | None:
    """Randomly break ties between trials with the same score."""
    from math import inf

    if best is None:
        return None
    best_score = best[0]
    tied_trials = [trial for trial in trials if trial.get("score", 0) == best_score]
    if len(tied_trials) <= 1:
        return best
    tied_trials = [min(tied_trials, key=lambda t: (t.get("embedding_dim", inf), t.get("batch_size", inf)))]
    selected_trial = Random(seed).choice(tied_trials)
    selected_run_dir = Path(selected_trial["run_dir"]) or None
    logger.info(
        "GridSearch: Tiebreaker selected DICE configuration dim=%d batch=%d (score=%.4f) at %s",
        selected_trial["embedding_dim"],
        selected_trial["batch_size"],
        selected_trial["score"],
        selected_run_dir,
    )
    return (
        selected_trial["score"],
        EmbeddingSettings(
            embedding_dim=selected_trial["embedding_dim"],
            batch_size=selected_trial["batch_size"],
        ),
        selected_trial.get("report", {}),
        selected_trial.get("validation_error", None),
    )