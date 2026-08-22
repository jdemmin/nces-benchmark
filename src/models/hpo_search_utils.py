
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import EmbeddingSettings


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