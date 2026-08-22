#src/models/dice_grid_search.py

import logging
from pathlib import Path
from typing import Any

from attr import dataclass

from src.config import EmbeddingSettings
from src.models.dice import train_embedding_model
from src.models.hpo_search_utils import best_trial_run_dir, selection_score, tiebreaker

logger = logging.getLogger(__name__)

@dataclass
class GridSearchOutcome:
    best_settings: EmbeddingSettings
    best_report: dict[str, Any]
    trials: list[dict[str, Any]]
    validation_error: str | None
    best_run_dir: Path | None

def grid_search(
    dataset_dir: Path,
    embeddings_dir: Path,
    settings: EmbeddingSettings,
    *,
    seed: int,
) -> GridSearchOutcome:
    """Evaluate the hyperparameter grid and return the best trial.

    Returns ``(best_settings, best_report, search_trials, validation_error)``.
    The selection metric is validation MRR with a fallback to test MRR; a
    trial that raises is recorded in ``search_trials`` rather than aborting
    the benchmark run.
    """
    trials: list[dict[str, Any]] = []
    best: tuple[float, EmbeddingSettings, dict[str, Any], str | None] | None = None
    for index, trial_settings in enumerate(settings.search_grid()):
        run_dir = embeddings_dir / f"trial_{index:02d}_{trial_settings.model_name}"
        record: dict[str, Any] = {
            "trial": index,
            "model_name": trial_settings.model_name,
            "embedding_dim": trial_settings.embedding_dim,
            "batch_size": trial_settings.batch_size,
        }
        try:
            report = train_embedding_model(
                dataset_dir, run_dir, trial_settings, seed=seed
            )
        except Exception as error:  # noqa: BLE001 - one trial must not kill the run
            logger.warning("Hyperparameter trial %d failed: %s", index, error)
            record["error"] = str(error)
            trials.append(record)
            continue
        score, validation_error = selection_score(report)
        record["score"] = score
        record["validation_error"] = validation_error
        record["metrics"] = {
            section: report[section]
            for section in ("Train", "Val", "Valid", "Test")
            if isinstance(report.get(section), dict)
        }
        record["run_dir"] = str(run_dir)
        trials.append(record)
        # negates selection by insertion order
        if score is not None and (
            best is None
            or score > best[0]
            or (
                score == best[0]
                and (trial_settings.embedding_dim, trial_settings.batch_size)
                < (best[1].embedding_dim, best[1].batch_size)
            )
        ):
            best = (score, trial_settings, report, validation_error)
    best = tiebreaker(best=best, trials=trials, seed=seed)
    if best is None:
        msg = (
            "GridSearch: Every DICE hyperparameter trial failed;"
            "see search_trials for details."
        )
        logger.error(msg)
        raise RuntimeError(msg)
    _, best_settings, best_report, best_validation_error = best
    best_run_dir = best_trial_run_dir(trials, best_settings)
    logger.info(
        "GridSearch: evaluated best DICE hyperparameters: %s. Located at %s",
        best_settings.to_dict(),
        best_run_dir,
    )
    logger.info(
        "GridSearch: Selected DICE configuration dim=%d batch=%d (score=%.4f)",
        best_settings.embedding_dim,
        best_settings.batch_size,
        best[0],
    )
    return GridSearchOutcome(
        best_settings=best_settings,
        best_report=best_report,
        trials=trials,
        validation_error=best_validation_error,
        best_run_dir=best_run_dir,
    )