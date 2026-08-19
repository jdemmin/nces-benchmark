# src/models/dice.py
"""DICE dataset preparation, embedding training, search, and export."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SPLIT_RATIOS, EmbeddingSettings
from src.data.ontology import Triple, local_name, parse_triples

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
        score, validation_error = _selection_score(report)
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
    if best is None:
        raise RuntimeError(
            "Every DICE hyperparameter trial failed; see search_trials for details."
        )
    _, best_settings, best_report, best_validation_error = best
    logger.info(
        "Selected DICE configuration dim=%d batch=%d (score=%.4f)",
        best_settings.embedding_dim,
        best_settings.batch_size,
        best[0],
    )
    return GridSearchOutcome(
        best_settings=best_settings,
        best_report=best_report,
        trials=trials,
        validation_error=best_validation_error,
        best_run_dir=_best_trial_run_dir(trials, best_settings),
    )

    
def _best_trial_run_dir(
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

class MRRNotFound(Enum):
        ValidationUnavailable = "Validation MRR unavailable. used test MRR."
        TrainOnly = (
            "Only train MRR was available; refusing to select on it "
            "(train MRR rewards memorization). Check that valid.txt/"
            "test.txt are discovered by dicee and that eval_model="
            "'train_val_test' took effect."
        )
        NoMRR = "No MRR metric was reported by DICE."

@dataclass
class EmbeddingResult:
    """Outcome of one DICE embedding-training workflow."""

    model_name: str
    embeddings_path: Path
    embedding_condition: str
    embedding_dim: int
    batch_size: int
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    search_trials: list[dict[str, Any]] = field(default_factory=list)
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embeddings_path": str(self.embeddings_path),
            "embedding_condition": self.embedding_condition,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "score": self.score,
            "metrics": self.metrics,
            "search_trials": self.search_trials,
            "validation_error": self.validation_error,
        }

#: dicee's ReadFromDisk does substring matching on full glob paths.
DICEE_RESERVED_PATH_TOKENS: tuple[str, ...] = ("train", "valid", "test")


def _selection_score(report: dict[str, Any]) -> tuple[float | None, str | None]:
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

def _assert_dicee_safe_dataset_dir(directory: Path) -> None:
    """Reject dataset directories dicee would mis-parse.

    ``ReadFromDisk`` iterates ``glob.glob(dataset_dir + '/*')`` and routes
    each file with ``if 'train' in i``, where ``i`` is the *full* path. A
    reserved token anywhere in an ancestor directory therefore captures
    every split file into ``raw_train_set``, leaving validation and test
    unset and the eval report train-only.
    """
    parent = str(directory.resolve().parent).lower()
    offenders = [t for t in DICEE_RESERVED_PATH_TOKENS if t in parent]
    if offenders:
        raise ValueError(
            f"The DICE dataset directory {directory} has ancestor path "
            f"segments containing {offenders}. dicee matches these as "
            f"substrings of the full path and would route valid.txt and "
            f"test.txt into the training set. Choose a benchmark name or "
            f"output directory without these tokens."
        )

def stage_dicee_dataset(data_dir: Path, staging_root: Path) -> Path:
    """Copy the split files into a dicee-safe directory.

    Returns the directory to pass as ``dataset_dir``.
    """
    import shutil

    staged = staging_root / "kg"
    staged.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid", "test"):
        shutil.copyfile(data_dir / f"{name}.txt", staged / f"{name}.txt")
    return staged

def write_dicee_dataset(
    triples: Sequence[Triple],
    directory: Path,
    *,
    seed: int,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> dict[str, int]:
    """Write ``train.txt``/``valid.txt``/``test.txt`` for ``dicee``.

    DICE consumes tab-separated triple files from a dataset directory. The
    split is deterministic in ``seed`` so a benchmark run is reproducible.
    """
    if not triples:
        raise ValueError("Cannot build a DICE dataset from zero triples.")

    # dicee's ReadFromDisk matches 'train'/'valid'/'test' as substrings of
    # the *full* glob path, so any of those words in an ancestor directory
    # silently misroutes every split file. See KNOWN_ISSUES.md.
    _assert_dicee_safe_dataset_dir(directory)

    directory.mkdir(parents=True, exist_ok=True)
    shuffled = [triple.as_tuple() for triple in triples]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    n_train = max(1, int(total * ratios[0]))
    n_valid = int(total * ratios[1])
    if n_train + n_valid >= total:
        n_valid = max(0, total - n_train - 1)

    partitions = {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train : n_train + n_valid],
        "test": shuffled[n_train + n_valid :],
    }
    # DICE requires non-empty validation/test files when eval_model spans them.
    for name in ("valid", "test"):
        if not partitions[name]:
            partitions[name] = partitions["train"][:1]

    for name, rows in partitions.items():
        path = directory / f"{name}.txt"
        with path.open("w", encoding="utf-8") as handle:
            for subject, predicate, obj in rows:
                handle.write(f"{subject}\t{predicate}\t{obj}\n")

    counts = {name: len(rows) for name, rows in partitions.items()}
    logger.info("Wrote DICE dataset to %s (%s)", directory, counts)
    return counts


def train_embedding_model(
    dataset_dir: Path,
    run_dir: Path,
    settings: EmbeddingSettings,
    *,
    seed: int,
) -> dict[str, Any]:
    """
    Train one DICE model and return its report.

    Uses ``dicee.executer.Execute`` with an explicit ``Namespace`` because the
    ``Namespace`` defaults differ from the ``dicee`` CLI defaults (notably
    ``trainer``), which would otherwise silently change the backend.
    """
    from dicee.config import Namespace
    from dicee.executer import Execute

    run_dir.parent.mkdir(parents=True, exist_ok=True)

    # TODO: dice does not use the given dataset_dir as intended.
    # It does not consider validation/test files in the given
    # directory... but why???
    args = Namespace()
    args.model = settings.model_name
    args.dataset_dir = str(dataset_dir)
    args.path_to_store_single_run = str(run_dir)
    args.num_epochs = settings.epochs
    args.embedding_dim = settings.embedding_dim
    args.batch_size = settings.batch_size
    args.scoring_technique = settings.scoring_technique
    args.trainer = settings.trainer
    args.eval_model = settings.eval_model
    args.num_core = settings.num_core
    args.random_seed = seed
    args.lr = settings.learning_rate

    logger.info(
        "Training DICE %s (dim=%d, batch=%d, epochs=%d)",
        settings.model_name,
        settings.embedding_dim,
        settings.batch_size,
        settings.epochs,
    )
    report = Execute(args).start()
    logger.info(
        f"DICE report section: {sorted(report)}"
    )
    return dict(report)




def search_best_embedding_setting(
    dataset_dir: Path,
    embeddings_dir: Path,
    settings: EmbeddingSettings,
    *,
    seed: int,
) -> tuple[
    EmbeddingSettings, dict[str, Any], list[dict[str, Any]], str | None, Path
]:
    """Optimize the DICE hyperparameters and return the best trial.

    Returns ``(best_settings, best_report, trials, validation_error,
    best_run_dir)``. The backend is selected by
    ``embedding.hpo_backend``: ``"smac"`` runs SMAC3's
    ``HyperparameterOptimizationFacade`` (random forest surrogate), while
    ``"grid"`` keeps the original exhaustive grid so results predating the
    switch stay reproducible.
    """
    outcome: Any
    if settings.hpo_backend == "smac":
        from src.models.dice_smac import run_smac_search

        outcome = run_smac_search(
            dataset_dir,
            embeddings_dir,
            settings,
            seed=seed,
            train_fn=lambda data_dir, run_dir, trial_settings: (
                train_embedding_model(
                    data_dir, run_dir, trial_settings, seed=seed
                )
            ),
            score_fn=_selection_score,
        )
    else:
        outcome = grid_search(
            dataset_dir, embeddings_dir, settings, seed=seed
        )
    return (
        outcome.best_settings,
        outcome.best_report,
        outcome.trials,
        outcome.validation_error,
        outcome.best_run_dir,
    )

def export_entity_embeddings(
    run_dir: Path,
    output_path: Path,
    *,
    use_local_names: bool = True,
    expected_dim: int | None = None,
) -> Path:
    """Export DICE entity embeddings to the CSV schema NCES expects.

    NCES reads a CSV whose index is the entity name and whose columns are the
    embedding dimensions. Entity IRIs are reduced to local names because NCES
    looks individuals up by local name (see ``src/data/ontology.local_name``).

    Embeddings are read through ``dicee.KGE`` rather than by globbing for
    ``*entity_embeddings.csv``: that file is only written when
    ``save_embeddings_as_csv`` is set, which is not the default.
    """
    import pandas as pd
    from dicee import KGE

    model = KGE(path=str(run_dir))

    names, positions = _entity_index_mapping(model)
    matrix = _entity_embedding_matrix(model)[positions]

    # Not necessary anymore: NCES now reads out the CSV width.
    #if expected_dim is not None and matrix.shape[1] != expected_dim:
    #    raise ValueError(
    #        f"DICE exported {matrix.shape[1]}-dimensional entity embeddings "
    #        f"but NCES expects {expected_dim}. Multi-component models "
    #        f"(Keci, ComplEx, QMult, OMult, DualE) widen the stored matrix; "
    #        f"set embedding.embedding_dim so the exported width matches "
    #        f"nces.embedding_dim."
    #    )

    frame = pd.DataFrame(
        matrix,
        index=[str(name) for name in names],
        columns=[str(i) for i in range(matrix.shape[1])],
    )

    if use_local_names:
        frame.index = [local_name(str(name)) for name in frame.index]
        # Duplicate local names would make the entity index mapping ambiguous.
        frame = frame[~frame.index.duplicated(keep="first")]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)
    logger.info(
        "Exported %d entity embeddings (dim=%d) to %s",
        frame.shape[0],
        frame.shape[1],
        output_path,
    )
    return output_path


def generate_random_embeddings(
    entity_names: Sequence[str],
    output_path: Path,
    *,
    embedding_dim: int,
    seed: int,
) -> Path:
    """Write the deterministic random embedding baseline.

    The baseline reuses the DICE CSV schema exactly so NCES cannot tell the
    two conditions apart, isolating the contribution of trained embeddings.
    """
    import pandas as pd

    generator = np.random.default_rng(seed)
    names = [local_name(name) for name in entity_names]
    unique = list(dict.fromkeys(names))
    matrix = generator.uniform(-1.0, 1.0, size=(len(unique), embedding_dim))

    frame = pd.DataFrame(
        matrix, index=unique, columns=[str(i) for i in range(embedding_dim)]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)
    logger.info(
        "Wrote random embedding baseline (%d x %d) to %s",
        len(unique),
        embedding_dim,
        output_path,
    )
    return output_path


def build_embeddings(
    kb_path: Path,
    embeddings_dir: Path,
    data_dir: Path,
    embedding_settings: EmbeddingSettings,
    *,
    seed: int,
    embedding_conditions: Sequence[str],
    expected_dim: int,
) -> dict[str, EmbeddingResult]:
    """Run the full embedding stage for every requested condition.

    The ``dice`` condition trains the hyperparameter grid and exports the best
    trial's entity embeddings. The ``random`` condition reuses the entity
    vocabulary and the selected dimensionality but never trains.
    """
    triples = parse_triples(kb_path)
    counts = write_dicee_dataset(triples, data_dir, seed=seed)

    entity_names = sorted(
        {triple.subject for triple in triples} | {triple.object for triple in triples}
    )
    results: dict[str, EmbeddingResult] = {}
    chosen = embedding_settings

    if "dice" in embedding_conditions:
        best, report, trials, validation_error, run_dir = search_best_embedding_setting(
            data_dir, embeddings_dir, embedding_settings, seed=seed
        )
        chosen = best
        output_path = embeddings_dir / f"{best.model_name}.csv"
        export_entity_embeddings(run_dir, output_path, expected_dim=expected_dim)
        score, _ = _selection_score(report)
        results["dice"] = EmbeddingResult(
            model_name=best.model_name,
            embeddings_path=output_path,
            embedding_condition="dice",
            embedding_dim=best.embedding_dim,
            batch_size=best.batch_size,
            score=score,
            metrics={
                section: report[section]
                for section in ("Train", "Val", "Valid", "Test")
                if isinstance(report.get(section), dict)
            },
            search_trials=trials,
            validation_error=validation_error,
        )

    if "random" in embedding_conditions:
        output_path = embeddings_dir / f"{chosen.model_name}_random.csv"
        # Match the exported DICE width, not the configured dimension: some
        # models store several components per dimension.
        baseline_dim = (
            results["dice"].embedding_dim
            if "dice" in results
            else chosen.embedding_dim
        )
        generate_random_embeddings(
            entity_names,
            output_path,
            embedding_dim=baseline_dim,
            seed=seed,
        )
        results["random"] = EmbeddingResult(
            model_name=chosen.model_name,
            embeddings_path=output_path,
            embedding_condition="random",
            embedding_dim=chosen.embedding_dim,
            batch_size=chosen.batch_size,
        )

    report_path = embeddings_dir / "embedding_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "triple_counts": counts,
                "num_entities": len(entity_names),
                "embedding_conditions": {
                    name: result.to_dict() for name, result in results.items()
                },
            },
            handle,
            indent=2,
        )
    return results

def _entity_embedding_matrix(model: Any) -> np.ndarray:
    """Return the dense entity-embedding matrix from a loaded ``KGE``.

    ``KGE.model`` is a union of every DICE architecture, so the attribute
    holding the embedding table is not uniform: most models expose
    ``entity_embeddings``, but ``Shallom`` and the convolutional models do
    not. Probe the known names and fail loudly rather than assuming.
    """
    import torch

    inner = model.model
    for attribute in ("entity_embeddings", "emb_ent_real", "entity_embedding"):
        table = getattr(inner, attribute, None)
        if isinstance(table, torch.nn.Embedding):
            return table.weight.detach().cpu().numpy()

    raise AttributeError(
        f"Could not locate the entity-embedding table on "
        f"{type(inner).__name__}. Supported attribute names: "
        f"entity_embeddings, emb_ent_real, entity_embedding."
    )


def _entity_index_mapping(model: Any) -> tuple[list[str], list[int]]:
    """Return ``(entity_names, row_positions)`` for a loaded ``KGE``."""
    import pandas as pd

    mapping = model.entity_to_idx
    if isinstance(mapping, pd.DataFrame):
        # Older dicee stores the mapping as a frame with an "index" column.
        return [str(n) for n in mapping.index], mapping.iloc[:, 0].tolist()
    return [str(k) for k in mapping], list(mapping.values())

def get_csv_dimension(embeddings_path: Path) -> int:
    """Return the number of columns in the CSV file at ``embeddings_path``."""
    import pandas as pd

    frame = pd.read_csv(embeddings_path, index_col=0)
    return frame.shape[1]