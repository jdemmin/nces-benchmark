# src/models/dice.py
"""DICE dataset preparation, embedding training, search, and export."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SPLIT_RATIOS, EmbeddingSettings
from src.data.ontology import Triple, local_name, parse_triples
from src.models.hpo_search_utils import selection_score

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResultDice:
    """
    Outcome of one DICE embedding-training workflow.
    Note, this class is ``NOT`` to be confused with
    EmbeddingResult, which is a more general class
    that represents the mean result of a single
    embedding evaluated by NCES across multiple
    learning problems.
    """

    embedding_settings: EmbeddingSettings
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    search_trials: list[dict[str, Any]] = field(default_factory=list)
    validation_error: str | None = None
    embeddings_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_settings": asdict(self.embedding_settings),
            "score": self.score,
            "metrics": self.metrics,
            "search_trials": self.search_trials,
            "validation_error": self.validation_error,
            "embeddings_path": str(self.embeddings_path) if self.embeddings_path else None,
        }

#: dicee's ReadFromDisk does substring matching on full glob paths.
DICEE_RESERVED_PATH_TOKENS: tuple[str, ...] = ("train", "valid", "test")


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
        msg = (
            f"The DICE dataset directory {directory} has ancestor path "
            f"segments containing {offenders}. dicee matches these as "
            f"substrings of the full path and would route valid.txt and "
            f"test.txt into the training set. Choose a benchmark name or "
            f"output directory without these tokens."
        )
        logger.error(msg)
        raise ValueError(msg)

def stage_dicee_dataset(data_dir: Path, staging_root: Path) -> Path:
    """Copy the split files into a dicee-safe directory.

    Returns the directory to pass as ``dataset_dir``.
    """
    import shutil

    staged = staging_root / "kg"
    staged.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid", "test"):
        shutil.copyfile(data_dir / f"{name}.txt", staged / f"{name}.txt")
    logger.info(
        "Staged DICE dataset from %s to %s (train=%d, valid=%d, test=%d)",
        data_dir,
        staged,
        len(list((data_dir / "train.txt").open())),
        len(list((data_dir / "valid.txt").open())),
        len(list((data_dir / "test.txt").open())),
    )
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

    # See KNOWN_ISSUES.md.
    _assert_dicee_safe_dataset_dir(directory)

    directory.mkdir(parents=True, exist_ok=True)
    shuffled = [triple.as_tuple() for triple in triples]
    # random.Random(seed).shuffle(shuffled)

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
            score_fn=selection_score,
        )
    else:
        from src.models.dice_grid_search import grid_search
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
) -> tuple[dict[str, EmbeddingResultDice], int]:
    """Run the full embedding stage for every requested condition.

    The ``dice`` condition trains the hyperparameter grid and exports the best
    trial's entity embeddings. The ``random`` condition reuses the entity
    vocabulary and the selected dimensionality but never trains.
    """
    
    triples = parse_triples(kb_path, seed=seed)
    counts = write_dicee_dataset(triples, data_dir, seed=seed)

    entity_names = sorted(
        {triple.subject for triple in triples} | {triple.object for triple in triples}
    )
    results: dict[str, EmbeddingResultDice] = {}
    chosen = embedding_settings

    if "dice" in embedding_conditions:
        best, report, trials, validation_error, run_dir = search_best_embedding_setting(
            data_dir, embeddings_dir, embedding_settings, seed=seed
        )
        _write_json(
            {
                "best": best.to_dict(),
                "report": report,
                "validation_error": validation_error,
            },
            embeddings_dir / "best_report.json",
        )
        chosen = best
        output_path = embeddings_dir / f"{best.model_name}.csv"
        export_entity_embeddings(run_dir, output_path, expected_dim=expected_dim)
        # _cleanup_run_dir(run_dir)
        logger.info("Cleaned up DICE run directory %s after exporting embeddings", run_dir)
        score, _ = selection_score(report)
        results["dice"] = EmbeddingResultDice(
            embedding_settings=best,
            score=score,
            metrics={
                section: report[section]
                for section in ("Train", "Val", "Valid", "Test")
                if isinstance(report.get(section), dict)
            },
            search_trials=trials,
            validation_error=validation_error,
            embeddings_path=output_path,
        )
        logger.info(
            "DICE embedding completed: %d entities, dim=%d, score=%.4f, "
            "validation_error=%s",
            len(entity_names),
            best.embedding_dim,
            score,
            validation_error,
        )

    if "random" in embedding_conditions:
        output_path = embeddings_dir / f"{chosen.model_name}_random.csv"
        # Match the exported DICE width, not the configured dimension: some
        # models store several components per dimension.
        baseline_dim: int
        try:
            baseline_dim = get_csv_dimension(embeddings_dir / f"{chosen.model_name}.csv")
        except FileNotFoundError as e:
            logger.error(
                "Failed to get CSV dimension for DICE embeddings: %s. "
                "Falling back to the configured embedding dimension %d.",
                e,
                chosen.embedding_dim,
            )
            baseline_dim = expected_dim
        generate_random_embeddings(
            entity_names,
            output_path,
            embedding_dim=baseline_dim,
            seed=seed,
        )
        results["random"] = EmbeddingResultDice(
            embedding_settings=chosen,
            embeddings_path=output_path,
        )
        logger.info(
            "Random embedding baseline completed: %d entities, dim=%d",
            len(entity_names),
            baseline_dim,
        )

    report_path = embeddings_dir / "embedding_report.json"
    _write_json(
        {
            "triple_counts": counts,
            "num_entities": len(entity_names),
            "embedding_conditions": {
                name: result.to_dict() for name, result in results.items()
            },
        },
        report_path,
    )
    return results, baseline_dim

def _write_json(data: dict, path: Path) -> None:
    """Write a JSON file with indentation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")

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
    msg = (
        f"Could not locate the entity-embedding table on "
        f"{type(inner).__name__}. Supported attribute names: "
        f"entity_embeddings, emb_ent_real, entity_embedding."
    )
    logger.error(msg)
    raise AttributeError(msg)


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
    
    try:
        frame = pd.read_csv(embeddings_path, index_col=0)
    except FileNotFoundError:
        from os import listdir
        logger.error(
            f"Embeddings CSV {embeddings_path} is empty."
            f"Current content: {listdir(embeddings_path.parent)}"
        )
        raise FileNotFoundError(f"Embeddings CSV {embeddings_path} is empty.")
    return frame.shape[1]