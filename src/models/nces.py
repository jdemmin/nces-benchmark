# src/models/nces.py
"""NCES training-data preparation, training, and hypothesis evaluation."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.benchmarking.metrics import aggregate_by_complexity, calculate_metrics
from src.config import NCESSettings
from src.data.lp import LearningProblem
from src.data.ontology import concept_extension, local_name

logger = logging.getLogger(__name__)


def prepare_nces_training_data(
    problems: Sequence[LearningProblem], path: Path
) -> list[tuple[str, dict[str, list[str]]]]:
    """Build and persist the NCES training data.

    NCES expects a sequence of ``(target_concept, {"positive examples": [...],
    "negative examples": [...]})`` tuples keyed on **local names**.
    """
    data = [problem.as_nces_datapoint() for problem in problems]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(data), handle, indent=2, ensure_ascii=False)
    logger.info("Prepared %d NCES training data points at %s", len(data), path)
    return data


def build_nces(
    kb_path: Path,
    embeddings_path: Path,
    trained_models_dir: Path,
    settings: NCESSettings,
    *,
    load_pretrained: bool,
):
    """Construct an ``ontolearn.concept_learner.NCES`` instance.

    ``auto_train`` is disabled so that training is always explicit and driven
    by the benchmark's own train split.
    """
    from ontolearn.concept_learner import NCES

    trained_models_dir.mkdir(parents=True, exist_ok=True)
    return NCES(
        knowledge_base_path=str(kb_path),
        path_of_embeddings=str(embeddings_path),
        path_of_trained_models=str(trained_models_dir),
        learner_names=settings.learner_names,
        quality_func=None,
        proj_dim=settings.proj_dim,
        rnn_n_layers=settings.rnn_n_layers,
        drop_prob=settings.drop_prob,
        num_heads=settings.num_heads,
        num_seeds=settings.num_seeds,
        m=settings.embedding_dim,
        max_length=settings.max_length,
        load_pretrained=load_pretrained,
        sorted_examples=settings.sorted_examples,
        verbose=0,
        num_predictions=settings.num_predictions,
        auto_train=False,
    )


def train_nces(
    kb_path: Path,
    embeddings_path: Path,
    trained_models_dir: Path,
    train_data: Sequence[tuple[str, dict[str, list[str]]]],
    settings: NCESSettings,
) -> dict[str, Any]:
    """Train NCES on the train split and save the weights.

    Returns the training summary, including runtime. NCES writes its own
    weights into ``trained_models_dir``; evaluation later reloads them with
    ``load_pretrained=True``.
    """
    model = build_nces(
        kb_path,
        embeddings_path,
        trained_models_dir,
        settings,
        load_pretrained=False,
    )

    logger.info(
        "Training NCES %s for %d epochs on %d learning problems",
        settings.learner_name,
        settings.epochs,
        len(train_data),
    )
    started = time.perf_counter()
    model.train(
        list(train_data),
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        learning_rate=settings.learning_rate,
        save_model=True,
        storage_path=str(trained_models_dir),
        record_runtime=True,
    )
    runtime = time.perf_counter() - started

    return {
        "learner_name": settings.learner_name,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "num_train_problems": len(train_data),
        "runtime_seconds": round(runtime, 3),
    }


def evaluate_nces(
    kb_path: Path,
    embeddings_path: Path,
    trained_models_dir: Path,
    problems: Sequence[LearningProblem],
    settings: NCESSettings,
    *,
    knowledge_base,
    all_individuals: Sequence[str],
    split_name: str,
) -> dict[str, Any]:
    """Evaluate the trained NCES learner on a held-out learning-problem split.

    Each hypothesis is rendered to DL syntax, its extension is computed with
    the reasoner, and the extension is compared to the target extension.
    """
    from owlapy.render import DLSyntaxObjectRenderer

    if not problems:
        return {"split": split_name, "results": [], "complexity_summary": {}}

    model = build_nces(
        kb_path,
        embeddings_path,
        trained_models_dir,
        settings,
        load_pretrained=True,
    )
    renderer = DLSyntaxObjectRenderer()

    records: list[dict[str, Any]] = []
    for problem in problems:
        positives = [local_name(iri) for iri in problem.pos_example]
        negatives = [local_name(iri) for iri in problem.neg_example]

        started = time.perf_counter()
        try:
            predictions = model.fit(positives, negatives)
            hypotheses = list(model.best_hypotheses(n=1) or [])
        except Exception as error:  # noqa: BLE001 - a single LP may fail
            logger.warning(
                "NCES failed on %s (%s): %s", problem.id, problem.target_concept, error
            )
            records.append(
                {
                    "id": problem.id,
                    "target_concept": problem.target_concept,
                    "complexity": problem.complexity,
                    "error": str(error),
                }
            )
            continue
        runtime = time.perf_counter() - started
        del predictions

        hypothesis = hypotheses[0] if hypotheses else None
        expression = getattr(hypothesis, "concept", hypothesis)
        hypothesis_dl = renderer.render(expression) if expression else ""

        predicted = (
            concept_extension(knowledge_base, hypothesis_dl)
            if hypothesis_dl
            else frozenset()
        )
        target = concept_extension(knowledge_base, problem.target_concept)
        # Fall back to the sampled positives when the target cannot be parsed.
        target = target or frozenset(problem.pos_example)

        metrics = calculate_metrics(predicted, target, all_individuals)
        negative_extension = set(all_individuals) - set(target)
        records.append(
            {
                "id": problem.id,
                "target_concept": problem.target_concept,
                "hypotheses": hypothesis_dl,
                "complexity": problem.complexity,
                "num_pos": problem.num_pos,
                "num_neg": problem.num_neg,
                "target_positive_count": len(target),
                "target_negative_count": len(negative_extension),
                "target_extension_size": {
                    "positive": len(target),
                    "negative": len(negative_extension),
                    "total": len(all_individuals),
                },
                "target_extension_overlap": {
                    "intersection": metrics.intersection,
                    "union": metrics.union,
                    "jaccard": metrics.jaccard,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                },
                "runtime_seconds": round(runtime, 3),
                **metrics.to_dict(),
            }
        )

    scored = [record for record in records if "error" not in record]
    return {
        "split": split_name,
        "num_problems": len(records),
        "num_scored": len(scored),
        "mean_f1": _mean(scored, "f1"),
        "mean_accuracy": _mean(scored, "accuracy"),
        "semantic_equivalence_rate": _mean(scored, "semantic_equivalence"),
        "results": records,
        "complexity_summary": aggregate_by_complexity(scored),
    }


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0)) for record in records]
    return sum(values) / len(values) if values else 0.0
