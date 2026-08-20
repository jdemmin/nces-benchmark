# src/models/nces.py
"""NCES training-data preparation, training, and hypothesis evaluation."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from src.benchmarking.metrics import (
    calculate_metrics,
    compute_lift,
    summarize_by_complexity,
)
from src.config import NCESSettings
from src.data.lp import LearningProblem
from src.data.ontology import concept_extension

logger = logging.getLogger(__name__)


def prepare_nces_training_data(
     problems: Sequence[LearningProblem], path: Path
 ) -> list[tuple[str, dict[str, list[str]]]]:
    """Build and persist the NCES training data.

    NCES expects a sequence of ``(target_concept, {"positive examples": [...],
    "negative examples": [...]})`` tuples keyed on **local names**.
    """
    data = [problem.as_nces_datapoint() for problem in problems]
    if len(dict(data)) != len(data):
        logger.warning(
            "%d of %d learning problems share a target concept; "
            "the persisted artifact is keyed by concept and will collapse them",
            len(data) - len(dict(data)),
            len(data),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            [{"target_concept": c, **ex} for c, ex in data],
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return data


def build_nces(
    kb_path: Path,
    embeddings_path: Path,
    trained_models_dir: Path,
    settings: NCESSettings,
    m: int,
    *,
    load_pretrained: bool,
):
    """Construct an ``ontolearn.concept_learner.NCES`` instance.

    ``auto_train`` is disabled so that training is always explicit and driven
    by the benchmark's own train split.
    """
    from ontolearn.knowledge_base import KnowledgeBase
    from ontolearn.learners import NCES

    trained_models_dir.mkdir(parents=True, exist_ok=True)
    return NCES(
        knowledge_base=KnowledgeBase(path=str(kb_path)),
        path_of_embeddings=str(embeddings_path),
        path_of_trained_models=str(trained_models_dir),
        learner_names=settings.learner_names,
        quality_func=None,
        proj_dim=settings.proj_dim,
        rnn_n_layers=settings.rnn_n_layers,
        drop_prob=settings.drop_prob,
        num_heads=settings.num_heads,
        num_seeds=settings.num_seeds,
        m=m,
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
    m: int,
) -> dict[str, Any]:
    """Train NCES on the train split and save the weights.

    Upstream ``NCESTrainer.train`` initializes ``best_weights = (None, None)``
    and only replaces it when an epoch's mean *hard* accuracy is strictly
    greater than the running best, which starts at 0. When no epoch achieves
    non-zero hard accuracy -- routine for small learning-problem counts or
    short training -- it calls ``load_state_dict(None)`` and raises TypeError.
    The trained parameters are unaffected: ``train_step`` updates the module in
    place, so only the "restore the best epoch" step is lost. Catch the
    TypeError and persist the final-epoch weights instead.
    """
    model = build_nces(
        kb_path,
        embeddings_path,
        trained_models_dir,
        settings,
        load_pretrained=False,
        m=m,
    )

    logger.info(
        "Training NCES %s for %d epochs on %d learning problems",
        settings.learner_name,
        settings.epochs,
        len(train_data),
    )
    started = time.perf_counter()
    degraded = None
    try:
        model.train(
            cast(Any, list(train_data)),
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            num_workers=settings.num_workers,
            learning_rate=settings.learning_rate,
            save_model=True,
            storage_path=str(trained_models_dir),
            record_runtime=True,
        )  
    except TypeError as error:
        if "state_dict to be dict-like" not in str(error):
            raise
        degraded = (
            "Upstream NCESTrainer never recorded best weights (no epoch "
            "exceeded zero hard accuracy); persisted final-epoch weights "
            "instead of best-epoch weights."
        )
        logger.warning("%s", degraded)
        _save_final_weights(model, trained_models_dir, settings)
    runtime = time.perf_counter() - started

    return {
        "learner_name": settings.learner_name,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "num_train_problems": len(train_data),
        "runtime_seconds": round(runtime, 3),
        "degraded": degraded,
    }

def _save_final_weights(model, trained_models_dir: Path, settings: NCESSettings) -> None:
    """Write the artifacts upstream would have written after restoring weights."""
    import json as _json

    import numpy as np
    import torch

    models_dir = trained_models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    learners = model.model if isinstance(model.model, dict) else {settings.learner_name: model.model}
    for learner_name, module in learners.items():
        net = module["model"] if isinstance(module, dict) else module
        net = getattr(net, "module", net)  # unwrap DataParallel
        torch.save(
            net.state_dict(), models_dir / f"trained_{learner_name}.pt"
        )
        logger.info(_fingerprint(net))

    with (models_dir / "config.json").open("w", encoding="utf-8") as handle:
        _json.dump(
            {
                "max_length": model.max_length,
                "proj_dim": model.proj_dim,
                "num_heads": model.num_heads,
                "num_seeds": model.num_seeds,
                "rnn_n_layers": model.rnn_n_layers,
            },
            handle,
        )
    with (models_dir / "vocab.json").open("w", encoding="utf-8") as handle:
        _json.dump(model.vocab, handle)
    np.save(models_dir / "inv_vocab.npy", model.inv_vocab)


def evaluate_nces(
    kb_path: Path,
    embeddings_path: Path,
    trained_models_dir: Path,
    problems: Sequence[LearningProblem],
    settings: NCESSettings,
    m: int,
    target_extensions: Mapping[str, frozenset[str]] | None = None,
    *,
    knowledge_base,
    all_individuals: Sequence[str],
    split_name: str,
) -> dict[str, Any]:
    """Evaluate the trained NCES learner on a held-out learning-problem split.

    Each hypothesis is rendered to DL syntax, its extension is computed with
    the reasoner, and the extension is compared to the target extension.
    """
    from ontolearn.learning_problem import PosNegLPStandard
    from owlapy.class_expression import OWLClassExpression
    from owlapy.owl_individual import OWLNamedIndividual
    from owlapy.render import DLSyntaxObjectRenderer

    if not problems:
        return {"split": split_name, "results": [], "complexity_summary": {}}

    model = build_nces(
        kb_path,
        embeddings_path,
        trained_models_dir,
        settings,
        load_pretrained=True,
        m=m,
    )

    expected = trained_models_dir / f"trained_{settings.learner_name}.pt"
    if not expected.is_file():
        raise FileNotFoundError(
            f"No trained NCES weights at {expected}; evaluation would score an "
            f"untrained learner. Contents: {sorted(p.name for p in trained_models_dir.glob('*'))}"
        )

    renderer = DLSyntaxObjectRenderer()

    records: list[dict[str, Any]] = []
    for problem in problems:
        positives = {OWLNamedIndividual(iri) for iri in problem.pos_example}
        negatives = {OWLNamedIndividual(iri) for iri in problem.neg_example}
        lp = PosNegLPStandard(pos=positives, neg=negatives)

        started = time.perf_counter()
        try:
            predictions = model.fit(lp)
            # returns Union type. Expect a single OWLClassExpression, so check the type and raise if not.
            hypothesis = (model.best_hypotheses())
            if type(hypothesis) is list and len(hypothesis) > 0:
                hypothesis = hypothesis[0]
            if not (isinstance(hypothesis, OWLClassExpression) or hypothesis is None):
                raise TypeError(
                    f"Expected a single OWLClassExpression, got {type(hypothesis)}"
                )
        except Exception as error: # NoQA: BLE001
            logger.warning(
                "NCES failed on %s (%s): %s: %s",
                problem.id,
                problem.target_concept,
                type(error).__name__,
                error,
            )
            records.append(
                {
                    "id": problem.id,
                    "target_concept": problem.target_concept,
                    "hypotheses": "",
                    "complexity": problem.complexity.to_dict(),
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
            )
            continue
        del predictions

        expression = getattr(hypothesis, "concept", hypothesis)
        hypothesis_dl = renderer.render(expression) if expression else ""

        predicted = (
            concept_extension(knowledge_base, hypothesis_dl)
            if hypothesis_dl
            else frozenset()
        )
        # Reuse the extension computed during hardness annotation; the two
        # stages must agree, including when the parse fallback fired.
        if target_extensions is not None and problem.id in target_extensions:
            target = target_extensions[problem.id]
        else:
            target = concept_extension(knowledge_base, problem.target_concept)
            target = target or frozenset(problem.pos_example)

        metrics = calculate_metrics(predicted, target, all_individuals)
        runtime = time.perf_counter() - started
        negative_extension = set(all_individuals) - set(target)
        records.append(
            {
                "id": problem.id,
                "target_concept": problem.target_concept,
                "hypotheses": hypothesis_dl,
                "complexity": problem.complexity.to_dict(),
                "num_pos": problem.num_pos,
                "num_neg": problem.num_neg,
                "target_positive_count": len(target),
                "target_negative_count": len(negative_extension),
                "target_extension_size": {
                    "positive": len(target),
                    "negative": len(negative_extension),
                    "total": len(all_individuals),
                },
                "runtime_seconds": round(runtime, 3),
                **metrics.to_dict(),
                "lift": compute_lift(complexity=problem.complexity, f1=metrics.f1),
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
        "complexity_summary": summarize_by_complexity(scored),
    }


def _mean(records: list[dict[str, Any]], key: str) -> float:
    """
    Compute the mean of a numeric key in a list of dicts.
    Ignores missing keys.
    """
    values = []
    for entry in records:
        entry = entry.get(key)
        if entry is not None:
            values.append(float(entry))
    return sum(values) / len(values) if values else 0.0

def _fingerprint(net) -> float:
    return sum(float(p.detach().abs().sum()) for p in net.parameters())
